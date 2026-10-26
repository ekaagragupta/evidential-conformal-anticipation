
import argparse
import time

import numpy as np
import pandas as pd

from conformal import conformal_threshold, curves_from, negative_max_scores
from inference import collect
from metrics import alarm_stats, at_recall, ece, brier, pooled_frames, risk_coverage, tta_metrics, video_ap
from models import load_ckpt
from tta import tent_adapt
from utils import get_device


def parse_method(spec):
    name, rest = spec.split("=", 1)
    kind, paths = rest.split(":", 1)
    return name, kind, paths.split(",")


def parse_dir(spec):
    d, s = spec.rsplit("@", 1)
    return d, s


def predictions(kind, models, feat_dir, split, dev, mc):
    if kind == "mcdropout":
        return collect(models[:1], feat_dir, split, dev, mc_passes=mc)
    return collect(models, feat_dir, split, dev)      # softmax/evidential: 1 model; ensemble: many


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", required=True)
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--alpha", type=float, default=0.1, help="conformal false-alarm level")
    ap.add_argument("--recall", type=float, default=0.8, help="target recall for FAR/mTTA comparison")
    ap.add_argument("--horizon", type=float, default=2.5, help="frame-label horizon for ECE/Brier/RC")
    ap.add_argument("--mc", type=int, default=30)
    ap.add_argument("--tent", action="store_true", help="also report TENT-adapted variant (single-model methods)")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()
    dev = get_device()
    cal_dir, cal_split = parse_dir(args.calib)
    rows = []

    for spec in args.method:
        name, kind, paths = parse_method(spec)
        loaded = [load_ckpt(p, dev)[0] for p in paths]
        variants = [(name, loaded)]
        if args.tent and kind in ("softmax", "evidential"):
            variants.append((name + "+tent", None))
        for vname, models in variants:
            cal = predictions(kind, models, cal_dir, cal_split, dev, args.mc) if models else None
            for espec in args.eval:
                ename, rest = espec.split("=", 1)
                fdir, split = parse_dir(rest)
                m_use = models
                if models is None:                        # TENT: adapt on the (unlabeled) target set itself
                    m_use = [tent_adapt(loaded[0], fdir, split, dev)]
                    cal = predictions(kind, m_use, cal_dir, cal_split, dev, args.mc)
                t0 = time.time()
                P = predictions(kind, m_use, fdir, split, dev, args.mc)
                row = dict(method=vname, eval=ename, n_videos=len(P["label"]),
                           sec_per_video=(time.time() - t0) / len(P["label"]))
                risk = curves_from(P, "risk")
                for d in (1.0, 1.5, 2.0):
                    row[f"AP@{d}s"] = video_ap(risk, P, d)
                f = at_recall(risk, P, args.recall)
                row.update({f"R{args.recall}_far": f["far"], f"R{args.recall}_mtta": f["mtta"]})
                p, y, u = pooled_frames(P, args.horizon)
                row.update(ECE=ece(p, y), Brier=brier(p, y))
                row["AURC"], row["AUROC_err"] = risk_coverage(p, y, u)
                row.update(tta_metrics(P))
                # conformal policies calibrated on SOURCE negatives only
                for mode in ("risk", "conf"):
                    thr = conformal_threshold(negative_max_scores(curves_from(cal, mode), cal), args.alpha)
                    s = alarm_stats(curves_from(P, mode), P, thr, strict=True)
                    row.update({f"conf[{mode}]_recall": s["recall"], f"conf[{mode}]_far": s["far"],
                                f"conf[{mode}]_mtta": s["mtta"]})
                rows.append(row)
                print(f"[{vname:>14} | {ename:>10}] AP@1.5={row['AP@1.5s']:.3f} ECE={row['ECE']:.3f} "
                      f"far@R{args.recall}={f['far']:.3f} conf[conf] far={row['conf[conf]_far']:.3f} "
                      f"recall={row['conf[conf]_recall']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nsaved {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
