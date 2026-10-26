"""Build meta.csv from a folder layout  ROOT/positive/*.mp4  and  ROOT/negative/*.mp4.

Accident time MUST come from the dataset's own annotation. Provide it with --toa_csv (columns video_id,toa_frame
in ORIGINAL-video frame indices). If omitted, positives fall back to --toa_from_end frames before the clip end
(a rough convention for clips trimmed to end just after the accident; check your dataset before using it!).
Optional --domain_csv (video_id,domain) for weather/lighting tags; otherwise domain = --domain.
Splits are stratified random: train/val/calib/test = 60/10/10/20 (seeded)."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="meta.csv")
    ap.add_argument("--toa_csv", default=None)
    ap.add_argument("--domain_csv", default=None)
    ap.add_argument("--domain", default="src")
    ap.add_argument("--toa_from_end", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", type=float, nargs=4, default=[0.6, 0.1, 0.1, 0.2])
    a = ap.parse_args()

    root = Path(a.root)
    toa_map = dict(pd.read_csv(a.toa_csv, dtype={"video_id": str}).values) if a.toa_csv else {}
    dom_map = dict(pd.read_csv(a.domain_csv, dtype={"video_id": str}).values) if a.domain_csv else {}
    rows = []
    for lab, sub in ((1, "positive"), (0, "negative")):
        for p in sorted((root / sub).glob("*.mp4")):
            vid = f"{sub[:3]}_{p.stem}"
            cap = cv2.VideoCapture(str(p)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            toa = -1
            if lab == 1:
                toa = int(toa_map.get(p.stem, n - a.toa_from_end))
            rows.append(dict(video_id=vid, path=str(p), label=lab, toa_frame=toa,
                             domain=dom_map.get(p.stem, a.domain)))
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(a.seed)
    df["split"] = ""
    names = ["train", "val", "calib", "test"]
    for lab in (0, 1):
        idx = rng.permutation(df.index[df.label == lab].to_numpy())
        cuts = (np.cumsum(a.splits) * len(idx)).astype(int)
        for name, lo, hi in zip(names, [0, *cuts[:-1]], cuts):
            df.loc[idx[lo:hi], "split"] = name
        df.loc[idx[cuts[-1]:], "split"] = "test"
    df.to_csv(a.out, index=False)
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
