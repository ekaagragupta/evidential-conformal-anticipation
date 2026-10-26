"""Train one model (one seed) on cached features.

Examples
  python train.py --feat_dir feats/dad_clean --kind softmax    --tta gauss --seed 0 --out runs/softmax/s0.pt
  python train.py --feat_dir feats/dad_clean --kind evidential --tta nig   --seed 0 --out runs/evi/s0.pt
  # optional DANN with unlabeled target features
  python train.py ... --kind evidential --tta nig --dann_dir feats/ccd_clean --out runs/evi_dann/s0.pt
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from conformal import curves_from
from data import make_loader
from dann import DomainClassifier, grl_lambda
from inference import collect
from losses import build_targets, total_loss
from metrics import video_ap
from models import build_model, save_ckpt
from utils import get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="evidential", choices=["softmax", "evidential"])
    ap.add_argument("--tta", default="nig", choices=["none", "gauss", "nig"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--label_mode", default="horizon", choices=["horizon", "weighted"])
    ap.add_argument("--horizon", type=float, default=2.5, help="seconds, label_mode=horizon")
    ap.add_argument("--tau", type=float, default=1.0, help="seconds, label_mode=weighted")
    ap.add_argument("--reg_horizon", type=float, default=3.0)
    ap.add_argument("--kl_coef", type=float, default=0.1, help="final KL weight (evidential lambda)")
    ap.add_argument("--kl_anneal", type=int, default=10, help="epochs to ramp the KL weight 0->kl_coef")
    ap.add_argument("--lam_tta", type=float, default=0.5)
    ap.add_argument("--lam_reg", type=float, default=0.01, help="NIG evidence regulariser")
    ap.add_argument("--dann_dir", default=None, help="unlabeled target feature dir -> enables DANN")
    ap.add_argument("--dann_lambda", type=float, default=0.1)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = get_device()
    tr_loader, tr_ds = make_loader(args.feat_dir, "train", args.bs, shuffle=True)
    va_loader, _ = make_loader(args.feat_dir, "val", args.bs)
    cfg = dict(in_dim=tr_ds.feat_dim, kind=args.kind, tta=args.tta, d_model=args.d_model,
               n_layers=args.n_layers, dropout=args.dropout)
    model = build_model(cfg).to(dev)
    params = list(model.parameters())

    dom = tgt_iter = None
    if args.dann_dir:
        dom = DomainClassifier(args.d_model).to(dev)
        params += list(dom.parameters())
        tgt_loader, _ = make_loader(args.dann_dir, "train", args.bs, shuffle=True)
        tgt_iter = itertools.cycle(tgt_loader)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(tr_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)

    best, step = -1.0, 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        kl_coef = args.kl_coef * min(1.0, ep / max(1, args.kl_anneal))
        logs = []
        for b in tr_loader:
            x, n = b["x"].to(dev), b["n"].to(dev)
            tg = build_targets(n, b["toa"].to(dev), b["label"].to(dev), b["step_sec"].to(dev), x.shape[1],
                               args.horizon, args.label_mode, args.tau, args.reg_horizon)
            out = model(x)
            loss, parts = total_loss(out, tg, args.kind, args.tta, kl_coef, args.lam_tta, args.lam_reg)
            if dom is not None:
                bt = next(tgt_iter)
                xt, nt = bt["x"].to(dev), bt["n"].to(dev)
                lam = grl_lambda(step / total_steps)
                ds = dom(out["h"], n, lam)
                dt = dom(model(xt)["h"], nt, lam)
                ld = (torch.nn.functional.cross_entropy(ds, torch.zeros(len(ds), dtype=torch.long, device=dev))
                      + torch.nn.functional.cross_entropy(dt, torch.ones(len(dt), dtype=torch.long, device=dev)))
                loss = loss + args.dann_lambda * ld
                parts["dom"] = ld.item()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            logs.append(parts)
        # model selection: validation PR-AUC at 1.5 s lead time
        P = collect([model], None, None, dev, loader=va_loader)
        ap15 = video_ap(curves_from(P, "risk"), P, 1.5)
        msg = {k: round(float(np.mean([l[k] for l in logs])), 4) for k in logs[0]}
        print(f"ep {ep:3d} {msg} val AP@1.5s={ap15:.4f}")
        if np.nan_to_num(ap15) > best:
            best = float(np.nan_to_num(ap15))
            save_ckpt(args.out, model, cfg, extra={"args": vars(args), "val_ap15": best})
    print("best val AP@1.5s:", round(best, 4), "->", args.out)


if __name__ == "__main__":
    main()
