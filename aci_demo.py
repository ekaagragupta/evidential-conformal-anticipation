
import argparse

import numpy as np

from conformal import aci_stream, conformal_threshold, curves_from, negative_max_scores
from evaluate import parse_dir, parse_method, predictions
from models import load_ckpt
from utils import get_device

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True)
ap.add_argument("--calib", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--mode", default="conf")
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--gamma", type=float, default=0.02)
ap.add_argument("--mc", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = get_device()
name, kind, paths = parse_method(a.method)
models = [load_ckpt(p, dev)[0] for p in paths]
cal = predictions(kind, models, *parse_dir(a.calib), dev, a.mc)
tgt = predictions(kind, models, *parse_dir(a.target), dev, a.mc)
cal_max = negative_max_scores(curves_from(cal, a.mode), cal)
tgt_max = negative_max_scores(curves_from(tgt, a.mode), tgt)
tgt_max = np.random.default_rng(a.seed).permutation(tgt_max)

thr = conformal_threshold(cal_max, a.alpha)
static_far = float(np.mean(tgt_max > thr))
h = aci_stream(cal_max, tgt_max, a.alpha, a.gamma)
print(f"target false-alarm level alpha      : {a.alpha}")
print(f"static conformal FAR on shifted set : {static_far:.3f}")
print(f"ACI long-run FAR on shifted stream  : {h[:, 2].mean():.3f}  (final alpha_t={h[-1, 0]:.3f})")
