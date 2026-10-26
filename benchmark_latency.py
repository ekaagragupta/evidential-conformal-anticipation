"""Inference cost: single-pass evidential vs MC-dropout (30 passes) vs 5-model ensemble (RQ1 efficiency claim).
Measures the temporal head only (backbone features are shared by all methods)."""
import time

import torch

from models import build_model
from uncertainty import predict
from utils import get_device

dev = get_device()
x = torch.randn(8, 50, 768, device=dev)
cfg = dict(in_dim=768, tta="nig")
evi = build_model(dict(cfg, kind="evidential")).to(dev).eval()
sm = [build_model(dict(cfg, kind="softmax")).to(dev).eval() for _ in range(5)]


def bench(fn, reps=50):
    for _ in range(5):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.time() - t) / reps


mc_model = sm[0].train()
print(f"evidential (1 pass) : {bench(lambda: predict([evi], x)):7.2f} ms/batch")
print(f"softmax (1 pass)    : {bench(lambda: predict([sm[0].eval()], x)):7.2f} ms/batch")
print(f"MC-dropout (30)     : {bench(lambda: predict([mc_model] * 30, x)):7.2f} ms/batch")
print(f"ensemble (5)        : {bench(lambda: predict(sm, x)):7.2f} ms/batch")
