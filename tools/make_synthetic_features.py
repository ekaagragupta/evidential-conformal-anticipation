
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--T", type=int, default=50)
ap.add_argument("--shift", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
direction = np.random.default_rng(123).normal(size=a.dim); direction /= np.linalg.norm(direction)
offset = np.random.default_rng(7).normal(size=a.dim) * a.shift
rows = []
for i in range(a.n):
    lab = int(i % 2 == 0)
    x = rng.normal(size=(a.T, a.dim)) * (1 + 0.5 * a.shift)
    toa = -1
    if lab:
        toa = int(rng.integers(35, 49))
        t = np.arange(a.T)
        ramp = np.clip(1 - (toa - t) / 30.0, 0, 1) * 4.0           # grows during the last 3 s
        x += ramp[:, None] * direction[None]
    x += offset
    vid = f"v{i:04d}"
    np.save(out / f"{vid}.npy", x.astype(np.float16))
    r = rng.random()
    split = "train" if r < 0.6 else "val" if r < 0.7 else "calib" if r < 0.8 else "test"
    rows.append(dict(video_id=vid, label=lab, toa_step=toa, n_steps=a.T, step_sec=0.1, split=split, domain="synthetic"))
pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
print("wrote", out)
