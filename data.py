"""Dataset over cached per-video features.

A feature directory (written by extract_features.py) contains
    index.csv   : video_id,label,toa_step,n_steps,step_sec,split,domain
    <video_id>.npy : float16 array [T, D], one feature vector per time step.
Every model in this project trains on these cached features, so experiments are cheap.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class FeatDataset(Dataset):
    def __init__(self, feat_dir, split=None):
        self.dir = Path(feat_dir)
        df = pd.read_csv(self.dir / "index.csv", dtype={"video_id": str})
        if split is not None:
            df = df[df["split"] == split]
        if len(df) == 0:
            raise ValueError(f"No videos for split={split!r} in {feat_dir}")
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = np.load(self.dir / f"{r.video_id}.npy").astype(np.float32)
        return dict(x=torch.from_numpy(x), label=int(r.label), toa=int(r.toa_step),
                    step_sec=float(r.step_sec), vid=str(r.video_id), domain=str(r.domain))

    @property
    def feat_dim(self):
        return self[0]["x"].shape[1]


def collate(batch):
    """Right-pad to the longest video. `n` holds true lengths."""
    B = len(batch)
    T = max(b["x"].shape[0] for b in batch)
    D = batch[0]["x"].shape[1]
    x = torch.zeros(B, T, D)
    n = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["x"].shape[0]
        x[i, :L] = b["x"]
        n[i] = L
    return dict(
        x=x, n=n,
        label=torch.tensor([b["label"] for b in batch]),
        toa=torch.tensor([b["toa"] for b in batch]),
        step_sec=torch.tensor([b["step_sec"] for b in batch]),
        vid=[b["vid"] for b in batch], domain=[b["domain"] for b in batch],
    )


def make_loader(feat_dir, split, batch_size=32, shuffle=False, num_workers=0):
    ds = FeatDataset(feat_dir, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      num_workers=num_workers, drop_last=False), ds
