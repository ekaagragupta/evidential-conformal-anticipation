
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from corruptions import corrupt_video

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def read_video(path, target_fps, size=224):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1, int(round(fps / target_fps)))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (size, size)))
        i += 1
    cap.release()
    return np.stack(frames), step


def build_backbone(device, weights="KINETICS400_V1"):
    from torchvision.models.video import Swin3D_T_Weights, swin3d_t
    model = swin3d_t(weights=Swin3D_T_Weights[weights] if weights else None)
    model.head = torch.nn.Identity()              # -> 768-d pooled feature
    return model.eval().to(device)


@torch.no_grad()
def video_features(model, frames, win, device, bs=16):
    N = len(frames)
    padded = np.concatenate([np.repeat(frames[:1], win - 1, 0), frames])      # causal windows
    out = []
    for s in range(0, N, bs):
        idx = np.arange(s, min(N, s + bs))
        clip = np.stack([padded[t:t + win] for t in idx])                      # [B,win,H,W,3]
        x = torch.from_numpy(clip).permute(0, 4, 1, 2, 3).float().div(255)     # [B,3,win,H,W]
        x = ((x - MEAN) / STD).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--corruption", default="none")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_pretrained", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(dev, None if args.no_pretrained else "KINETICS400_V1")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.meta, dtype={"video_id": str})
    rows = []
    for r in tqdm(meta.itertuples(), total=len(meta)):
        vid = re.sub(r"[^A-Za-z0-9_.-]", "_", r.video_id)
        frames, step = read_video(r.path, args.fps)
        frames = corrupt_video(frames, args.corruption, args.severity, args.seed)
        feats = video_features(model, frames, args.win, dev)
        np.save(out / f"{vid}.npy", feats.astype(np.float16))
        toa = -1 if r.label == 0 else min(len(frames) - 1, int(r.toa_frame) // step)
        rows.append(dict(video_id=vid, label=int(r.label), toa_step=toa, n_steps=len(frames),
                         step_sec=1.0 / args.fps, split=r.split, domain=r.domain))
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
    print("done:", len(rows), "videos ->", out)


if __name__ == "__main__":
    main()
