"""Fully test-time adaptation (TENT-style) for the temporal model.
Swin/transformer backbones have LayerNorm, not BatchNorm, so we adapt the LayerNorm affine parameters of the
temporal encoder by minimising the entropy of the risk prediction on unlabeled target batches.
EXPERIMENTAL: entropy minimisation can collapse to constant predictions; keep lr small, steps few, and always
report it as an ablation next to the non-adapted model."""
import copy

import torch
import torch.nn as nn

from data import make_loader
from uncertainty import bin_entropy, dirichlet_from_logits
import torch.nn.functional as F


def tent_adapt(model, feat_dir, split, device, lr=1e-4, epochs=1, batch_size=32):
    model = copy.deepcopy(model).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for m in model.enc.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad_(True)
                params.append(p)
    opt = torch.optim.Adam(params, lr=lr)
    loader, _ = make_loader(feat_dir, split, batch_size, shuffle=True)
    model.eval()                       # keep dropout off during adaptation
    for _ in range(epochs):
        for b in loader:
            x, n = b["x"].to(device), b["n"].to(device)
            logits = model(x)["cls"]
            if model.kind == "evidential":
                a = dirichlet_from_logits(logits)
                p1 = a[..., 1] / a.sum(-1)
            else:
                p1 = F.softmax(logits, -1)[..., 1]
            mask = (torch.arange(x.shape[1], device=device)[None] < n[:, None]).float()
            loss = (bin_entropy(p1) * mask).sum() / mask.sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()
