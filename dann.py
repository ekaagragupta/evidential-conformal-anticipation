
import math

import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DomainClassifier(nn.Module):
    def __init__(self, d_model, n_domains=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, n_domains))

    def forward(self, h, n, lam):
        T = h.shape[1]
        m = (torch.arange(T, device=h.device)[None] < n[:, None]).float().unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
        return self.net(GradReverse.apply(pooled, lam))


def grl_lambda(progress):
    """Standard DANN schedule: 0 -> 1 as training progresses (progress in [0,1])."""
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
