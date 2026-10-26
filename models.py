"""Temporal model on top of frozen video features.

TemporalEncoder : causal transformer (a step only sees the past, as in real deployment).
AnticipationModel : encoder + classification head (risk) + optional regression head (time-to-accident).

kind='softmax'    -> 2 logits, used by the softmax / MC-dropout / deep-ensemble baselines
kind='evidential' -> 2 logits interpreted as Dirichlet evidence (see uncertainty.py)
tta ='none' | 'gauss' (mean, log-variance) | 'nig' (gamma, nu, alpha, beta)
"""
import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, d_model=256, n_layers=2, n_heads=4, dropout=0.2, max_len=128):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        T = x.shape[1]
        h = self.inp(x) + self.pos[:, :T]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        # Videos are right-padded and attention is causal, so valid steps never attend to padding;
        # no key-padding mask is needed (and avoids all-masked-row NaNs).
        return self.norm(self.enc(h, mask=causal))


class AnticipationModel(nn.Module):
    def __init__(self, in_dim, kind="softmax", tta="none", d_model=256, n_layers=2,
                 n_heads=4, dropout=0.2, max_len=128):
        super().__init__()
        assert kind in ("softmax", "evidential") and tta in ("none", "gauss", "nig")
        self.kind, self.tta = kind, tta
        self.enc = TemporalEncoder(in_dim, d_model, n_layers, n_heads, dropout, max_len)
        self.cls = nn.Linear(d_model, 2)
        self.reg = None if tta == "none" else nn.Linear(d_model, {"gauss": 2, "nig": 4}[tta])

    def forward(self, x):
        h = self.enc(x)
        out = {"h": h, "cls": self.cls(h)}
        if self.reg is not None:
            out["reg"] = self.reg(h)
        return out


def build_model(cfg):
    return AnticipationModel(**cfg)


def save_ckpt(path, model, cfg, extra=None):
    torch.save({"cfg": cfg, "state": model.state_dict(), "extra": extra or {}}, path)


def load_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)  # our own checkpoints
    m = build_model(ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval(), ck["cfg"]
