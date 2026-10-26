"""Turns raw network outputs into risk + uncertainty, for every method we compare.

All `unc` values are scaled to roughly [0, 1] so they are comparable across methods:
  softmax    : predictive entropy / ln2
  mcdropout / ensemble : mutual information (epistemic) / ln2
  evidential : Dirichlet vacuity K/S (K=2 classes, S = total strength) -> epistemic
For time-to-accident (TTA, seconds): mu, tta_epi, tta_alea, tta_var (= epi + alea).
"""
import math

import torch
import torch.nn.functional as F

EPS = 1e-6
LN2 = math.log(2.0)


def dirichlet_from_logits(logits):
    """alpha_k = softplus(f_k) + 1 (softplus is more stable than exp)."""
    return F.softplus(logits) + 1.0


def nig_from_raw(raw):
    """Normal-Inverse-Gamma params (Amini et al. 2020): gamma, nu>0, alpha>1, beta>0."""
    gamma = raw[..., 0]
    nu = F.softplus(raw[..., 1]) + EPS
    alpha = F.softplus(raw[..., 2]) + 1.0 + EPS
    beta = F.softplus(raw[..., 3]) + EPS
    return gamma, nu, alpha, beta


def bin_entropy(p):
    p = p.clamp(EPS, 1 - EPS)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def _single(model, x):
    out = model(x)
    if model.kind == "evidential":
        alpha = dirichlet_from_logits(out["cls"])
        S = alpha.sum(-1)
        res = {"risk": alpha[..., 1] / S, "unc": 2.0 / S}
    else:
        p = F.softmax(out["cls"], -1)[..., 1]
        res = {"risk": p, "ent": bin_entropy(p)}
    if model.tta == "nig":
        g, nu, a, b = nig_from_raw(out["reg"])
        res.update(mu=g, alea=b / (a - 1), epi=b / (nu * (a - 1)))
    elif model.tta == "gauss":
        res.update(mu=out["reg"][..., 0], alea=out["reg"][..., 1].clamp(-6, 6).exp())
    return res


@torch.no_grad()
def predict(models, x):
    """`models`: list of models. A list of length >1 is treated as samples (ensemble, or one
    dropout-active model repeated for MC-dropout). Returns dict of [B,T] tensors."""
    runs = [_single(m, x) for m in models]
    if len(runs) == 1:
        r = runs[0]
        res = {"risk": r["risk"], "unc": r["unc"] if "unc" in r else r["ent"] / LN2}
        if "mu" in r:
            epi = r.get("epi", torch.zeros_like(r["mu"]))
            res.update(mu=r["mu"], tta_epi=epi, tta_alea=r["alea"], tta_var=epi + r["alea"])
        return res
    risk = torch.stack([r["risk"] for r in runs])               # [P,B,T]
    mean_risk = risk.mean(0)
    mi = bin_entropy(mean_risk) - bin_entropy(risk).mean(0)      # mutual information
    res = {"risk": mean_risk, "unc": (mi / LN2).clamp(0, 1)}
    if "mu" in runs[0]:
        mus = torch.stack([r["mu"] for r in runs])
        epi = mus.var(0, unbiased=False)
        alea = torch.stack([r["alea"] for r in runs]).mean(0)
        res.update(mu=mus.mean(0), tta_epi=epi, tta_alea=alea, tta_var=epi + alea)
    return res
