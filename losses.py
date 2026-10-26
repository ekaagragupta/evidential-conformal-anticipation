"""Training targets and losses.

Targets (build_targets)
  label_mode='horizon' (default): frame label = 1 if an accident happens within `horizon` seconds.
        The risk output is then a probability of "accident within H s", which is what ECE/Brier/conformal
        calibration need to mean something.
  label_mode='weighted' (DSA-style): label 1 on every pre-accident frame of a positive video, loss weight
        exp(-seconds_to_accident / tau), so early frames matter less.
  Frames after the accident are dropped for positive videos. TTA regression only uses positive-video
  frames within `reg_horizon` seconds of the accident.

Losses
  softmax    : weighted cross-entropy
  evidential : Sensoy et al. (2018) Bayes-risk of log-likelihood + annealed KL(Dir(alpha_tilde)||Dir(1))
  gauss TTA  : heteroscedastic Gaussian NLL
  nig TTA    : Amini et al. (2020) NIG NLL + |error|*(2nu+alpha) evidence regulariser
"""
import math

import torch
import torch.nn.functional as F

from uncertainty import dirichlet_from_logits, nig_from_raw


def build_targets(n, toa, label, step_sec, T, horizon=2.5, mode="horizon", tau=1.0, reg_horizon=3.0):
    dev = n.device
    t = torch.arange(T, device=dev)[None, :].float()
    pos = (label == 1)[:, None]
    toa_f = toa[:, None].float()
    valid = (t < n[:, None]) & (~pos | (t <= toa_f))
    dt = (toa_f - t) * step_sec[:, None]                      # seconds until accident (positives)
    if mode == "horizon":
        y = (pos & (dt >= 0) & (dt <= horizon)).float()
        w = torch.ones_like(y)
    elif mode == "weighted":
        y = pos.float().expand_as(dt).clone()
        w = torch.where(pos, torch.exp(-dt.clamp_min(0) / tau), torch.ones_like(dt))
    else:
        raise ValueError(mode)
    reg_mask = (valid & pos & (dt <= reg_horizon)).float()
    return dict(y=y, w=w, valid=valid.float(), tta=dt.clamp_min(0), reg_mask=reg_mask)


def kl_dirichlet_uniform(alpha):
    """KL( Dir(alpha) || Dir(1,...,1) ), summed over classes -> [...]."""
    K = alpha.shape[-1]
    S = alpha.sum(-1, keepdim=True)
    t1 = torch.lgamma(S.squeeze(-1)) - math.lgamma(K) - torch.lgamma(alpha).sum(-1)
    t2 = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(-1)
    return t1 + t2


def evidential_cls_loss(logits, y, w, valid, kl_coef):
    alpha = dirichlet_from_logits(logits)
    y1h = F.one_hot(y.long(), 2).float()
    S = alpha.sum(-1, keepdim=True)
    nll = (y1h * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)
    alpha_tilde = y1h + (1 - y1h) * alpha                      # remove evidence for the true class
    kl = kl_dirichlet_uniform(alpha_tilde)
    return ((nll + kl_coef * kl) * w * valid).sum() / valid.sum().clamp_min(1)


def softmax_cls_loss(logits, y, w, valid):
    ce = F.cross_entropy(logits.reshape(-1, 2), y.long().reshape(-1), reduction="none").view_as(y)
    return (ce * w * valid).sum() / valid.sum().clamp_min(1)


def gauss_nll(raw, target, mask):
    mu, logvar = raw[..., 0], raw[..., 1].clamp(-6, 6)
    nll = 0.5 * (logvar + (target - mu) ** 2 * torch.exp(-logvar))
    return (nll * mask).sum() / mask.sum().clamp_min(1)


def nig_nll(raw, target, mask, lam_reg=0.01):
    gamma, nu, alpha, beta = nig_from_raw(raw)
    omega = 2 * beta * (1 + nu)
    nll = (0.5 * torch.log(math.pi / nu) - alpha * torch.log(omega)
           + (alpha + 0.5) * torch.log((target - gamma) ** 2 * nu + omega)
           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
    reg = (target - gamma).abs() * (2 * nu + alpha)
    return ((nll + lam_reg * reg) * mask).sum() / mask.sum().clamp_min(1)


def total_loss(out, tg, kind, tta, kl_coef=0.0, lam_tta=0.5, lam_reg=0.01):
    if kind == "evidential":
        loss = evidential_cls_loss(out["cls"], tg["y"], tg["w"], tg["valid"], kl_coef)
    else:
        loss = softmax_cls_loss(out["cls"], tg["y"], tg["w"], tg["valid"])
    parts = {"cls": loss.item()}
    if tta == "gauss":
        l = gauss_nll(out["reg"], tg["tta"], tg["reg_mask"])
        loss = loss + lam_tta * l
        parts["tta"] = l.item()
    elif tta == "nig":
        l = nig_nll(out["reg"], tg["tta"], tg["reg_mask"], lam_reg)
        loss = loss + lam_tta * l
        parts["tta"] = l.item()
    return loss, parts
