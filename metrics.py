"""Evaluation metrics (pure numpy, no torch needed).

`P` is a dict of per-video lists/arrays (see evaluate.collect):
    risk[i], unc[i], (mu[i], tta_var[i])  : arrays of length T_i
    label[i], toa[i] (step index, -1 for negatives), step_sec[i]
A "curve" is any per-step alarm score for a video (risk by default; conformal.py builds others).
"""
import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def frame_labels(n, toa, label, step_sec, horizon):
    """y_t = 1 iff accident within `horizon` s; valid excludes post-accident frames."""
    t = np.arange(n)
    if label == 0:
        return np.zeros(n), np.ones(n, bool)
    dt = (toa - t) * step_sec
    valid = t <= toa
    return ((dt >= 0) & (dt <= horizon)).astype(float), valid


def pooled_frames(P, horizon):
    ps, ys, us = [], [], []
    for r, u, l, t0, ss in zip(P["risk"], P["unc"], P["label"], P["toa"], P["step_sec"]):
        y, v = frame_labels(len(r), t0, l, ss, horizon)
        ps.append(r[v]); us.append(u[v]); ys.append(y[v])
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(us)


def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def risk_coverage(p, y, u):
    """Selective prediction: keep the most certain frames first. Returns (AURC, AUROC error-detection)."""
    err = ((p >= 0.5).astype(float) != y).astype(float)
    order = np.argsort(u, kind="stable")
    cum_risk = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    aurc = float(cum_risk.mean())
    auroc = float(roc_auc_score(err, u)) if 0 < err.sum() < len(err) else float("nan")
    return aurc, auroc


def video_ap(curves, P, delta):
    """PR-AUC at lead time `delta`: each video is observed only up to `delta` s before its accident (negatives: before clip end)."""
    s, y = [], []
    for c, l, t0, ss in zip(curves, P["label"], P["toa"], P["step_sec"]):
        if l == 1:
            k = t0 - int(math.ceil(delta / ss - 1e-9))
            if k < 0:
                continue
            s.append(np.max(c[: k + 1])); y.append(1)
        else:   # negatives are cut the same lead time before their clip end, so both classes get comparable windows
            k = len(c) - 1 - int(math.ceil(delta / ss - 1e-9))
            if k < 0:
                continue
            s.append(np.max(c[: k + 1])); y.append(0)
    if len(set(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def alarm_stats(curves, P, tau, strict=False):
    det, tta, fa = [], [], []
    for c, l, t0, ss in zip(curves, P["label"], P["toa"], P["step_sec"]):
        hit = (c > tau) if strict else (c >= tau)
        if l == 1:
            h = hit[: t0 + 1]
            det.append(float(h.any()))
            if h.any():
                tta.append((t0 - int(np.argmax(h))) * ss)
        else:
            fa.append(float(hit.any()))
    return dict(recall=float(np.mean(det)) if det else float("nan"),
                far=float(np.mean(fa)) if fa else float("nan"),
                mtta=float(np.mean(tta)) if tta else float("nan"))


def at_recall(curves, P, target_recall=0.8):
    """Pick the strictest threshold that still reaches `target_recall`; report FAR and mean TTA there."""
    m = sorted([np.max(c[: t0 + 1]) for c, l, t0 in zip(curves, P["label"], P["toa"]) if l == 1], reverse=True)
    k = max(1, int(math.ceil(target_recall * len(m))))
    tau = m[k - 1]
    out = alarm_stats(curves, P, tau)
    out["tau"] = float(tau)
    return out


def tta_metrics(P, reg_horizon=3.0, z=1.645):
    """MAE (s) and 90% interval coverage of the TTA prediction on positive pre-accident frames."""
    if "mu" not in P:
        return {}
    err, cov = [], []
    for mu, var, l, t0, ss in zip(P["mu"], P["tta_var"], P["label"], P["toa"], P["step_sec"]):
        if l != 1:
            continue
        t = np.arange(len(mu))
        true = (t0 - t) * ss
        m = (t <= t0) & (true <= reg_horizon)
        err.append(np.abs(mu[m] - true[m]))
        cov.append((np.abs(mu[m] - true[m]) <= z * np.sqrt(var[m])).astype(float))
    if not err:
        return {}
    return dict(tta_mae=float(np.concatenate(err).mean()), tta_cov90=float(np.concatenate(cov).mean()))
