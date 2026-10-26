
import math

import numpy as np


def score_curve(risk, unc, mode="risk", u_max=0.5):
    if mode == "risk":
        return risk
    if mode == "conf":                      # risk discounted by epistemic uncertainty
        return risk * (1.0 - np.clip(unc, 0, 1))
    if mode == "gate":                      # suppress warnings when too uncertain
        return np.where(unc <= u_max, risk, 0.0)
    raise ValueError(mode)


def curves_from(P, mode="risk", u_max=0.5):
    return [score_curve(r, u, mode, u_max) for r, u in zip(P["risk"], P["unc"])]


def negative_max_scores(curves, P):
    return np.array([np.max(c) for c, l in zip(curves, P["label"]) if l == 0])


def conformal_threshold(cal_neg_max, alpha):
    n = len(cal_neg_max)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")                  # too few calibration videos for this alpha: never alarm
    return float(np.sort(cal_neg_max)[k - 1])


def aci_stream(cal_neg_max, stream_neg_max, alpha=0.1, gamma=0.01):
    """Adaptive conformal inference over a stream of negative target videos (false alarms are observed).
    Returns per-step alpha_t, threshold and error (1 = false alarm)."""
    cal = np.sort(cal_neg_max)
    n = len(cal)
    a_t = alpha
    hist = []
    for s in stream_neg_max:
        if a_t <= 0:
            thr = float("inf")
        elif a_t >= 1:
            thr = -float("inf")
        else:
            q = min(1.0, (1 - a_t) * (1 + 1.0 / n))
            thr = float(np.quantile(cal, q, method="higher"))
        err = float(s > thr)
        hist.append((a_t, thr, err))
        a_t = a_t + gamma * (alpha - err)
    return np.array(hist)
