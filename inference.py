"""Run models over a feature directory and gather per-video predictions (numpy)."""
import numpy as np
import torch

from data import make_loader
from uncertainty import predict


def collect(models, feat_dir, split, device, batch_size=32, mc_passes=0, loader=None):
    """models: list of AnticipationModel. mc_passes>0 -> one model repeated with dropout ACTIVE (MC-dropout)."""
    if mc_passes > 0:
        assert len(models) == 1
        models = [models[0]] * mc_passes
        models[0].train()
    else:
        for m in models:
            m.eval()
    if loader is None:
        loader, _ = make_loader(feat_dir, split, batch_size)
    P = {k: [] for k in ["risk", "unc", "mu", "tta_var", "tta_epi", "label", "toa", "step_sec", "vid", "domain"]}
    with torch.no_grad():
        for b in loader:
            res = predict(models, b["x"].to(device))
            res = {k: v.cpu().numpy() for k, v in res.items()}
            for i, n in enumerate(b["n"].tolist()):
                P["risk"].append(res["risk"][i, :n]); P["unc"].append(res["unc"][i, :n])
                if "mu" in res:
                    P["mu"].append(res["mu"][i, :n]); P["tta_var"].append(res["tta_var"][i, :n])
                    P["tta_epi"].append(res["tta_epi"][i, :n])
            P["label"] += b["label"].tolist(); P["toa"] += b["toa"].tolist()
            P["step_sec"] += b["step_sec"].tolist(); P["vid"] += b["vid"]; P["domain"] += b["domain"]
    for k in ["mu", "tta_var", "tta_epi"]:
        if not P[k]:
            del P[k]
    for m in models:
        m.eval()
    return P
