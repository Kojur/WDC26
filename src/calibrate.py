"""Blend a fitted Dixon-Coles model's net strength toward an external prior.

The MLE fit is left untouched; this is a transparent post-fit adjustment that
shrinks each team's net strength (attack - defense) toward a standardized FIFA
prior, splitting the change evenly between attack and defense.
"""
import copy

import numpy as np


def calibrate(model, fifa_points, alpha):
    """Return a calibrated *copy* of ``model``; the input is not mutated.

    alpha = 1.0 -> identity (pure model).
    alpha = 0.0 -> net ordering follows the FIFA prior on the reference set.

    Reference set = teams present in both the model and ``fifa_points``. Teams
    without a FIFA entry keep their model ratings unchanged.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    cal = copy.deepcopy(model)
    idx = cal.team_index
    ref = [t for t in cal.teams if t in fifa_points]
    if len(ref) < 2:
        return cal
    net = np.array([cal.attack[idx[t]] - cal.defense[idx[t]] for t in ref])
    pts = np.array([float(fifa_points[t]) for t in ref])
    net_mean, net_std = net.mean(), net.std()
    pts_mean, pts_std = pts.mean(), pts.std()
    if net_std == 0 or pts_std == 0:
        return cal
    z_model = (net - net_mean) / net_std
    z_fifa = (pts - pts_mean) / pts_std
    z_final = alpha * z_model + (1.0 - alpha) * z_fifa
    net_final = net_mean + net_std * z_final
    delta = net_final - net
    for t, d in zip(ref, delta):
        i = idx[t]
        cal.attack[i] += d / 2.0
        cal.defense[i] -= d / 2.0
    return cal
