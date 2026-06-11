"""Proper scoring rules, baselines, calibration, and walk-forward backtesting."""
import numpy as np
import pandas as pd


def result_outcome(home_goals, away_goals):
    """0 = home win, 1 = draw, 2 = away win."""
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def rps(probs, outcome):
    """Ranked Probability Score for an ordered 3-outcome forecast (lower better)."""
    p = np.asarray(probs, dtype=float)
    o = np.zeros(p.size)
    o[outcome] = 1.0
    cp, co = np.cumsum(p), np.cumsum(o)
    return float(np.sum((cp[:-1] - co[:-1]) ** 2) / (p.size - 1))


def log_loss(probs, outcome, eps=1e-15):
    return float(-np.log(np.clip(probs[outcome], eps, 1.0)))


def evaluate(pred_probs, outcomes):
    rps_v = [rps(p, o) for p, o in zip(pred_probs, outcomes)]
    ll_v = [log_loss(p, o) for p, o in zip(pred_probs, outcomes)]
    acc = np.mean([int(np.argmax(p) == o) for p, o in zip(pred_probs, outcomes)])
    return {"rps": float(np.mean(rps_v)),
            "log_loss": float(np.mean(ll_v)),
            "accuracy": float(acc)}
