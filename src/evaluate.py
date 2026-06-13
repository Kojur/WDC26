"""Proper scoring rules, baselines, calibration, and walk-forward backtesting."""
import numpy as np
import pandas as pd
from src.calibrate import calibrate
from src.data import latest_points


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


def base_rate_probs(train_outcomes):
    """[P(home), P(draw), P(away)] from training-set base rates (Laplace-smoothed)."""
    counts = np.bincount(np.asarray(train_outcomes), minlength=3) + 1
    return (counts / counts.sum()).tolist()


def rank_baseline_probs(home_rank, away_rank):
    """Coarse baseline: the better-ranked (lower number) side is favored."""
    if home_rank < away_rank:
        return [0.50, 0.27, 0.23]
    if home_rank > away_rank:
        return [0.23, 0.27, 0.50]
    return [0.36, 0.28, 0.36]


def calibration_curve(home_probs, home_won, n_bins=10):
    """Bin predicted home-win probs; return (mean predicted, observed freq) per bin."""
    hp = np.asarray(home_probs, dtype=float)
    hw = np.asarray(home_won, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(hp, edges) - 1, 0, n_bins - 1)
    centers, freqs = [], []
    for b in range(n_bins):
        m = which == b
        if m.sum() > 0:
            centers.append(hp[m].mean())
            freqs.append(hw[m].mean())
    return np.array(centers), np.array(freqs)


def walk_forward_worldcups(matches, wc_years, model_factory, xi,
                           fifa_rankings=None, alpha=1.0):
    """Train on all matches before each World Cup year; predict that tournament.

    Returns (pred_probs, outcomes) as parallel lists across all predicted matches.
    """
    preds, outs = [], []
    for year in wc_years:
        cutoff = pd.Timestamp(year, 1, 1)
        train = matches[matches["date"] < cutoff]
        test = matches[(matches["tournament"] == "FIFA World Cup")
                       & (matches["date"].dt.year == year)]
        if len(train) == 0 or len(test) == 0:
            continue
        model = model_factory().fit(train, xi=xi, ref_date=cutoff)
        if fifa_rankings is not None and alpha < 1.0:
            points = latest_points(fifa_rankings, as_of=cutoff)
            model = calibrate(model, points, alpha)
        for _, row in test.iterrows():
            d = model.predict_result(row["home_team"], row["away_team"],
                                     neutral=bool(row["neutral"]))
            preds.append([d["home_win"], d["draw"], d["away_win"]])
            outs.append(result_outcome(row["home_score"], row["away_score"]))
    return preds, outs


def alpha_backtest_curve(matches, wc_years, model_factory, xi,
                         fifa_rankings, alphas):
    """Walk-forward backtest at each alpha; return list of {alpha, log_loss, rps}.

    Note: re-fits the model per (alpha, world-cup-year); for the production grid
    (11 alphas x 2 cups) this is a few minutes offline.
    """
    out = []
    for a in alphas:
        preds, outs = walk_forward_worldcups(
            matches, wc_years, model_factory, xi,
            fifa_rankings=fifa_rankings, alpha=a)
        m = evaluate(preds, outs)
        out.append({"alpha": float(a), "log_loss": m["log_loss"], "rps": m["rps"]})
    return out


def choose_alpha(curve, tol=0.0):
    """Pick alpha minimizing log-loss; among alphas within `tol` of the best (a
    statistically indifferent band), prefer the smaller alpha (more FIFA
    correction). The (alpha, rps) key is lexicographic, so alpha dominates: RPS
    only differentiates identical alpha values, which a normal distinct-alpha
    grid never produces. Net rule: smallest alpha in the indifferent band.
    """
    if not curve:
        raise ValueError("curve must be non-empty")
    best = min(c["log_loss"] for c in curve)
    band = [c for c in curve if c["log_loss"] <= best + tol]
    return min(band, key=lambda c: (c["alpha"], c["rps"]))["alpha"]
