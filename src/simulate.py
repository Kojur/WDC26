"""Monte Carlo simulation of the 2026 World Cup (12 groups of 4)."""
import numpy as np
import pandas as pd

from src.evaluate import result_outcome

_POINTS = {0: (3, 0), 1: (1, 1), 2: (0, 3)}  # outcome -> (home_pts, away_pts)


def simulate_group(teams, model, rng):
    """Round-robin a 4-team group; return a standings DataFrame."""
    stats = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            h, a = teams[i], teams[j]
            hg, ag = model.sample_scoreline(h, a, rng, neutral=True)
            stats[h]["gf"] += hg; stats[h]["ga"] += ag
            stats[a]["gf"] += ag; stats[a]["ga"] += hg
            hp, ap = _POINTS[result_outcome(hg, ag)]
            stats[h]["pts"] += hp; stats[a]["pts"] += ap
    rows = [{"team": t, "pts": s["pts"], "gd": s["gf"] - s["ga"], "gf": s["gf"]}
            for t, s in stats.items()]
    return pd.DataFrame(rows)


def rank_group(standings, rng):
    """Order teams by points, then goal difference, goals for, then random."""
    df = standings.copy()
    df["_rand"] = rng.random(len(df))
    df = df.sort_values(["pts", "gd", "gf", "_rand"], ascending=False)
    return df["team"].tolist()


def select_best_thirds(third_rows, rng):
    """Pick the 8 best third-placed teams across the 12 groups."""
    df = pd.DataFrame(third_rows)
    df["_rand"] = rng.random(len(df))
    df = df.sort_values(["pts", "gd", "gf", "_rand"], ascending=False)
    return df["team"].head(8).tolist()


def simulate_knockout_match(home, away, model, rng):
    """Simulate one knockout match; a draw is resolved by a strength-weighted flip."""
    hg, ag = model.sample_scoreline(home, away, rng, neutral=True)
    if hg > ag:
        return home
    if ag > hg:
        return away
    a_h, _ = model._params(home)
    a_a, _ = model._params(away)
    p_home = np.exp(a_h) / (np.exp(a_h) + np.exp(a_a))
    return home if rng.random() < p_home else away
