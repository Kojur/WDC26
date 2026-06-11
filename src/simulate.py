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


_STAGES = ["r32", "r16", "qf", "sf", "final", "champion"]


def simulate_tournament(groups, model, rng):
    """Simulate one tournament; return {team: furthest stage reached}.

    Seeding note: the 32 qualifiers are seeded by net strength and paired
    high-vs-low each round. This is a deliberate simplification of FIFA's
    published third-place-assignment bracket (see plan 'Deviations from spec').
    """
    winners, runners, third_rows = [], [], []
    reached = {t: "group" for g in groups.values() for t in g}
    for teams in groups.values():
        standings = simulate_group(teams, model, rng)
        order = rank_group(standings, rng)
        winners.append(order[0])
        runners.append(order[1])
        srow = standings[standings["team"] == order[2]].iloc[0]
        third_rows.append({"team": order[2], "pts": int(srow["pts"]),
                           "gd": int(srow["gd"]), "gf": int(srow["gf"])})
    best_thirds = select_best_thirds(third_rows, rng)
    qualifiers = winners + runners + best_thirds  # 12 + 12 + 8 = 32

    def net(t):
        a, d = model._params(t)
        return a - d

    bracket = sorted(qualifiers, key=net, reverse=True)
    for t in qualifiers:
        reached[t] = "r32"
    ri = 0
    while len(bracket) > 1:
        n = len(bracket)
        bracket = [simulate_knockout_match(bracket[k], bracket[n - 1 - k], model, rng)
                   for k in range(n // 2)]
        ri += 1
        for t in bracket:
            reached[t] = _STAGES[ri]
    return reached


def monte_carlo(groups, model, n_sims=20000, seed=0):
    """Run many tournaments; return per-team cumulative stage-reach probabilities."""
    rng = np.random.default_rng(seed)
    teams = [t for g in groups.values() for t in g]
    order = {s: i for i, s in enumerate(["group"] + _STAGES)}
    counts = {t: {s: 0 for s in _STAGES} for t in teams}
    for _ in range(n_sims):
        reached = simulate_tournament(groups, model, rng)
        for t in teams:
            ridx = order[reached[t]]
            for s in _STAGES:
                if order[s] <= ridx:
                    counts[t][s] += 1
    rows = [dict(team=t, **{f"p_{s}": counts[t][s] / n_sims for s in _STAGES})
            for t in teams]
    return pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)
