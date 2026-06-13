"""Export the fitted Dixon-Coles model's outputs to JSON for the showcase site."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import load_results, clean_results, load_rankings, latest_rankings, latest_points
from src.calibrate import calibrate
from src.evaluate import (walk_forward_worldcups, evaluate, base_rate_probs,
                          result_outcome, alpha_backtest_curve, choose_alpha)
from src.dixon_coles import DixonColesModel
from src.simulate import monte_carlo
from wc2026_groups import GROUPS

OUT = REPO / "site" / "src" / "data"
_STAGE_COLS = ["p_r32", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"]


def ratings_payload(model, fit):
    teams = list(model.teams)
    return {
        "teams": teams,
        "attack": {t: float(model.attack[i]) for i, t in enumerate(teams)},
        "defense": {t: float(model.defense[i]) for i, t in enumerate(teams)},
        "home_adv": float(model.home_adv),
        "rho": float(model.rho),
        "max_goals": int(model.max_goals),
        "mean_defense": float(model._mean_defense),
        "fit": fit,
    }


def simulation_payload(probs, n_sims, seed):
    teams = [{"team": r["team"], **{c: float(r[c]) for c in _STAGE_COLS}}
             for _, r in probs.iterrows()]
    return {"n_sims": int(n_sims), "seed": int(seed), "teams": teams}


def groups_payload(groups):
    return {k: list(v) for k, v in groups.items()}


def parity_payload(model, pairs):
    out = []
    for home, away in pairs:
        d = model.predict_result(home, away, neutral=True)
        out.append({"home": home, "away": away, "neutral": True,
                    "home_win": d["home_win"], "draw": d["draw"],
                    "away_win": d["away_win"]})
    return out


def meta_payload(matches, model, rankings, alpha):
    teams = list(model.teams)
    net = {t: float(model.attack[i] - model.defense[i]) for i, t in enumerate(teams)}
    ranks = latest_rankings(rankings)
    rr = pd.DataFrame({"net": [net[t] for t in teams],
                       "rank": [ranks.get(t, np.nan) for t in teams]}).dropna()
    corr = float(rr["net"].corr(rr["rank"]))
    wc_years = [2018, 2022]
    preds, outs = walk_forward_worldcups(matches, wc_years, DixonColesModel, xi=0.0019,
                                         fifa_rankings=rankings, alpha=alpha)
    model_metrics = evaluate(preds, outs)
    train_out = [result_outcome(r.home_score, r.away_score)
                 for r in matches[matches.date < "2018-01-01"].itertuples()]
    base = base_rate_probs(train_out)
    base_metrics = evaluate([base] * len(outs), outs)
    return {
        "dataset": {"n_matches": int(len(matches)),
                    "n_teams": int(matches.home_team.nunique()),
                    "date_min": str(matches.date.min().date()),
                    "date_max": str(matches.date.max().date())},
        "ratings_sanity": {"fifa_rank_corr": round(corr, 3), "n_matched": int(len(rr))},
        "backtest": {"wc_years": wc_years, "n_matches": int(len(outs)),
                     "model": {k: round(v, 4) for k, v in model_metrics.items()},
                     "baseline": {k: round(v, 4) for k, v in base_metrics.items()},
                     "best_xi": 0.0019, "alpha": float(alpha)},
        "model": {"home_adv": round(float(model.home_adv), 3),
                  "rho": round(float(model.rho), 3)},
    }


def calibration_payload(alpha, curve, fifa_as_of, before_probs, after_probs, top_n=12):
    def top(df):
        ordered = df.sort_values("p_champion", ascending=False)
        return [{"team": r["team"], "p_champion": round(float(r["p_champion"]), 4)}
                for _, r in ordered.head(top_n).iterrows()]
    return {
        "prior": "fifa_total_points",
        "fifa_as_of": fifa_as_of,
        "alpha": float(alpha),
        "alpha_curve": [{"alpha": float(c["alpha"]),
                         "log_loss": round(float(c["log_loss"]), 4),
                         "rps": round(float(c["rps"]), 4)} for c in curve],
        "before_top12": top(before_probs),
        "after_top12": top(after_probs),
    }


def _write(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", path.name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matches = clean_results(load_results(REPO / "data/raw/results.csv"))
    fit_df = matches[matches.date >= "2010-01-01"]
    model = DixonColesModel().fit(fit_df, xi=0.0019)

    rankings = load_rankings(REPO / "data/raw/fifa_ranking.csv")
    points = latest_points(rankings)
    fifa_as_of = str(rankings.rank_date.max().date())
    alphas = [round(i / 10, 1) for i in range(11)]  # 0.0 .. 1.0
    curve = alpha_backtest_curve(matches, [2018, 2022], DixonColesModel, 0.0019,
                                 fifa_rankings=rankings, alphas=alphas)
    alpha = choose_alpha(curve, tol=0.01)
    print(f"chosen alpha={alpha}; curve={curve}")
    cal = calibrate(model, points, alpha)

    fit = {"min_date": "2010-01-01", "xi": 0.0019, "n_matches": int(len(fit_df))}
    _write(OUT / "ratings.json", ratings_payload(cal, fit))

    before = monte_carlo(GROUPS, model, n_sims=10000, seed=2026)
    after = monte_carlo(GROUPS, cal, n_sims=10000, seed=2026)
    _write(OUT / "simulation.json", simulation_payload(after, 10000, 2026))
    _write(OUT / "groups.json", groups_payload(GROUPS))
    pairs = [("Brazil", "Croatia"), ("Argentina", "Mexico"), ("Spain", "Uruguay"),
             ("France", "Norway"), ("Germany", "Ecuador")]
    _write(OUT / "parity_fixtures.json", parity_payload(cal, pairs))

    meta = meta_payload(matches, model, rankings, alpha)
    meta["calibration"] = calibration_payload(alpha, curve, fifa_as_of, before, after)
    _write(OUT / "meta.json", meta)
    print("done")


if __name__ == "__main__":
    main()
