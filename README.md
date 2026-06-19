# WDC26 — World Cup 2026 Scoreline & Result Prediction

**🔗 Live showcase: https://kojur.github.io/WDC26/**

A Dixon-Coles goals model that predicts football scoreline probabilities, backtested
against past World Cups and used to Monte-Carlo-simulate the 2026 tournament.

## Setup
1. `python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt`
2. Download datasets into `data/raw/` (see `notebooks/01_eda.ipynb`):
   - `results.csv` — international match results
   - `fifa_ranking.csv` — FIFA world rankings (secondary signal)
3. (Optional) `python scripts/refresh_fifa_snapshot.py` appends the 11 June 2026
   pre-tournament FIFA release used for the live 2026 calibration (the Kaggle export
   ends mid-2024).
4. Run the notebooks in order: `01_eda` → `02_model` → `03_backtest` → `04_simulation`.

## Layout
- `src/` — tested core logic (data, model, evaluation, simulation).
- `notebooks/` — the narrative: EDA, fitting, backtest, simulation.
- `tests/` — `pytest` unit tests (run `python -m pytest`).
- `wc2026_groups.py` — the official 2026 group draw.

## Method
Each team has an attack and a defense rating; match goals are modeled as Poisson with the
Dixon-Coles low-score correction and time-decay weighting, fit by maximum likelihood. The
scoreline grid yields exact-score, draw, and win/loss probabilities, which drive a Monte
Carlo simulation of the 48-team tournament.

The model's results-only net strength already correlates about −0.92 with the official
FIFA ranking. On top of that we apply a light **FIFA calibration**: a results-only fit
over-rates regions that mostly play among themselves (South America especially), so each
team's strength is blended toward the FIFA ranking by a weight chosen on the walk-forward
backtest. This anchors the confederations to each other and gives a sensible set of title
contenders (Brazil, Argentina, France, Spain, England, …).

## Known simplifications
- A light FIFA-points calibration is blended into the learned ratings (weight tuned on the
  backtest) to correct the closed-pool over-rating of South America. The FIFA snapshot used
  is the June 2026 pre-tournament release. Cold-start for unseen teams uses the
  global-average rating.
- Knockout seeding is a simplified high-vs-low bracket, not FIFA's exact third-place table.
- Backtest log-loss/RPS are scored on W/D/L result probabilities (a standard, practical metric).

## Future enhancements
- FIFA-rank-based cold-start prior for sparse teams.
- Streamlit app for interactive matchup predictions.
- Bayesian hierarchical (bivariate Poisson via PyMC) variant.
