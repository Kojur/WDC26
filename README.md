# WDC26 — World Cup 2026 Scoreline & Result Prediction

A Dixon-Coles goals model that predicts football scoreline probabilities, backtested
against past World Cups and used to Monte-Carlo-simulate the 2026 tournament.

## Setup
1. `python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt`
2. Download datasets into `data/raw/` (see `notebooks/01_eda.ipynb`):
   - `results.csv` — international match results
   - `fifa_ranking.csv` — FIFA world rankings (secondary signal)
3. Run the notebooks in order: `01_eda` → `02_model` → `03_backtest` → `04_simulation`.

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

A quick sanity check: the model's learned net strength correlates about −0.92 with the
official FIFA ranking, and the strongest sides (Brazil, Argentina, Spain, France, …) come
out on top — without ever being told the rankings.

## Known simplifications
- Cold-start for unseen teams uses the global-average rating (FIFA rank used only for the
  model sanity-check).
- Knockout seeding is a simplified high-vs-low bracket, not FIFA's exact third-place table.
- Backtest log-loss/RPS are scored on W/D/L result probabilities (a standard, practical metric).

## Future enhancements
- FIFA-rank-based cold-start prior for sparse teams.
- Streamlit app for interactive matchup predictions.
- Bayesian hierarchical (bivariate Poisson via PyMC) variant.
