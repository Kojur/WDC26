# World Cup 2026 Scoreline & Result Prediction — Design Spec

**Date:** 2026-06-11
**Status:** Approved design, pending implementation plan
**Type:** Data science learning project

## 1. Goal & scope

Build an interpretable statistical model that predicts the **goals each team scores** in
an international football match, and use it to:

1. Predict **exact scoreline probabilities** for any matchup (and, derived from those, the
   win/draw/loss result probabilities and the most likely score).
2. **Backtest** the model rigorously against past World Cups using proper scoring rules.
3. **Simulate the 2026 World Cup** (Monte Carlo) to estimate each team's odds of advancing
   through each stage and winning the tournament.

This is a learning project for an intermediate Python/ML practitioner. Priorities, in order:
correctness, interpretability, and instructional value over raw predictive performance.

Non-goals (YAGNI): live data ingestion, player-level/event data (xG, passes), betting-odds
integration, deployment infrastructure. A Streamlit app and a Bayesian model variant are
explicit *stretch* goals only.

## 2. Decisions (locked in during brainstorming)

| Decision | Choice |
|----------|--------|
| Experience level | Intermediate |
| Prediction target | Exact scoreline (goals-based), which yields W/D/L for free |
| Primary dataset | Kaggle *International football results 1872–2026* (single CSV) |
| Secondary data | FIFA world rankings (sanity-check + cold-start fallback) |
| Model | Dixon-Coles (Poisson + low-score correction + time decay) |
| Scope | Model + backtest + Monte Carlo tournament simulation |

## 3. The model: Dixon-Coles

### Parameters
- Per team `i`: an **attack** strength `αᵢ` and a **defense** strength `βᵢ`.
- A global **home-advantage** term `γ` (applied only to non-neutral matches; most WC games
  are neutral and drop it).
- A **low-score correction** parameter `ρ`.
- Identifiability constraint: mean attack strength fixed to 1 (equivalently, the log-attack
  terms sum to 0).

### Expected goals (Poisson rates)
```
λ_home = exp(α_home + β_away + γ)
λ_away = exp(α_away + β_home)
```
A strong attack ⇒ large `α`; a strong defense ⇒ very negative `β` (suppresses the opponent).

### Scoreline probability
Each team's goals ~ Poisson around its λ. For an exact scoreline `(x, y)`:
```
P(x, y) = τ(x, y) · Poisson(x; λ_home) · Poisson(y; λ_away)
Poisson(k; λ) = e^(−λ) · λ^k / k!
```
`τ(x, y)` is the Dixon-Coles correction: exactly 1 except for the four lowest scores
(0-0, 1-0, 0-1, 1-1), where it adjusts probabilities via `ρ` to fix plain Poisson's
under-counting of draws.

Computing `P(x, y)` over a grid (e.g. 0–10 goals each, an 11×11 matrix) gives the full
distribution. From it:
- `P(home win)` = sum of cells where `x > y`
- `P(draw)`     = sum of the diagonal `x = y`
- `P(away win)` = sum of cells where `x < y`
- Most likely scoreline = the highest single cell

### Fitting
Maximum likelihood: minimize the **negative weighted log-likelihood** over all historical
matches with `scipy.optimize.minimize` (L-BFGS-B). Each match is weighted by a **time-decay**
factor `w = exp(−ξ · Δt)` so recent matches count more than old ones. The decay rate `ξ` is a
hyperparameter tuned by backtesting (Section 5).

## 4. Data pipeline (`src/data.py`)

- Load `results.csv` — columns: date, home_team, away_team, home_score, away_score,
  tournament, neutral.
- Parse dates; normalize team-name aliases (known gotcha — e.g. "West Germany"/"Germany",
  "USA"/"United States") so current teams join cleanly.
- Keep all history; rely on time decay rather than a hard cutoff.
- Join **FIFA world rankings** as a secondary signal: used to (a) sanity-check the model's
  learned ratings and (b) provide a cold-start fallback for teams with very few recent
  matches. The model's learned strengths remain primary.
- Persist cleaned data to `data/processed/` as parquet.

## 5. Backtesting & evaluation (`src/evaluate.py`)

**Protocol:** walk-forward. Train only on matches *before* a given World Cup, predict that
tournament's matches, roll forward. No data the model has seen is ever used for testing
(no leakage).

**Metrics:**
- **Ranked Probability Score (RPS)** — primary metric for ordered W/D/L outcomes.
- **Log-loss** on the full scoreline distribution.
- **Calibration plot** — reliability of stated probabilities (does "60%" happen ~60%?).

**Baselines to beat:** (a) always predict home/favorite, (b) pick by FIFA ranking, (c) plain
Poisson without the Dixon-Coles correction.

**Hyperparameter tuning:** choose the time-decay rate `ξ` that minimizes RPS in backtesting.

## 6. Tournament simulation (`src/simulate.py`)

Encodes the **2026 format**: 12 groups of 4 → round-robin → top 2 per group + 8 best
third-placed teams = 32 → knockout round of 32 → R16 → QF → SF → final.

**One Monte Carlo trial:**
1. Simulate every group match by drawing a random scoreline from the model's grid.
2. Tally points (3/1/0); resolve standings with goal-difference / goals-scored tiebreakers.
3. Rank the third-placed teams; select the 8 best.
4. Build the knockout bracket; simulate each knockout match. A drawn knockout match forces a
   win via a coin-flip weighted by the two teams' relative strength (stands in for extra
   time/penalties).
5. Record how far each team advanced.

Run ~20,000 trials and average ⇒ per-team `P(advance)`, `P(reach QF/SF/final)`,
`P(win the cup)`. The actual group draw is stored in `wc2026_groups.py` as plain data.

## 7. Project structure

```
WDC26/
├── data/
│   ├── raw/              # downloaded CSVs (results.csv, fifa_ranking.csv)
│   └── processed/        # cleaned data (parquet)
├── notebooks/
│   ├── 01_eda.ipynb            # explore data, sanity-check, plots
│   ├── 02_model.ipynb          # build & fit Dixon-Coles
│   ├── 03_backtest.ipynb       # evaluate on past World Cups
│   └── 04_simulation.ipynb     # Monte Carlo WC2026
├── src/
│   ├── data.py          # load, clean, normalize names, join rankings
│   ├── dixon_coles.py   # likelihood, fit, scoreline-grid prediction
│   ├── evaluate.py      # RPS, log-loss, calibration, baselines
│   └── simulate.py      # tournament format + Monte Carlo
├── tests/               # pytest unit tests
├── wc2026_groups.py     # the actual WC2026 group draw (data input)
├── requirements.txt
└── README.md
```

Principle: `src/` holds reusable, tested logic; `notebooks/` tell the story.

## 8. Tech stack

Python 3.11+, `pandas`, `numpy`, `scipy` (optimize + stats), `matplotlib`/`seaborn`,
Jupyter, `pytest`. No heavy frameworks. Dependencies pinned in `requirements.txt`.

## 9. Testing

Unit tests that catch real errors:
- Scoreline grid sums to ~1.
- Stronger team receives the higher win probability (monotonicity sanity check).
- Simulation always advances the correct *number* of teams; no team ever plays itself.
- Tiebreakers resolve deterministically.
- A fixed random seed reproduces identical simulation results.

## 10. Build order (learning milestones)

1. **EDA** — load & sanity-check data, plot goal distributions.
2. **Model** — implement & fit Dixon-Coles; verify ratings are sane (top sides rank near top).
3. **Backtest** — evaluate vs baselines; tune `ξ`.
4. **Simulate** — run WC2026; produce probability table + visualizations.
5. *(Stretch)* — Streamlit app, and/or Bayesian hierarchical (bivariate Poisson via PyMC).

## 11. Risks & known gotchas

- **Team-name normalization** is the most common silent bug; build an alias map early.
- **Sparse opponents** — some teams play rarely; time decay + FIFA-rank cold-start mitigate.
- **Format novelty** — the 48-team / best-third-placed structure is new for 2026; the
  tiebreaker and bracket-building logic need explicit tests.
- **Over-interpreting exact scorelines** — no single scoreline is "likely"; always reason
  with the distribution, not the modal score.
