# World Cup 2026 Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interpretable Dixon-Coles goals model that predicts football scoreline probabilities, backtest it against past World Cups, and Monte-Carlo-simulate the 2026 tournament.

**Architecture:** Pure-Python statistical core in `src/` (data cleaning → maximum-likelihood Dixon-Coles fit → scoreline grid → proper-scoring backtest → Monte Carlo simulation), driven by four narrative notebooks. All logic is unit-tested against small synthetic fixtures so tests run in seconds without the real dataset.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy (`optimize`, `stats`), matplotlib/seaborn, Jupyter, pytest.

---

## Spec reference

Design spec: `docs/superpowers/specs/2026-06-11-world-cup-2026-prediction-design.md`. Read it first.

## Deviations from spec (deliberate, scoped-down)

1. **Cold-start for sparse/unseen teams** uses the model's **global-average** attack/defense rather than a FIFA-rank-derived prior. FIFA rankings are still loaded and used for the model **sanity-check** (notebook 02). Rank-based cold-start is listed under "Future enhancements."
2. **Knockout bracket seeding** uses a documented simplified seeding (qualifiers ranked by net strength, standard high-vs-low pairing) rather than FIFA's exact published third-place-assignment table. The group stage and best-third selection follow the real 2026 format exactly.

These keep the core tight and fully testable; both are flagged in code comments and the README.

## File structure

```
WDC26/
├── data/raw/            # downloaded CSVs (gitignored)
├── data/processed/      # cleaned parquet (gitignored)
├── src/
│   ├── __init__.py
│   ├── data.py          # load_results, clean_results, load_rankings, latest_rankings
│   ├── dixon_coles.py   # dc_tau, _vec_tau, _neg_log_likelihood, DixonColesModel
│   ├── evaluate.py      # result_outcome, rps, log_loss, evaluate, baselines, calibration, walk_forward
│   └── simulate.py      # simulate_group, rank_group, select_best_thirds, knockout, tournament, monte_carlo
├── tests/
│   ├── __init__.py
│   ├── conftest.py      # synthetic_matches + fitted_model + StubModel fixtures
│   ├── test_data.py
│   ├── test_dixon_coles.py
│   ├── test_evaluate.py
│   └── test_simulate.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model.ipynb
│   ├── 03_backtest.ipynb
│   └── 04_simulation.ipynb
├── wc2026_groups.py     # GROUPS dict (official draw, transcribed)
├── requirements.txt
└── README.md
```

---

## Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `pytest.ini`

- [ ] **Step 1: Write `requirements.txt`**

```
pandas>=2.0
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
pyarrow>=12.0
pytest>=7.0
```

- [ ] **Step 2: Create empty package markers**

`src/__init__.py`:
```python
```
`tests/__init__.py`:
```python
```

- [ ] **Step 3: Write `pytest.ini`** (so `from src...` imports resolve from repo root)

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 4: Create a virtual env and install**

Run:
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```
Expected: installs complete without error. (On PowerShell use `.venv\Scripts\python`.)

- [ ] **Step 5: Verify pytest collects nothing yet**

Run: `.venv/Scripts/python -m pytest -q`
Expected: "no tests ran" (exit OK).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/__init__.py tests/__init__.py pytest.ini
git commit -m "chore: project scaffolding and dependencies"
```

---

## Task 2: Test fixtures (`tests/conftest.py`)

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the fixtures** (synthetic data where "Strong" > "Medium" > "Weak", plus a simulation stub)

```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_matches():
    """300 friendly matches among three teams of known relative strength."""
    rng = np.random.default_rng(0)
    teams = ["Strong", "Medium", "Weak"]
    base = {"Strong": 1.9, "Medium": 1.1, "Weak": 0.6}
    dates = pd.date_range("2010-01-01", periods=300, freq="W")
    rows = []
    for d in dates:
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append({
            "date": d, "home_team": h, "away_team": a,
            "home_score": int(rng.poisson(base[h])),
            "away_score": int(rng.poisson(base[a])),
            "tournament": "Friendly", "neutral": True,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_model(synthetic_matches):
    from src.dixon_coles import DixonColesModel
    return DixonColesModel().fit(synthetic_matches, xi=0.0)


class StubModel:
    """Deterministic-strength model for simulation tests (no fitting needed)."""
    def __init__(self, strengths):
        self.strengths = strengths

    def _params(self, team):
        return self.strengths.get(team, 0.0), 0.0

    def sample_scoreline(self, home, away, rng, neutral=True):
        hg = rng.poisson(max(0.1, 1.0 + self.strengths.get(home, 0.0)))
        ag = rng.poisson(max(0.1, 1.0 + self.strengths.get(away, 0.0)))
        return int(hg), int(ag)


@pytest.fixture
def stub_model_factory():
    return StubModel
```

- [ ] **Step 2: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add synthetic match fixtures and simulation stub"
```

---

## Task 3: Data loading & cleaning (`src/data.py`)

**Files:**
- Create: `src/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from src.data import clean_results, DEFAULT_ALIASES


def test_clean_results_normalizes_aliases_and_types():
    raw = pd.DataFrame({
        "date": ["1990-06-08", "2002-06-01"],
        "home_team": ["West Germany", "USA"],
        "away_team": ["USA", "Korea Republic"],
        "home_score": [4.0, 1.0],
        "away_score": [1.0, 1.0],
        "tournament": ["FIFA World Cup", "FIFA World Cup"],
        "neutral": [True, True],
        "extra_col": ["x", "y"],
    })
    out = clean_results(raw)
    assert list(out.columns) == [
        "date", "home_team", "away_team", "home_score",
        "away_score", "tournament", "neutral",
    ]
    assert out.loc[0, "home_team"] == "Germany"
    assert out.loc[0, "away_team"] == "United States"
    assert out.loc[1, "away_team"] == "South Korea"
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out["home_score"].dtype == int
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data'`.

- [ ] **Step 3: Implement `src/data.py`**

```python
"""Load and clean the international results dataset and FIFA rankings."""
import pandas as pd

DEFAULT_ALIASES = {
    "West Germany": "Germany",
    "East Germany": "Germany",
    "USA": "United States",
    "Soviet Union": "Russia",
    "Czechoslovakia": "Czechia",
    "Yugoslavia": "Serbia",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
}

_COLS = ["date", "home_team", "away_team", "home_score",
         "away_score", "tournament", "neutral"]


def load_results(path):
    """Read results.csv with parsed dates."""
    return pd.read_csv(path, parse_dates=["date"])


def clean_results(df, aliases=None):
    """Normalize team names, coerce types, drop incomplete rows, select columns."""
    aliases = DEFAULT_ALIASES if aliases is None else aliases
    out = df.copy()
    out["home_team"] = out["home_team"].replace(aliases)
    out["away_team"] = out["away_team"].replace(aliases)
    if not pd.api.types.is_datetime64_any_dtype(out["date"]):
        out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=["home_score", "away_score"])
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)
    out["neutral"] = out["neutral"].astype(bool)
    return out[_COLS].reset_index(drop=True)


def load_rankings(path):
    """Read the FIFA ranking CSV (Kaggle 'cashncarry/fifaworldranking')."""
    return pd.read_csv(path, parse_dates=["rank_date"])


def latest_rankings(rankings, as_of=None):
    """Return {country: rank} from the most recent snapshot on/before as_of."""
    df = rankings if as_of is None else rankings[rankings["rank_date"] <= as_of]
    snap = df[df["rank_date"] == df["rank_date"].max()]
    return dict(zip(snap["country_full"], snap["rank"]))
```

Note: adjust `country_full`/`rank`/`rank_date` to the actual column names in your downloaded ranking file if they differ.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_data.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: data loading, cleaning, and team-name normalization"
```

---

## Task 4: Dixon-Coles primitives (`dc_tau`, `_vec_tau`)

**Files:**
- Create: `src/dixon_coles.py`
- Test: `tests/test_dixon_coles.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.dixon_coles import dc_tau, _vec_tau


def test_dc_tau_is_one_for_high_scores():
    assert dc_tau(3, 2, 1.5, 1.1, -0.05) == 1.0
    assert dc_tau(2, 2, 1.5, 1.1, -0.05) == 1.0


def test_dc_tau_adjusts_the_four_low_scores():
    rho = -0.05
    assert dc_tau(0, 0, 1.5, 1.1, rho) == 1.0 - 1.5 * 1.1 * rho
    assert dc_tau(0, 1, 1.5, 1.1, rho) == 1.0 + 1.5 * rho
    assert dc_tau(1, 0, 1.5, 1.1, rho) == 1.0 + 1.1 * rho
    assert dc_tau(1, 1, 1.5, 1.1, rho) == 1.0 - rho


def test_vec_tau_matches_scalar():
    x = np.array([0, 0, 1, 1, 3])
    y = np.array([0, 1, 0, 1, 2])
    lam = np.full(5, 1.5)
    mu = np.full(5, 1.1)
    vec = _vec_tau(x, y, lam, mu, -0.05)
    scalar = np.array([dc_tau(a, b, 1.5, 1.1, -0.05) for a, b in zip(x, y)])
    np.testing.assert_allclose(vec, scalar)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_dixon_coles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.dixon_coles'`.

- [ ] **Step 3: Implement the primitives**

```python
"""Dixon-Coles bivariate-Poisson model for football scoreline prediction."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def dc_tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction (scalar)."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _vec_tau(x, y, lam, mu, rho):
    """Vectorized Dixon-Coles correction over arrays of scores/rates."""
    tau = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return tau
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_dixon_coles.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dixon_coles.py tests/test_dixon_coles.py
git commit -m "feat: Dixon-Coles low-score correction (scalar + vectorized)"
```

---

## Task 5: Negative log-likelihood + model fit

**Files:**
- Modify: `src/dixon_coles.py` (append)
- Test: `tests/test_dixon_coles.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_fit_recovers_strength_ordering(fitted_model):
    m = fitted_model
    i_strong = m.team_index["Strong"]
    i_weak = m.team_index["Weak"]
    assert m.attack[i_strong] > m.attack[i_weak]
    assert abs(m.attack.mean()) < 1e-6  # identifiability: mean attack == 0
    assert isinstance(m.home_adv, float)


def test_fit_sets_all_teams(fitted_model):
    assert set(fitted_model.teams) == {"Strong", "Medium", "Weak"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_dixon_coles.py::test_fit_recovers_strength_ordering -v`
Expected: FAIL — `AttributeError`/`ImportError` for `DixonColesModel`.

- [ ] **Step 3: Append the NLL function and the model class with `fit`**

```python
def _neg_log_likelihood(params, hi, ai, x, y, neutral, weights, n_teams):
    attack = params[:n_teams]
    defense = params[n_teams:2 * n_teams]
    gamma = params[2 * n_teams]
    rho = params[2 * n_teams + 1]
    home_term = gamma * (~neutral)
    lam = np.exp(attack[hi] + defense[ai] + home_term)
    mu = np.exp(attack[ai] + defense[hi])
    tau = np.clip(_vec_tau(x, y, lam, mu, rho), 1e-10, None)
    ll = weights * (np.log(tau) + poisson.logpmf(x, lam) + poisson.logpmf(y, mu))
    return -float(np.sum(ll))


class DixonColesModel:
    """Maximum-likelihood Dixon-Coles model with time-decay weighting."""

    def __init__(self, max_goals=10):
        self.max_goals = max_goals
        self.teams = None
        self.team_index = None
        self.attack = None
        self.defense = None
        self.home_adv = None
        self.rho = None
        self._mean_defense = None

    def fit(self, matches, xi=0.0, ref_date=None):
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = matches["home_team"].map(idx).to_numpy()
        ai = matches["away_team"].map(idx).to_numpy()
        x = matches["home_score"].to_numpy().astype(int)
        y = matches["away_score"].to_numpy().astype(int)
        neutral = matches["neutral"].to_numpy().astype(bool)
        ref_date = matches["date"].max() if ref_date is None else ref_date
        age = (ref_date - matches["date"]).dt.days.to_numpy() / 365.25
        weights = np.exp(-xi * age)

        p0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])
        res = minimize(
            _neg_log_likelihood, p0,
            args=(hi, ai, x, y, neutral, weights, n),
            method="L-BFGS-B",
        )
        att = res.x[:n]
        dfn = res.x[n:2 * n]
        c = att.mean()                 # recenter for identifiability
        att, dfn = att - c, dfn + c
        self.teams = teams
        self.team_index = idx
        self.attack = att
        self.defense = dfn
        self.home_adv = float(res.x[2 * n])
        self.rho = float(res.x[2 * n + 1])
        self._mean_defense = float(dfn.mean())
        self.result_ = res
        return self

    def _params(self, team):
        if team in self.team_index:
            i = self.team_index[team]
            return float(self.attack[i]), float(self.defense[i])
        return 0.0, self._mean_defense  # global-average cold-start fallback
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_dixon_coles.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add src/dixon_coles.py tests/test_dixon_coles.py
git commit -m "feat: weighted negative log-likelihood and Dixon-Coles fit"
```

---

## Task 6: Prediction — expected goals, scoreline grid, result

**Files:**
- Modify: `src/dixon_coles.py` (append methods to `DixonColesModel`)
- Test: `tests/test_dixon_coles.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_score_matrix_sums_to_one(fitted_model):
    mat = fitted_model.score_matrix("Strong", "Weak", neutral=True)
    assert mat.shape == (11, 11)
    np.testing.assert_allclose(mat.sum(), 1.0, atol=1e-9)


def test_predict_result_sums_to_one_and_favors_strong(fitted_model):
    p = fitted_model.predict_result("Strong", "Weak", neutral=True)
    np.testing.assert_allclose(p["home_win"] + p["draw"] + p["away_win"], 1.0, atol=1e-9)
    assert p["home_win"] > p["away_win"]


def test_expected_goals_higher_for_strong(fitted_model):
    lam, mu = fitted_model.expected_goals("Strong", "Weak", neutral=True)
    assert lam > mu


def test_unseen_team_uses_fallback(fitted_model):
    lam, mu = fitted_model.expected_goals("Strong", "Atlantis", neutral=True)
    assert lam > 0 and mu > 0  # no KeyError; fallback applied
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_dixon_coles.py::test_score_matrix_sums_to_one -v`
Expected: FAIL — `AttributeError: 'DixonColesModel' object has no attribute 'score_matrix'`.

- [ ] **Step 3: Append the prediction methods**

```python
    def expected_goals(self, home, away, neutral=True):
        a_h, d_h = self._params(home)
        a_a, d_a = self._params(away)
        gamma = 0.0 if neutral else self.home_adv
        lam = float(np.exp(a_h + d_a + gamma))
        mu = float(np.exp(a_a + d_h))
        return lam, mu

    def score_matrix(self, home, away, neutral=True):
        lam, mu = self.expected_goals(home, away, neutral)
        g = np.arange(self.max_goals + 1)
        mat = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        mat[0, 0] *= 1.0 - lam * mu * self.rho
        mat[0, 1] *= 1.0 + lam * self.rho
        mat[1, 0] *= 1.0 + mu * self.rho
        mat[1, 1] *= 1.0 - self.rho
        return mat / mat.sum()

    def predict_result(self, home, away, neutral=True):
        mat = self.score_matrix(home, away, neutral)
        return {
            "home_win": float(np.tril(mat, -1).sum()),  # home goals > away goals
            "draw": float(np.trace(mat)),
            "away_win": float(np.triu(mat, 1).sum()),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_dixon_coles.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dixon_coles.py tests/test_dixon_coles.py
git commit -m "feat: expected goals, scoreline grid, and result probabilities"
```

---

## Task 7: Scoreline sampling (for simulation)

**Files:**
- Modify: `src/dixon_coles.py` (append method)
- Test: `tests/test_dixon_coles.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_sample_scoreline_reproducible_with_seed(fitted_model):
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    s1 = [fitted_model.sample_scoreline("Strong", "Weak", rng1) for _ in range(20)]
    s2 = [fitted_model.sample_scoreline("Strong", "Weak", rng2) for _ in range(20)]
    assert s1 == s2


def test_sample_scoreline_returns_goal_pair(fitted_model):
    rng = np.random.default_rng(0)
    hg, ag = fitted_model.sample_scoreline("Strong", "Weak", rng)
    assert 0 <= hg <= fitted_model.max_goals
    assert 0 <= ag <= fitted_model.max_goals


def test_sampled_mean_favors_strong(fitted_model):
    rng = np.random.default_rng(1)
    diffs = [hg - ag for hg, ag in
             (fitted_model.sample_scoreline("Strong", "Weak", rng) for _ in range(2000))]
    assert np.mean(diffs) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_dixon_coles.py::test_sample_scoreline_returns_goal_pair -v`
Expected: FAIL — no attribute `sample_scoreline`.

- [ ] **Step 3: Append the sampling method**

```python
    def sample_scoreline(self, home, away, rng, neutral=True):
        mat = self.score_matrix(home, away, neutral)
        flat = mat.ravel()
        k = rng.choice(flat.size, p=flat)
        hg, ag = divmod(int(k), mat.shape[1])
        return hg, ag
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_dixon_coles.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dixon_coles.py tests/test_dixon_coles.py
git commit -m "feat: draw random scorelines from the model distribution"
```

---

## Task 8: Scoring metrics (`result_outcome`, `rps`, `log_loss`, `evaluate`)

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.evaluate import result_outcome, rps, log_loss, evaluate


def test_result_outcome():
    assert result_outcome(2, 1) == 0
    assert result_outcome(1, 1) == 1
    assert result_outcome(0, 2) == 2


def test_rps_perfect_is_zero():
    assert rps([1.0, 0.0, 0.0], 0) == 0.0


def test_rps_known_value():
    # forecast [0.5,0.3,0.2], outcome=home(0): cum=[.5,.8], obs cum=[1,1]
    # ((.5-1)^2 + (.8-1)^2)/2 = (.25 + .04)/2 = 0.145
    np.testing.assert_allclose(rps([0.5, 0.3, 0.2], 0), 0.145)


def test_log_loss_decreases_with_confidence():
    assert log_loss([0.7, 0.2, 0.1], 0) < log_loss([0.4, 0.3, 0.3], 0)


def test_evaluate_aggregates():
    preds = [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]]
    outs = [0, 2]
    out = evaluate(preds, outs)
    assert set(out) == {"rps", "log_loss", "accuracy"}
    assert out["accuracy"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluate'`.

- [ ] **Step 3: Implement metrics in `src/evaluate.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: RPS, log-loss, and aggregate evaluation metrics"
```

---

## Task 9: Baselines, calibration, walk-forward backtest

**Files:**
- Modify: `src/evaluate.py` (append)
- Test: `tests/test_evaluate.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from src.evaluate import (base_rate_probs, rank_baseline_probs,
                          calibration_curve, walk_forward_worldcups)
from src.dixon_coles import DixonColesModel


def test_base_rate_probs_sums_to_one():
    p = base_rate_probs([0, 0, 1, 2, 0])
    np.testing.assert_allclose(sum(p), 1.0)
    assert p[0] > p[2]  # home most common in this sample


def test_rank_baseline_favors_better_rank():
    p = rank_baseline_probs(home_rank=3, away_rank=40)
    assert p[0] > p[2]


def test_calibration_curve_perfect_model():
    home_probs = [0.1, 0.1, 0.9, 0.9]
    home_won = [0, 0, 1, 1]
    centers, freqs = calibration_curve(home_probs, home_won, n_bins=10)
    assert np.all((freqs >= 0) & (freqs <= 1))
    assert len(centers) == len(freqs)


def test_walk_forward_runs(synthetic_matches):
    # relabel a slice as a "World Cup" so the loop has a test set
    m = synthetic_matches.copy()
    m.loc[m["date"].dt.year == 2015, "tournament"] = "FIFA World Cup"
    preds, outs = walk_forward_worldcups(
        m, wc_years=[2015], model_factory=DixonColesModel, xi=0.0)
    assert len(preds) == len(outs) > 0
    assert all(abs(sum(p) - 1.0) < 1e-6 for p in preds)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py::test_walk_forward_runs -v`
Expected: FAIL — `ImportError` for the new names.

- [ ] **Step 3: Append baselines, calibration, and walk-forward**

```python
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


def walk_forward_worldcups(matches, wc_years, model_factory, xi):
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
        for _, row in test.iterrows():
            d = model.predict_result(row["home_team"], row["away_team"],
                                     neutral=bool(row["neutral"]))
            preds.append([d["home_win"], d["draw"], d["away_win"]])
            outs.append(result_outcome(row["home_score"], row["away_score"]))
    return preds, outs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: baselines, calibration curve, and walk-forward backtest"
```

---

## Task 10: Group-stage simulation (`simulate_group`, `rank_group`)

**Files:**
- Create: `src/simulate.py`
- Test: `tests/test_simulate.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.simulate import simulate_group, rank_group


def test_simulate_group_structure(stub_model_factory):
    model = stub_model_factory({"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5})
    rng = np.random.default_rng(0)
    standings = simulate_group(["A", "B", "C", "D"], model, rng)
    assert set(standings["team"]) == {"A", "B", "C", "D"}
    assert set(["team", "pts", "gd", "gf"]).issubset(standings.columns)
    assert standings["pts"].sum() <= 6 * 3  # 6 matches, max 3 pts each


def test_rank_group_orders_by_points(stub_model_factory):
    import pandas as pd
    standings = pd.DataFrame({
        "team": ["A", "B", "C", "D"],
        "pts": [9, 6, 3, 0], "gd": [5, 1, -1, -5], "gf": [8, 4, 2, 1]})
    rng = np.random.default_rng(0)
    assert rank_group(standings, rng) == ["A", "B", "C", "D"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_simulate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.simulate'`.

- [ ] **Step 3: Implement group simulation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_simulate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simulate.py tests/test_simulate.py
git commit -m "feat: group-stage round-robin simulation and ranking"
```

---

## Task 11: Best-third selection + knockout match

**Files:**
- Modify: `src/simulate.py` (append)
- Test: `tests/test_simulate.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from src.simulate import select_best_thirds, simulate_knockout_match


def test_select_best_thirds_returns_eight():
    thirds = [{"team": f"T{i}", "pts": i, "gd": i, "gf": i} for i in range(12)]
    rng = np.random.default_rng(0)
    chosen = select_best_thirds(thirds, rng)
    assert len(chosen) == 8
    assert "T11" in chosen and "T0" not in chosen  # best kept, worst dropped


def test_knockout_returns_one_of_the_two(stub_model_factory):
    model = stub_model_factory({"A": 2.0, "B": -1.0})
    rng = np.random.default_rng(0)
    winner = simulate_knockout_match("A", "B", model, rng)
    assert winner in ("A", "B")


def test_knockout_favors_stronger_over_many_runs(stub_model_factory):
    model = stub_model_factory({"A": 2.0, "B": -1.0})
    rng = np.random.default_rng(0)
    wins = sum(simulate_knockout_match("A", "B", model, rng) == "A"
               for _ in range(500))
    assert wins > 250
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_simulate.py::test_select_best_thirds_returns_eight -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Append best-third selection and knockout logic**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_simulate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/simulate.py tests/test_simulate.py
git commit -m "feat: best-third selection and knockout match resolution"
```

---

## Task 12: Full tournament + Monte Carlo

**Files:**
- Modify: `src/simulate.py` (append)
- Test: `tests/test_simulate.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from src.simulate import simulate_tournament, monte_carlo


def _toy_groups():
    # 12 groups of 4 = 48 teams named G{group}{slot}
    return {chr(65 + g): [f"G{chr(65 + g)}{s}" for s in range(4)]
            for g in range(12)}


def _toy_model(stub_model_factory):
    teams = [t for g in _toy_groups().values() for t in g]
    strengths = {t: i / len(teams) for i, t in enumerate(teams)}
    return stub_model_factory(strengths)


def test_simulate_tournament_produces_one_champion(stub_model_factory):
    model = _toy_model(stub_model_factory)
    rng = np.random.default_rng(0)
    reached = simulate_tournament(_toy_groups(), model, rng)
    assert len(reached) == 48
    assert sum(1 for s in reached.values() if s == "champion") == 1


def test_monte_carlo_probabilities_valid(stub_model_factory):
    model = _toy_model(stub_model_factory)
    df = monte_carlo(_toy_groups(), model, n_sims=40, seed=0)
    assert len(df) == 48
    for col in ["p_r32", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"]:
        assert ((df[col] >= 0) & (df[col] <= 1)).all()
    # cumulative-reach monotonicity: reaching the final implies reaching r16
    assert (df["p_r16"] >= df["p_final"] - 1e-9).all()
    np.testing.assert_allclose(df["p_champion"].sum(), 1.0, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_simulate.py::test_simulate_tournament_produces_one_champion -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Append tournament and Monte Carlo (with simplified, documented seeding)**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_simulate.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/simulate.py tests/test_simulate.py
git commit -m "feat: full tournament simulation and Monte Carlo aggregation"
```

---

## Task 13: WC2026 group draw (`wc2026_groups.py`)

**Files:**
- Create: `wc2026_groups.py`
- Test: `tests/test_groups.py`

- [ ] **Step 1: Write the failing structural test**

```python
from wc2026_groups import GROUPS


def test_twelve_groups_of_four():
    assert len(GROUPS) == 12
    assert all(len(v) == 4 for v in GROUPS.values())


def test_forty_eight_unique_teams():
    teams = [t for g in GROUPS.values() for t in g]
    assert len(teams) == 48
    assert len(set(teams)) == 48
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_groups.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026_groups'`.

- [ ] **Step 3: Create `wc2026_groups.py` and transcribe the official draw**

Transcribe the official FIFA 2026 group draw (groups A–L). Team names MUST match the
cleaned dataset's names (apply the same aliases as `DEFAULT_ALIASES`, e.g. "United States",
"South Korea"). Use this exact schema:

```python
"""WC2026 group draw. Transcribed from the official FIFA draw.

Team names must match the cleaned results dataset (see src.data.DEFAULT_ALIASES).
"""

GROUPS = {
    "A": ["Mexico", "...", "...", "..."],
    "B": ["Canada", "...", "...", "..."],
    "C": ["United States", "...", "...", "..."],
    "D": ["...", "...", "...", "..."],
    "E": ["...", "...", "...", "..."],
    "F": ["...", "...", "...", "..."],
    "G": ["...", "...", "...", "..."],
    "H": ["...", "...", "...", "..."],
    "I": ["...", "...", "...", "..."],
    "J": ["...", "...", "...", "..."],
    "K": ["...", "...", "...", "..."],
    "L": ["...", "...", "...", "..."],
}
```

Replace every `"..."` with the real drawn team. The structural test in Step 1 fails until
all 48 slots hold 48 distinct teams.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_groups.py -v`
Expected: PASS once the draw is fully transcribed.

- [ ] **Step 5: Commit**

```bash
git add wc2026_groups.py tests/test_groups.py
git commit -m "feat: WC2026 official group draw with structural validation"
```

---

## Task 14: Data download + EDA notebook (`notebooks/01_eda.ipynb`)

**Files:**
- Create: `notebooks/01_eda.ipynb`

- [ ] **Step 1: Download the datasets into `data/raw/`**

Two Kaggle datasets:
- Results: `martj42/international-football-results-from-1872-to-2017` → save `results.csv` to `data/raw/results.csv`.
- FIFA rankings: `cashncarry/fifaworldranking` → save the ranking CSV to `data/raw/fifa_ranking.csv`.

Via Kaggle API (after `pip install kaggle` and placing `kaggle.json`):
```bash
kaggle datasets download -d martj42/international-football-results-from-1872-to-2017 -p data/raw --unzip
kaggle datasets download -d cashncarry/fifaworldranking -p data/raw --unzip
```
Or download manually from kaggle.com and place the CSVs in `data/raw/`.

- [ ] **Step 2: Create the EDA notebook with these cells**

Cell 1 (load & clean):
```python
import sys; sys.path.append("..")
import pandas as pd
import matplotlib.pyplot as plt
from src.data import load_results, clean_results

raw = load_results("../data/raw/results.csv")
df = clean_results(raw)
print(df.shape)
df.head()
```

Cell 2 (sanity checks):
```python
assert df["home_score"].min() >= 0 and df["away_score"].min() >= 0
print("date range:", df["date"].min(), "→", df["date"].max())
print("matches:", len(df), "| unique teams:", len(set(df.home_team) | set(df.away_team)))
df["tournament"].value_counts().head(10)
```

Cell 3 (goal distribution — the key modeling assumption):
```python
ax = df["home_score"].value_counts().sort_index().plot(kind="bar")
ax.set_title("Distribution of home goals"); ax.set_xlabel("goals"); plt.show()
print("mean home goals:", round(df["home_score"].mean(), 3))
print("mean away goals:", round(df["away_score"].mean(), 3))
```

Cell 4 (persist cleaned data):
```python
import os
os.makedirs("../data/processed", exist_ok=True)
df.to_parquet("../data/processed/matches.parquet")
print("saved", len(df), "rows")
```

- [ ] **Step 3: Run the notebook top to bottom**

Run: `.venv/Scripts/jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb`
Expected: executes without error; `data/processed/matches.parquet` exists; home goals mean > away goals mean (home advantage visible).

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_eda.ipynb
git commit -m "feat: EDA notebook — load, clean, sanity-check, persist"
```

---

## Task 15: Model notebook (`notebooks/02_model.ipynb`)

**Files:**
- Create: `notebooks/02_model.ipynb`

- [ ] **Step 1: Create the notebook with these cells**

Cell 1 (fit — note the runtime caveat):
```python
import sys; sys.path.append("..")
import pandas as pd
from src.dixon_coles import DixonColesModel

df = pd.read_parquet("../data/processed/matches.parquet")
# Fitting ~400 params via numerical-gradient L-BFGS over all matches can take a few
# minutes. Time decay already down-weights old games, so restricting to recent years
# costs little accuracy and speeds iteration:
df_fit = df[df["date"] >= "1995-01-01"]
model = DixonColesModel().fit(df_fit, xi=0.0019)  # xi ~ tuned in notebook 03
print("home advantage (gamma):", round(model.home_adv, 3), "| rho:", round(model.rho, 4))
```

Cell 2 (read off ratings — the interpretability payoff):
```python
ratings = pd.DataFrame({
    "team": model.teams, "attack": model.attack, "defense": model.defense})
ratings["net"] = ratings["attack"] - ratings["defense"]
ratings.sort_values("net", ascending=False).head(15)
```
Sanity check: traditional powerhouses (Brazil, France, Argentina, Spain, Germany) should rank near the top.

Cell 3 (FIFA-rank sanity check):
```python
from src.data import load_rankings, latest_rankings
ranks = latest_rankings(load_rankings("../data/raw/fifa_ranking.csv"))
ratings["fifa_rank"] = ratings["team"].map(ranks)
corr = ratings.dropna(subset=["fifa_rank"])[["net", "fifa_rank"]].corr().iloc[0, 1]
print("corr(net strength, FIFA rank):", round(corr, 3), "(expect strongly negative)")
```

Cell 4 (example prediction + scoreline grid):
```python
print(model.predict_result("Brazil", "Croatia", neutral=True))
print("most likely score:",
      divmod(int(model.score_matrix("Brazil", "Croatia").argmax()), 11))
```

- [ ] **Step 2: Run the notebook top to bottom**

Run: `.venv/Scripts/jupyter nbconvert --to notebook --execute notebooks/02_model.ipynb --output 02_model.ipynb`
Expected: executes without error; top-rated teams are sane; correlation with FIFA rank is strongly negative (better rank number ⇒ higher strength).

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_model.ipynb
git commit -m "feat: model notebook — fit, interpret ratings, FIFA-rank sanity check"
```

---

## Task 16: Backtest notebook (`notebooks/03_backtest.ipynb`)

**Files:**
- Create: `notebooks/03_backtest.ipynb`

- [ ] **Step 1: Create the notebook with these cells**

Cell 1 (walk-forward over recent World Cups):
```python
import sys; sys.path.append("..")
import numpy as np, pandas as pd
from src.dixon_coles import DixonColesModel
from src.evaluate import (walk_forward_worldcups, evaluate, base_rate_probs,
                          result_outcome, calibration_curve)

df = pd.read_parquet("../data/processed/matches.parquet")
wc_years = [2010, 2014, 2018, 2022]
preds, outs = walk_forward_worldcups(df, wc_years, DixonColesModel, xi=0.0019)
print("model:", evaluate(preds, outs))
```

Cell 2 (compare against baselines):
```python
train_outcomes = [result_outcome(r.home_score, r.away_score)
                  for r in df[df.date < "2010-01-01"].itertuples()]
base = base_rate_probs(train_outcomes)
base_preds = [base] * len(outs)
print("base-rate baseline:", evaluate(base_preds, outs))
```
The model's RPS and log-loss should be lower (better) than the baseline's.

Cell 3 (tune the time-decay rate xi):
```python
results = {}
for xi in [0.0, 0.001, 0.0019, 0.003, 0.005]:
    p, o = walk_forward_worldcups(df, wc_years, DixonColesModel, xi=xi)
    results[xi] = evaluate(p, o)["rps"]
best_xi = min(results, key=results.get)
print("RPS by xi:", {k: round(v, 4) for k, v in results.items()})
print("best xi:", best_xi)
```

Cell 4 (calibration plot):
```python
import matplotlib.pyplot as plt
home_probs = [p[0] for p in preds]
home_won = [int(o == 0) for o in outs]
centers, freqs = calibration_curve(home_probs, home_won, n_bins=8)
plt.plot([0, 1], [0, 1], "--", label="perfect")
plt.plot(centers, freqs, "o-", label="model")
plt.xlabel("predicted P(home win)"); plt.ylabel("observed frequency")
plt.legend(); plt.title("Calibration"); plt.show()
```

- [ ] **Step 2: Run the notebook top to bottom**

Run: `.venv/Scripts/jupyter nbconvert --to notebook --execute notebooks/03_backtest.ipynb --output 03_backtest.ipynb`
Expected: executes without error; model RPS < base-rate RPS; calibration points lie near the diagonal. Note the `best_xi` value and use it in notebooks 02 and 04.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_backtest.ipynb
git commit -m "feat: backtest notebook — walk-forward, baselines, xi tuning, calibration"
```

---

## Task 17: Simulation notebook (`notebooks/04_simulation.ipynb`) + README

**Files:**
- Create: `notebooks/04_simulation.ipynb`
- Create: `README.md`

- [ ] **Step 1: Create the simulation notebook with these cells**

Cell 1 (fit on full recent data + run Monte Carlo):
```python
import sys; sys.path.append("..")
import pandas as pd
from src.dixon_coles import DixonColesModel
from src.simulate import monte_carlo
from wc2026_groups import GROUPS

df = pd.read_parquet("../data/processed/matches.parquet")
model = DixonColesModel().fit(df[df.date >= "1995-01-01"], xi=0.0019)
probs = monte_carlo(GROUPS, model, n_sims=20000, seed=2026)
probs.head(12)
```

Cell 2 (title-odds bar chart):
```python
import matplotlib.pyplot as plt
top = probs.head(12).iloc[::-1]
plt.barh(top["team"], top["p_champion"] * 100)
plt.xlabel("P(win World Cup) %"); plt.title("WC2026 title odds (top 12)")
plt.tight_layout(); plt.show()
```

Cell 3 (per-stage table for a chosen team):
```python
row = probs[probs["team"] == "Brazil"].iloc[0]
print({s: f"{row[f'p_{s}'] * 100:.1f}%"
       for s in ["r32", "r16", "qf", "sf", "final", "champion"]})
```

- [ ] **Step 2: Run the notebook top to bottom**

Run: `.venv/Scripts/jupyter nbconvert --to notebook --execute notebooks/04_simulation.ipynb --output 04_simulation.ipynb`
Expected: executes without error; `p_champion` sums to ~1.0 across all 48 teams; favorites top the chart. (20k sims may take a few minutes.)

- [ ] **Step 3: Write `README.md`**

```markdown
# WC2026 — World Cup 2026 Scoreline & Result Prediction

A Dixon-Coles goals model that predicts football scoreline probabilities, backtested
against past World Cups and used to Monte-Carlo-simulate the 2026 tournament.

## Setup
1. `python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt`
2. Download datasets into `data/raw/` (see `notebooks/01_eda.ipynb`, Step 1).
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

## Known simplifications
- Cold-start for unseen teams uses the global-average rating (FIFA rank used only for the
  model sanity-check).
- Knockout seeding is a simplified high-vs-low bracket, not FIFA's exact third-place table.

## Future enhancements
- FIFA-rank-based cold-start prior for sparse teams.
- Streamlit app for interactive matchup predictions.
- Bayesian hierarchical (bivariate Poisson via PyMC) variant.
```

- [ ] **Step 4: Run the full test suite one last time**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/04_simulation.ipynb README.md
git commit -m "feat: simulation notebook and project README"
```

---

## Self-review (completed by plan author)

**Spec coverage:**
- §3 model (attack/defense, λ, scoreline grid, τ, MLE fit, time decay) → Tasks 4–7. ✓
- §4 data pipeline (load, clean, alias normalization, rankings, parquet) → Tasks 3, 14. ✓
- §5 backtest (walk-forward, RPS, log-loss, calibration, baselines, xi tuning) → Tasks 8, 9, 16. ✓
- §6 simulation (2026 format, group RR, best thirds, knockout, Monte Carlo) → Tasks 10–12, 17. ✓
- §7 structure, §8 stack → Task 1 + layout. ✓
- §9 testing (grid sums to 1, stronger team favored, sim counts, tiebreakers, seeded reproducibility) → Tasks 6, 7, 10–12. ✓
- §10 build order → task ordering matches milestones. ✓
- §11 risks (name normalization, sparse teams, format novelty, distribution reasoning) → Tasks 3, 5/6 fallback, 10–13 tests, README. ✓

**Placeholder scan:** Only intentional data-entry blanks in Task 13 (the official draw), guarded by a structural test. No logic placeholders.

**Type/name consistency:** Checked — `clean_results`, `DixonColesModel.{fit,_params,expected_goals,score_matrix,predict_result,sample_scoreline}`, `result_outcome`, `rps`, `log_loss`, `evaluate`, `base_rate_probs`, `rank_baseline_probs`, `calibration_curve`, `walk_forward_worldcups`, `simulate_group`, `rank_group`, `select_best_thirds`, `simulate_knockout_match`, `simulate_tournament`, `monte_carlo`, `GROUPS` are used consistently across tasks and tests.
