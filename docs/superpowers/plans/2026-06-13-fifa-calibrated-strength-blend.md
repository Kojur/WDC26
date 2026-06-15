# FIFA-Calibrated Strength Blend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the Dixon-Coles model's over-rating of the South American pool by blending each team's learned net strength toward a FIFA-points prior, with the blend weight α chosen by the existing walk-forward backtest so accuracy is preserved.

**Architecture:** A pure, post-fit transform. The MLE fit is unchanged; a new `calibrate()` function takes a fitted model + a `{team: fifa_points}` dict + α and returns a calibrated copy (net strength shrunk toward the standardized FIFA prior, the change split evenly between attack and defense). A backtest sweep picks α. The export script fits → calibrates → simulates, and writes calibrated ratings; the browser predictor and Python↔TS parity fixtures stay in sync automatically because both read the exported `attack`/`defense`.

**Tech Stack:** Python (numpy, scipy, pandas, pytest), TypeScript/React (Vite, Vitest) for the showcase site.

**Reference spec:** `docs/superpowers/specs/2026-06-13-fifa-calibrated-strength-blend-design.md`

**Branch:** `feat/fifa-calibration` (already checked out).

**Environment note:** The Python interpreter is `.venv/Scripts/python`. Run tests with
`.venv/Scripts/python -m pytest`. Site commands run from `site/` with `npm`.

---

## File Structure

**Python core (`src/`)**
- `src/data.py` — *modify*: add `latest_points(rankings, as_of=None)` (FIFA `total_points` lookup, mirrors `latest_rankings`).
- `src/calibrate.py` — *create*: the blend transform `calibrate(model, fifa_points, alpha)`.
- `src/evaluate.py` — *modify*: `walk_forward_worldcups` gains optional `(fifa_rankings, alpha)`; add `alpha_backtest_curve(...)` and `choose_alpha(curve, tol)`.

**Export (`scripts/`)**
- `scripts/export_site_data.py` — *modify*: add pure `calibration_payload(...)`; wire fit → α-sweep → calibrate → simulate into `main()` and `meta_payload(...)`.

**Tests (`tests/`)**
- `tests/test_data.py` — *modify*: test `latest_points`.
- `tests/test_calibrate.py` — *create*: unit tests for the blend.
- `tests/test_evaluate.py` — *modify*: tests for calibrated walk-forward, curve, and `choose_alpha`.
- `tests/test_export_site_data.py` — *modify*: test `calibration_payload`.

**Showcase (`site/src/`)**
- `site/src/data/types.ts` — *modify*: extend `Meta` (`backtest.alpha`, optional `calibration`).
- `site/src/components/HowItWorks.tsx`, `Validation.tsx`, `Limitations.tsx` — *modify*: reframe copy.
- `site/src/components/sections.test.tsx` — *verify still passing* (keep "South American" in Limitations copy).
- `README.md` — *modify*: Method / Known simplifications / sanity-check wording.

**Regenerated data (committed):** `site/src/data/{ratings,simulation,meta,parity_fixtures}.json`.

---

## Task 1: FIFA points lookup (`latest_points`)

**Files:**
- Modify: `src/data.py` (add function after `latest_rankings`, ~line 50)
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_data.py`:

```python
def test_latest_points_picks_most_recent_on_or_before():
    import pandas as pd
    from src.data import latest_points
    rankings = pd.DataFrame({
        "rank_date": pd.to_datetime(["2014-01-01", "2014-01-01", "2018-06-01"]),
        "country_full": ["France", "Brazil", "France"],
        "total_points": [1500.0, 1600.0, 1837.0],
    })
    # as_of before the 2018 snapshot -> uses the 2014 snapshot
    early = latest_points(rankings, as_of=pd.Timestamp("2015-01-01"))
    assert early == {"France": 1500.0, "Brazil": 1600.0}
    # no as_of -> latest snapshot only (2018, France only)
    assert latest_points(rankings) == {"France": 1837.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_data.py::test_latest_points_picks_most_recent_on_or_before -v`
Expected: FAIL with `ImportError: cannot import name 'latest_points'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/data.py` (after `latest_rankings`):

```python
def latest_points(rankings, as_of=None):
    """Return {country: total_points} from the most recent snapshot on/before as_of."""
    df = rankings if as_of is None else rankings[rankings["rank_date"] <= as_of]
    snap = df[df["rank_date"] == df["rank_date"].max()]
    return dict(zip(snap["country_full"], snap["total_points"]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_data.py -v`
Expected: PASS (all data tests green).

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: latest_points FIFA total_points lookup (as-of aware)"
```

---

## Task 2: The blend transform (`src/calibrate.py`)

**Files:**
- Create: `src/calibrate.py`
- Test: `tests/test_calibrate.py`

Uses the `fitted_model` fixture from `tests/conftest.py` (teams `Strong`/`Medium`/`Weak`,
net strength `Strong > Medium > Weak`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibrate.py`:

```python
import numpy as np
from src.calibrate import calibrate


def _net(model, team):
    i = model.team_index[team]
    return model.attack[i] - model.defense[i]


def test_alpha_one_is_identity(fitted_model):
    pts = {"Strong": 100.0, "Medium": 200.0, "Weak": 300.0}  # inverts model order
    cal = calibrate(fitted_model, pts, alpha=1.0)
    np.testing.assert_allclose(cal.attack, fitted_model.attack, atol=1e-9)
    np.testing.assert_allclose(cal.defense, fitted_model.defense, atol=1e-9)


def test_alpha_zero_follows_fifa_ordering(fitted_model):
    # FIFA says Weak is strongest -> after full pull, Weak has highest net
    pts = {"Strong": 100.0, "Medium": 200.0, "Weak": 300.0}
    cal = calibrate(fitted_model, pts, alpha=0.0)
    assert _net(cal, "Weak") > _net(cal, "Medium") > _net(cal, "Strong")


def test_more_fifa_pull_lifts_underrated_team(fitted_model):
    # Weak is FIFA-strong but model-weak: lowering alpha (more FIFA) raises its net
    pts = {"Strong": 100.0, "Medium": 200.0, "Weak": 300.0}
    high_alpha = _net(calibrate(fitted_model, pts, alpha=0.8), "Weak")
    low_alpha = _net(calibrate(fitted_model, pts, alpha=0.3), "Weak")
    assert low_alpha > high_alpha


def test_team_without_fifa_entry_is_untouched(fitted_model):
    pts = {"Strong": 100.0, "Medium": 200.0}  # Weak omitted
    cal = calibrate(fitted_model, pts, alpha=0.3)
    i = fitted_model.team_index["Weak"]
    assert cal.attack[i] == fitted_model.attack[i]
    assert cal.defense[i] == fitted_model.defense[i]


def test_does_not_mutate_original(fitted_model):
    before = fitted_model.attack.copy()
    pts = {"Strong": 100.0, "Medium": 200.0, "Weak": 300.0}
    calibrate(fitted_model, pts, alpha=0.5)
    np.testing.assert_array_equal(fitted_model.attack, before)


def test_rejects_alpha_out_of_range(fitted_model):
    import pytest
    with pytest.raises(ValueError):
        calibrate(fitted_model, {"Strong": 1.0}, alpha=1.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.calibrate'`.

- [ ] **Step 3: Write the implementation**

Create `src/calibrate.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_calibrate.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/calibrate.py tests/test_calibrate.py
git commit -m "feat: calibrate() blends net strength toward a FIFA prior"
```

---

## Task 3: Calibrated walk-forward backtest

**Files:**
- Modify: `src/evaluate.py:67-86` (`walk_forward_worldcups`)
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluate.py`:

```python
import pandas as pd


def _fifa_fixture():
    return pd.DataFrame({
        "rank_date": pd.to_datetime(["2014-01-01"] * 3),
        "country_full": ["Strong", "Medium", "Weak"],
        "total_points": [1500.0, 1300.0, 1100.0],
    })


def test_walk_forward_calibrated_runs(synthetic_matches):
    m = synthetic_matches.copy()
    m.loc[m["date"].dt.year == 2015, "tournament"] = "FIFA World Cup"
    preds, outs = walk_forward_worldcups(
        m, wc_years=[2015], model_factory=DixonColesModel, xi=0.0,
        fifa_rankings=_fifa_fixture(), alpha=0.5)
    assert len(preds) == len(outs) > 0
    assert all(abs(sum(p) - 1.0) < 1e-6 for p in preds)


def test_walk_forward_alpha_one_matches_uncalibrated(synthetic_matches):
    m = synthetic_matches.copy()
    m.loc[m["date"].dt.year == 2015, "tournament"] = "FIFA World Cup"
    base, _ = walk_forward_worldcups(
        m, wc_years=[2015], model_factory=DixonColesModel, xi=0.0)
    same, _ = walk_forward_worldcups(
        m, wc_years=[2015], model_factory=DixonColesModel, xi=0.0,
        fifa_rankings=_fifa_fixture(), alpha=1.0)
    np.testing.assert_allclose(base, same, atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py::test_walk_forward_calibrated_runs -v`
Expected: FAIL with `TypeError: walk_forward_worldcups() got an unexpected keyword argument 'fifa_rankings'`.

- [ ] **Step 3: Write the implementation**

In `src/evaluate.py`, add imports at the top (after the existing `import pandas as pd`):

```python
from src.calibrate import calibrate
from src.data import latest_points
```

Replace the signature and fit block of `walk_forward_worldcups`. Change the
`def` line to:

```python
def walk_forward_worldcups(matches, wc_years, model_factory, xi,
                           fifa_rankings=None, alpha=1.0):
```

and replace the single line `model = model_factory().fit(train, xi=xi, ref_date=cutoff)`
with:

```python
        model = model_factory().fit(train, xi=xi, ref_date=cutoff)
        if fifa_rankings is not None and alpha < 1.0:
            points = latest_points(fifa_rankings, as_of=cutoff)
            model = calibrate(model, points, alpha)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py -v`
Expected: PASS (existing evaluate tests + the two new ones).

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: optional FIFA calibration in walk_forward_worldcups"
```

---

## Task 4: α sweep and selection (`alpha_backtest_curve`, `choose_alpha`)

**Files:**
- Modify: `src/evaluate.py` (add two functions at end of file)
- Test: `tests/test_evaluate.py`

`choose_alpha` rule: minimize log-loss; among α within `tol` of the best (a
statistically indifferent band), prefer the **smaller** α — i.e. more FIFA
correction, which fixes the top-end bias that motivated the change.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_evaluate.py`:

```python
def test_alpha_backtest_curve_shape(synthetic_matches):
    from src.evaluate import alpha_backtest_curve
    m = synthetic_matches.copy()
    m.loc[m["date"].dt.year == 2015, "tournament"] = "FIFA World Cup"
    curve = alpha_backtest_curve(
        m, wc_years=[2015], model_factory=DixonColesModel, xi=0.0,
        fifa_rankings=_fifa_fixture(), alphas=[0.5, 1.0])
    assert [c["alpha"] for c in curve] == [0.5, 1.0]
    assert all({"alpha", "log_loss", "rps"} <= set(c) for c in curve)


def test_choose_alpha_picks_min_log_loss():
    from src.evaluate import choose_alpha
    curve = [{"alpha": 0.0, "log_loss": 1.10, "rps": 0.25},
             {"alpha": 0.5, "log_loss": 1.00, "rps": 0.21},
             {"alpha": 1.0, "log_loss": 1.05, "rps": 0.23}]
    assert choose_alpha(curve, tol=0.0) == 0.5


def test_choose_alpha_tie_prefers_more_fifa():
    from src.evaluate import choose_alpha
    # 0.3 and 0.6 are within tol of the best (1.00); prefer the smaller alpha
    curve = [{"alpha": 0.3, "log_loss": 1.004, "rps": 0.210},
             {"alpha": 0.6, "log_loss": 1.000, "rps": 0.210},
             {"alpha": 1.0, "log_loss": 1.090, "rps": 0.230}]
    assert choose_alpha(curve, tol=0.01) == 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py::test_choose_alpha_picks_min_log_loss -v`
Expected: FAIL with `ImportError: cannot import name 'choose_alpha'`.

- [ ] **Step 3: Write the implementation**

Append to `src/evaluate.py`:

```python
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
    """Pick alpha minimizing log-loss; within `tol` of the best, prefer smaller
    alpha (more FIFA correction). RPS breaks remaining ties."""
    best = min(c["log_loss"] for c in curve)
    band = [c for c in curve if c["log_loss"] <= best + tol]
    return min(band, key=lambda c: (c["alpha"], c["rps"]))["alpha"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py -v`
Expected: PASS (all evaluate tests).

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: alpha_backtest_curve and choose_alpha for blend tuning"
```

---

## Task 5: Wire calibration into the export script

**Files:**
- Modify: `scripts/export_site_data.py`
- Test: `tests/test_export_site_data.py`

### 5a — pure `calibration_payload` helper

- [ ] **Step 1: Write the failing test**

Add to `tests/test_export_site_data.py`:

```python
def test_calibration_payload_shape():
    import pandas as pd
    from scripts.export_site_data import calibration_payload
    before = pd.DataFrame([{"team": "Brazil", "p_champion": 0.28},
                           {"team": "Colombia", "p_champion": 0.11}])
    after = pd.DataFrame([{"team": "Argentina", "p_champion": 0.21},
                          {"team": "France", "p_champion": 0.09}])
    curve = [{"alpha": 0.0, "log_loss": 1.10, "rps": 0.25},
             {"alpha": 0.5, "log_loss": 1.00, "rps": 0.21}]
    p = calibration_payload(0.5, curve, "2024-06-20", before, after, top_n=2)
    assert p["prior"] == "fifa_total_points"
    assert p["alpha"] == 0.5 and p["fifa_as_of"] == "2024-06-20"
    assert p["alpha_curve"][0] == {"alpha": 0.0, "log_loss": 1.1, "rps": 0.25}
    assert p["before_top12"][0] == {"team": "Brazil", "p_champion": 0.28}
    assert p["after_top12"][0] == {"team": "Argentina", "p_champion": 0.21}
    assert len(p["before_top12"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_export_site_data.py::test_calibration_payload_shape -v`
Expected: FAIL with `ImportError: cannot import name 'calibration_payload'`.

- [ ] **Step 3: Write the implementation**

In `scripts/export_site_data.py`, add the helper (after `parity_payload`):

```python
def calibration_payload(alpha, curve, fifa_as_of, before_probs, after_probs, top_n=12):
    def top(df):
        return [{"team": r["team"], "p_champion": round(float(r["p_champion"]), 4)}
                for _, r in df.head(top_n).iterrows()]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_export_site_data.py -v`
Expected: PASS (existing payload tests + the new one).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_site_data.py tests/test_export_site_data.py
git commit -m "feat: calibration_payload for meta.json"
```

### 5b — update `meta_payload` and `main()`

- [ ] **Step 6: Update imports**

In `scripts/export_site_data.py`, extend the `src` imports to include the new
functions:

```python
from src.data import load_results, clean_results, load_rankings, latest_rankings, latest_points
from src.calibrate import calibrate
from src.evaluate import (walk_forward_worldcups, evaluate, base_rate_probs,
                          result_outcome, alpha_backtest_curve, choose_alpha)
```

- [ ] **Step 7: Rewrite `meta_payload` to take rankings + alpha**

Replace the entire `meta_payload` function with:

```python
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
```

Note: `ratings_sanity` is intentionally computed on the **base** (uncalibrated)
model's net strength — it stays an honest "the raw data already aligns with FIFA"
sanity check, separate from the calibration we then apply.

- [ ] **Step 8: Rewrite `main()`**

Replace the entire `main()` function with:

```python
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
```

Key points: `ratings.json` and `parity_fixtures.json` are built from the
**calibrated** model (`cal`) so the browser predictor and parity fixtures match;
`before` (uncalibrated) vs `after` (calibrated) simulations feed the before/after
table; `simulation.json` is the calibrated result.

- [ ] **Step 9: Run the export and inspect the chosen α + curve**

Run: `.venv/Scripts/python scripts/export_site_data.py`
Expected: prints `chosen alpha=<value>; curve=[...]`, then `wrote ratings.json` … `done`.

**DECISION CHECKPOINT (per spec §6 contingency):** read the printed curve and the
new `site/src/data/meta.json` `calibration.after_top12`:
- If the chosen α is interior (e.g. 0.3–0.7) and the top of `after_top12` looks
  sane (Brazil/Argentina/France/Spain/England/Portugal-ish), proceed.
- If `choose_alpha` returned **1.0** (the backtest strictly prefers the pure model),
  **0.0** (the model was discarded entirely for pure FIFA), or the top still looks
  wrong, **stop and surface the curve + trade-off to the user** before continuing —
  do not silently ship.

- [ ] **Step 10: Run the full Python suite**

Run: `.venv/Scripts/python -m pytest`
Expected: all tests PASS.

- [ ] **Step 11: Commit**

```bash
git add scripts/export_site_data.py site/src/data/ratings.json site/src/data/simulation.json site/src/data/meta.json site/src/data/parity_fixtures.json site/src/data/groups.json
git commit -m "feat: export calibrated ratings, simulation, and before/after meta"
```

---

## Task 6: Showcase types, copy, and parity check

**Files:**
- Modify: `site/src/data/types.ts`
- Modify: `site/src/components/HowItWorks.tsx`, `Validation.tsx`, `Limitations.tsx`
- Modify: `README.md`
- Verify: `site/src/components/sections.test.tsx`, `site/src/lib/dixonColes.test.ts`

- [ ] **Step 1: Extend the `Meta` type**

In `site/src/data/types.ts`, replace the `Meta` interface with:

```typescript
export interface Meta {
  dataset: { n_matches: number; n_teams: number; date_min: string; date_max: string };
  ratings_sanity: { fifa_rank_corr: number; n_matched: number };
  backtest: {
    wc_years: number[]; n_matches: number;
    model: { rps: number; log_loss: number; accuracy: number };
    baseline: { rps: number; log_loss: number; accuracy: number };
    best_xi: number; alpha: number;
  };
  model: { home_adv: number; rho: number };
  calibration?: {
    prior: string; fifa_as_of: string; alpha: number;
    alpha_curve: { alpha: number; log_loss: number; rps: number }[];
    before_top12: { team: string; p_champion: number }[];
    after_top12: { team: string; p_champion: number }[];
  };
}
```

- [ ] **Step 2: Update `HowItWorks.tsx` (add the calibration step)**

In `site/src/components/HowItWorks.tsx`, replace the final `<p>` (the "ten thousand
times" paragraph) with these two paragraphs:

```tsx
      <p>
        One adjustment first: a model trained only on results over-rates regions that
        mostly play among themselves — South America especially. So we gently calibrate
        each team's strength toward the FIFA ranking, by a blend weight chosen on the
        backtest, before simulating. The data still does the heavy lifting; FIFA just
        anchors the regions to each other.
      </p>
      <p>
        To predict the whole tournament, we simulate all 104 matches — group stage through
        final — <strong>ten thousand times</strong>, and count how often each team lifts
        the trophy.
      </p>
```

- [ ] **Step 3: Update `Validation.tsx` (reframe sanity check + show backtest with α)**

In `site/src/components/Validation.tsx`, replace the final `<p>` (the FIFA-correlation
paragraph) with:

```tsx
      <p>
        The raw, results-only ratings already line up strongly with the official FIFA
        ranking — an inverse correlation of {M.ratings_sanity.fifa_rank_corr} (higher-rated
        teams hold lower, i.e. better, rank numbers). We then apply a light FIFA calibration
        (blend weight {M.backtest.alpha}) to fix a known top-end bias: a results-only model
        over-rates South American teams, who play a dense schedule among themselves. The
        backtest numbers above are measured <em>with</em> that calibration applied.
      </p>
```

- [ ] **Step 4: Update `Limitations.tsx` (corrected bias + FIFA caveats; keep "South American")**

Replace the body of `site/src/components/Limitations.tsx` with:

```tsx
export function Limitations() {
  return (
    <section className="section">
      <h2>The catch</h2>
      <p>
        Left to its own devices, the model over-rates South American sides: CONMEBOL teams
        play each other constantly in a brutal round-robin, while Europe's giants rarely face
        them, so the regions never get anchored against one another. We correct this by
        blending each team's strength toward the FIFA ranking — but that borrows FIFA's own
        quirks too (a little ranking inertia for fading sides), and the FIFA snapshot we use
        is a couple of years old.
      </p>
      <p>
        It also can't see what isn't in the scoreline data: injuries, squad turnover, form, or
        tactics. Treat the numbers as an informed baseline, not a crystal ball.
      </p>
    </section>
  );
}
```

- [ ] **Step 5: Update `README.md`**

In `README.md`, replace the sanity-check paragraph (the "A quick sanity check…"
paragraph, ~lines 27-29) with:

```markdown
The model's results-only net strength already correlates about −0.92 with the official
FIFA ranking. On top of that we apply a light **FIFA calibration**: a results-only fit
over-rates regions that mostly play among themselves (South America especially), so each
team's strength is blended toward the FIFA ranking by a weight chosen on the walk-forward
backtest. This anchors the confederations to each other and gives a sensible set of title
contenders (Brazil, Argentina, France, Spain, England, …).
```

In the "## Known simplifications" section, replace the first bullet (the cold-start
bullet) with:

```markdown
- A light FIFA-points calibration is blended into the learned ratings (weight tuned on the
  backtest) to correct the closed-pool over-rating of South America. The FIFA snapshot used
  is from mid-2024, so it lags current form. Cold-start for unseen teams uses the
  global-average rating.
```

- [ ] **Step 6: Run the site test suite**

Run (from `site/`): `npm test`
Expected: PASS. `sections.test.tsx` still passes because the new `Limitations`
copy contains "South American" and `Validation` still renders "RPS". The Python↔TS
parity test (`dixonColes.test.ts`) still passes because `parity_fixtures.json` and
`ratings.json` were regenerated together from the calibrated model.

If any text assertion fails, read the failing assertion and align it with the exact
copy above (do not weaken the copy).

- [ ] **Step 7: Build the site to catch type errors**

Run (from `site/`): `npm run build`
Expected: `tsc -b` passes (the new optional `calibration` field and `backtest.alpha`
type-check) and Vite build succeeds.

- [ ] **Step 8: Commit**

```bash
git add site/src/data/types.ts site/src/components/HowItWorks.tsx site/src/components/Validation.tsx site/src/components/Limitations.tsx README.md
git commit -m "docs: reframe showcase + README around FIFA calibration"
```

---

## Task 7: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Full Python suite**

Run: `.venv/Scripts/python -m pytest`
Expected: all PASS.

- [ ] **Step 2: Full site suite + build**

Run (from `site/`): `npm test` then `npm run build`
Expected: both PASS.

- [ ] **Step 3: Confirm the rebalance and that accuracy held**

Inspect `site/src/data/meta.json`:
- `calibration.after_top12` — top contenders look sane; Colombia/Uruguay/Ecuador no
  longer sit above Spain/France.
- `backtest.model` log-loss/RPS are still **below** `backtest.baseline` (the calibrated
  model still beats the base rate).
- `calibration.before_top12` vs `after_top12` shows the intended shift.

Report these three facts (chosen α, before/after top, model-vs-baseline backtest) as the
evidence the change worked.

- [ ] **Step 4: Final commit (if any verification fixups were made)**

```bash
git add -A
git commit -m "test: verify FIFA-calibrated model rebalances top and holds backtest"
```

---

## Self-Review Notes (filled during planning)

- **Spec coverage:** §4 prior → Task 1; §5 blend → Task 2; §6 α selection → Task 4 +
  Task 5 Step 9 checkpoint; §7 code touch-points → Tasks 1-6; §8 validation/report →
  Task 7; §9 caveats (FIFA inertia, stale snapshot) → Task 6 copy. All covered.
- **No-leakage requirement** (spec §4): `latest_points(as_of=cutoff)` is used inside the
  walk-forward loop (Task 3) — the prior for each past World Cup uses only data on/before
  that cup's cutoff.
- **TS sync** (spec §7): `ratings.json` and `parity_fixtures.json` are regenerated from the
  same calibrated model in one export run (Task 5 Step 8), so parity holds with no TS logic
  change.
- **Type consistency:** `calibrate(model, fifa_points, alpha)`, `latest_points(rankings,
  as_of)`, `alpha_backtest_curve(..., fifa_rankings, alphas)`, `choose_alpha(curve, tol)`,
  and `calibration_payload(alpha, curve, fifa_as_of, before_probs, after_probs, top_n)` are
  used with identical signatures everywhere they appear.
