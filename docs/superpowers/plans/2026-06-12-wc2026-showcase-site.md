# WC2026 Showcase Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the WDC26 analysis as a polished, static, React + Vite single-page site with an in-browser Dixon-Coles matchup predictor, deployed to GitHub Pages.

**Architecture:** A Python script exports the fitted model's outputs to JSON bundled into the React app. The Dixon-Coles model is ported to TypeScript so the matchup predictor runs client-side. Pure functions are TDD'd (pytest for the exporter, Vitest for the TS model + components); the site is static-built and deployed via GitHub Actions.

**Tech Stack:** Python 3.11 (existing `src/`), React 18 + TypeScript + Vite, Recharts, Vitest + React Testing Library, GitHub Pages.

---

## Spec reference

Design spec: `docs/superpowers/specs/2026-06-12-wc2026-showcase-site-design.md`. Read it first.

## Deviation from spec (deliberate)

The spec put the JSON in `site/public/data/` fetched at runtime via `BASE_URL`. This plan instead exports to **`site/src/data/`** and **imports** the JSON (bundled at build time). This is simpler, type-safe, removes the base-path fetch gotcha, and needs no async loading state. `ratings.json` also gains a `mean_defense` field so the TS cold-start fallback exactly mirrors the Python `_params`.

## Prerequisites

- Node.js 18+ and npm. Check with `node -v`. If missing (Windows): `winget install OpenJS.NodeJS.LTS`, then reopen the terminal.
- The Python venv and data from the main project (`data/raw/results.csv`, `data/raw/fifa_ranking.csv`) must already exist (they do).

## File structure

```
WDC26/
├── scripts/export_site_data.py        # model → site/src/data/*.json
├── tests/test_export_site_data.py      # pytest for the exporter's pure functions
├── site/
│   ├── package.json · tsconfig.json · vite.config.ts · index.html
│   ├── src/
│   │   ├── main.tsx · App.tsx · index.css · setupTests.ts
│   │   ├── data/                        # GENERATED json + types.ts
│   │   │   ├── types.ts
│   │   │   ├── ratings.json · simulation.json · groups.json · meta.json · parity_fixtures.json
│   │   ├── lib/dixonColes.ts · lib/dixonColes.test.ts
│   │   └── components/
│   │       ├── ScorelineGrid.tsx · ScorelineGrid.test.tsx
│   │       ├── MatchupPredictor.tsx · MatchupPredictor.test.tsx
│   │       ├── TitleOddsChart.tsx · TitleOddsChart.test.tsx
│   │       ├── Hero.tsx · HowItWorks.tsx · Validation.tsx · Limitations.tsx · About.tsx
│   │       └── sections.test.tsx
└── .github/workflows/deploy-site.yml
```

---

## Task 1: Python data-export script

**Files:**
- Create: `scripts/export_site_data.py`
- Test: `tests/test_export_site_data.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
from scripts.export_site_data import (
    ratings_payload, simulation_payload, groups_payload, parity_payload)


def test_ratings_payload_shape(fitted_model):
    fit = {"min_date": "2010-01-01", "xi": 0.0019, "n_matches": 300}
    p = ratings_payload(fitted_model, fit)
    assert set(p) == {"teams", "attack", "defense", "home_adv", "rho",
                      "max_goals", "mean_defense", "fit"}
    assert set(p["attack"]) == set(fitted_model.teams)
    assert isinstance(p["attack"]["Strong"], float)
    assert p["attack"]["Strong"] > p["attack"]["Weak"]
    assert p["fit"] == fit


def test_simulation_payload_shape():
    df = pd.DataFrame([{"team": "Brazil", "p_r32": 0.99, "p_r16": 0.94, "p_qf": 0.77,
                        "p_sf": 0.6, "p_final": 0.43, "p_champion": 0.28}])
    p = simulation_payload(df, n_sims=10000, seed=2026)
    assert p["n_sims"] == 10000 and p["seed"] == 2026
    assert p["teams"][0]["team"] == "Brazil"
    assert p["teams"][0]["p_champion"] == 0.28


def test_groups_payload():
    p = groups_payload({"A": ["X", "Y"]})
    assert p == {"A": ["X", "Y"]}


def test_parity_payload(fitted_model):
    cases = parity_payload(fitted_model, [("Strong", "Weak")])
    c = cases[0]
    assert c["home"] == "Strong" and c["away"] == "Weak" and c["neutral"] is True
    assert abs(c["home_win"] + c["draw"] + c["away_win"] - 1.0) < 1e-9
    assert c["home_win"] > c["away_win"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_export_site_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.export_site_data'`.

- [ ] **Step 3: Implement `scripts/export_site_data.py`**

```python
"""Export the fitted Dixon-Coles model's outputs to JSON for the showcase site."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import load_results, clean_results, load_rankings, latest_rankings
from src.dixon_coles import DixonColesModel
from src.evaluate import (walk_forward_worldcups, evaluate, base_rate_probs,
                          result_outcome)
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


def meta_payload(matches, model):
    teams = list(model.teams)
    net = {t: float(model.attack[i] - model.defense[i]) for i, t in enumerate(teams)}
    ranks = latest_rankings(load_rankings(REPO / "data/raw/fifa_ranking.csv"))
    rr = pd.DataFrame({"net": [net[t] for t in teams],
                       "rank": [ranks.get(t, np.nan) for t in teams]}).dropna()
    corr = float(rr["net"].corr(rr["rank"]))
    wc_years = [2018, 2022]
    preds, outs = walk_forward_worldcups(matches, wc_years, DixonColesModel, xi=0.0019)
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
                     "best_xi": 0.0019},
        "model": {"home_adv": round(float(model.home_adv), 3),
                  "rho": round(float(model.rho), 3)},
    }


def _write(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", path.name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    matches = clean_results(load_results(REPO / "data/raw/results.csv"))
    fit_df = matches[matches.date >= "2010-01-01"]
    model = DixonColesModel().fit(fit_df, xi=0.0019)
    fit = {"min_date": "2010-01-01", "xi": 0.0019, "n_matches": int(len(fit_df))}
    _write(OUT / "ratings.json", ratings_payload(model, fit))
    probs = monte_carlo(GROUPS, model, n_sims=10000, seed=2026)
    _write(OUT / "simulation.json", simulation_payload(probs, 10000, 2026))
    _write(OUT / "groups.json", groups_payload(GROUPS))
    pairs = [("Brazil", "Croatia"), ("Argentina", "Mexico"), ("Spain", "Uruguay"),
             ("France", "Norway"), ("Germany", "Ecuador")]
    _write(OUT / "parity_fixtures.json", parity_payload(model, pairs))
    _write(OUT / "meta.json", meta_payload(matches, model))
    print("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_export_site_data.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_site_data.py tests/test_export_site_data.py
git commit -m "feat: site data-export script (model -> JSON)"
```

---

## Task 2: Scaffold the Vite React + TS app

**Files:**
- Create: `site/` (Vite scaffold), `site/vite.config.ts`, `site/src/setupTests.ts`
- Modify: `site/package.json`

- [ ] **Step 1: Scaffold and install**

Run (from repo root):
```bash
npm create vite@latest site -- --template react-ts
cd site && npm install && npm install recharts && npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom
```
Expected: `site/` created with a working Vite React-TS app; dependencies install.

- [ ] **Step 2: Overwrite `site/vite.config.ts`**

```ts
/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/WDC26/",
  plugins: [react()],
  test: { environment: "jsdom", globals: true, setupFiles: "./src/setupTests.ts" },
});
```

- [ ] **Step 3: Create `site/src/setupTests.ts`**

```ts
import "@testing-library/jest-dom";
```

- [ ] **Step 4: Add the test script to `site/package.json`**

In the `"scripts"` object add:
```json
"test": "vitest run"
```

- [ ] **Step 5: Verify the dev build and empty test run**

Run (from `site/`):
```bash
npm run build
npm run test
```
Expected: build succeeds; `vitest` reports "no test files found" (exit 0) or runs 0 tests.

- [ ] **Step 6: Commit**

```bash
git add site/ -- ':!site/node_modules'
git commit -m "chore: scaffold Vite React-TS site with vitest"
```
(Confirm `site/node_modules/` is ignored — the Vite template adds a `.gitignore`; if not, add `node_modules/` to it.)

---

## Task 3: Generate the site data

**Files:**
- Create (generated): `site/src/data/{ratings,simulation,groups,meta,parity_fixtures}.json`

- [ ] **Step 1: Run the exporter**

Run (from repo root): `.venv/Scripts/python scripts/export_site_data.py`
Expected: prints `wrote ratings.json` … `done`. Takes several minutes (it fits the model, runs 10,000 simulations, and computes the backtest) — run in the background if preferred. Five JSON files appear in `site/src/data/`.

- [ ] **Step 2: Sanity-check the output**

Run: `.venv/Scripts/python -c "import json; r=json.load(open('site/src/data/ratings.json',encoding='utf-8')); print(len(r['teams']),'teams; rho',r['rho']); s=json.load(open('site/src/data/simulation.json',encoding='utf-8')); print('top', s['teams'][0]['team'], s['teams'][0]['p_champion'])"`
Expected: prints the team count and the top title-odds team (Brazil, ~0.28).

- [ ] **Step 3: Commit the generated data**

```bash
git add site/src/data/*.json
git commit -m "feat: generate site data (ratings, simulation, groups, meta, parity)"
```

---

## Task 4: TypeScript data types

**Files:**
- Create: `site/src/data/types.ts`

- [ ] **Step 1: Create `site/src/data/types.ts`**

```ts
export interface Ratings {
  teams: string[];
  attack: Record<string, number>;
  defense: Record<string, number>;
  home_adv: number;
  rho: number;
  max_goals: number;
  mean_defense: number;
  fit: { min_date: string; xi: number; n_matches: number };
}

export interface TeamOdds {
  team: string;
  p_r32: number; p_r16: number; p_qf: number;
  p_sf: number; p_final: number; p_champion: number;
}

export interface Simulation { n_sims: number; seed: number; teams: TeamOdds[]; }

export type Groups = Record<string, string[]>;

export interface Meta {
  dataset: { n_matches: number; n_teams: number; date_min: string; date_max: string };
  ratings_sanity: { fifa_rank_corr: number; n_matched: number };
  backtest: {
    wc_years: number[]; n_matches: number;
    model: { rps: number; log_loss: number; accuracy: number };
    baseline: { rps: number; log_loss: number; accuracy: number };
    best_xi: number;
  };
  model: { home_adv: number; rho: number };
}

export interface ParityCase {
  home: string; away: string; neutral: boolean;
  home_win: number; draw: number; away_win: number;
}
```

- [ ] **Step 2: Verify it type-checks**

Run (from `site/`): `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add site/src/data/types.ts
git commit -m "feat: TS types for site data"
```

---

## Task 5: Dixon-Coles ported to TypeScript

**Files:**
- Create: `site/src/lib/dixonColes.ts`
- Test: `site/src/lib/dixonColes.test.ts`

- [ ] **Step 1: Write the failing test** (parity against the Python-exported fixtures)

```ts
import { describe, it, expect } from "vitest";
import ratings from "../data/ratings.json";
import parity from "../data/parity_fixtures.json";
import type { Ratings, ParityCase } from "../data/types";
import { scoreMatrix, predictResult, expectedGoals } from "./dixonColes";

const R = ratings as Ratings;

describe("dixonColes parity with Python", () => {
  it("reproduces predict_result for every fixture", () => {
    (parity as ParityCase[]).forEach((c) => {
      const res = predictResult(scoreMatrix(c.home, c.away, R, c.neutral));
      expect(res.homeWin).toBeCloseTo(c.home_win, 6);
      expect(res.draw).toBeCloseTo(c.draw, 6);
      expect(res.awayWin).toBeCloseTo(c.away_win, 6);
    });
  });

  it("score matrix sums to 1", () => {
    const m = scoreMatrix(R.teams[0], R.teams[1], R);
    const sum = m.flat().reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 9);
  });

  it("stronger expected goals for the better team", () => {
    const [lam, mu] = expectedGoals("Brazil", "Croatia", R);
    expect(lam).toBeGreaterThan(mu);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `site/`): `npx vitest run src/lib/dixonColes.test.ts`
Expected: FAIL — cannot find `./dixonColes`.

- [ ] **Step 3: Implement `site/src/lib/dixonColes.ts`**

```ts
import type { Ratings } from "../data/types";

export function poissonPmf(k: number, lambda: number): number {
  let fact = 1;
  for (let i = 2; i <= k; i++) fact *= i;
  return (Math.exp(-lambda) * Math.pow(lambda, k)) / fact;
}

function params(team: string, r: Ratings): [number, number] {
  if (Object.prototype.hasOwnProperty.call(r.attack, team)) {
    return [r.attack[team], r.defense[team]];
  }
  return [0, r.mean_defense];
}

export function expectedGoals(
  home: string, away: string, r: Ratings, neutral = true,
): [number, number] {
  const [aH, dH] = params(home, r);
  const [aA, dA] = params(away, r);
  const gamma = neutral ? 0 : r.home_adv;
  return [Math.exp(aH + dA + gamma), Math.exp(aA + dH)];
}

export function scoreMatrix(
  home: string, away: string, r: Ratings, neutral = true,
): number[][] {
  const [lam, mu] = expectedGoals(home, away, r, neutral);
  const n = r.max_goals + 1;
  const mat: number[][] = [];
  for (let x = 0; x < n; x++) {
    mat[x] = [];
    for (let y = 0; y < n; y++) mat[x][y] = poissonPmf(x, lam) * poissonPmf(y, mu);
  }
  mat[0][0] *= 1 - lam * mu * r.rho;
  mat[0][1] *= 1 + lam * r.rho;
  mat[1][0] *= 1 + mu * r.rho;
  mat[1][1] *= 1 - r.rho;
  let sum = 0;
  for (let x = 0; x < n; x++)
    for (let y = 0; y < n; y++) { if (mat[x][y] < 0) mat[x][y] = 0; sum += mat[x][y]; }
  for (let x = 0; x < n; x++)
    for (let y = 0; y < n; y++) mat[x][y] /= sum;
  return mat;
}

export interface Result { homeWin: number; draw: number; awayWin: number; }

export function predictResult(mat: number[][]): Result {
  let homeWin = 0, draw = 0, awayWin = 0;
  for (let x = 0; x < mat.length; x++)
    for (let y = 0; y < mat[x].length; y++) {
      if (x > y) homeWin += mat[x][y];
      else if (x === y) draw += mat[x][y];
      else awayWin += mat[x][y];
    }
  return { homeWin, draw, awayWin };
}

export function mostLikelyScore(mat: number[][]): { home: number; away: number; prob: number } {
  let best = -1, home = 0, away = 0;
  for (let x = 0; x < mat.length; x++)
    for (let y = 0; y < mat[x].length; y++)
      if (mat[x][y] > best) { best = mat[x][y]; home = x; away = y; }
  return { home, away, prob: best };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `site/`): `npx vitest run src/lib/dixonColes.test.ts`
Expected: PASS (3 tests) — the TS model matches the Python fixtures.

- [ ] **Step 5: Commit**

```bash
git add site/src/lib/dixonColes.ts site/src/lib/dixonColes.test.ts
git commit -m "feat: Dixon-Coles ported to TS with Python-parity tests"
```

---

## Task 6: ScorelineGrid component

**Files:**
- Create: `site/src/components/ScorelineGrid.tsx`
- Test: `site/src/components/ScorelineGrid.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ScorelineGrid } from "./ScorelineGrid";

describe("ScorelineGrid", () => {
  it("renders a cell for each scoreline up to display size", () => {
    const m = Array.from({ length: 3 }, () => [0.2, 0.1, 0.0]);
    render(<ScorelineGrid matrix={m} homeName="A" awayName="B" display={3} />);
    expect(screen.getByTestId("scoreline-grid")).toBeInTheDocument();
    // 3x3 = 9 probability cells
    expect(screen.getAllByTestId("grid-cell")).toHaveLength(9);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `site/`): `npx vitest run src/components/ScorelineGrid.test.tsx`
Expected: FAIL — cannot find `./ScorelineGrid`.

- [ ] **Step 3: Implement `site/src/components/ScorelineGrid.tsx`**

```tsx
interface Props {
  matrix: number[][];
  homeName: string;
  awayName: string;
  display?: number;
}

const REGION = {
  home: { bg: "#e6f1eb", fg: "#1a7f5a" },
  draw: { bg: "#f4efe2", fg: "#8a6d1a" },
  away: { bg: "#f6eae7", fg: "#a8553a" },
};

export function ScorelineGrid({ matrix, homeName, awayName, display = 6 }: Props) {
  const n = Math.min(display, matrix.length);
  const cols = `36px repeat(${n}, 1fr)`;
  const cells = [];
  // header row
  cells.push(<div key="corner" />);
  for (let y = 0; y < n; y++)
    cells.push(<div key={`h${y}`} style={{ textAlign: "center", fontSize: 12, color: "#888" }}>{y}</div>);
  for (let x = 0; x < n; x++) {
    cells.push(<div key={`r${x}`} style={{ display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, color: "#888" }}>{x}</div>);
    for (let y = 0; y < n; y++) {
      const region = x > y ? REGION.home : x === y ? REGION.draw : REGION.away;
      const pct = matrix[x][y] * 100;
      cells.push(
        <div key={`${x}-${y}`} data-testid="grid-cell"
          style={{ background: region.bg, color: region.fg, borderRadius: 4, height: 34,
                   display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12 }}>
          {pct >= 0.5 ? Math.round(pct) : ""}
        </div>,
      );
    }
  }
  return (
    <div>
      <div style={{ fontSize: 13, color: "#666", marginBottom: 6 }}>
        {awayName} goals across (→), {homeName} goals down (↓)
      </div>
      <div data-testid="scoreline-grid"
        style={{ display: "grid", gridTemplateColumns: cols, gap: 3, maxWidth: 480 }}>
        {cells}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `site/`): `npx vitest run src/components/ScorelineGrid.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add site/src/components/ScorelineGrid.tsx site/src/components/ScorelineGrid.test.tsx
git commit -m "feat: ScorelineGrid component"
```

---

## Task 7: MatchupPredictor component (interactive centerpiece)

**Files:**
- Create: `site/src/components/MatchupPredictor.tsx`
- Test: `site/src/components/MatchupPredictor.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MatchupPredictor } from "./MatchupPredictor";

describe("MatchupPredictor", () => {
  it("shows win/draw/loss odds that sum to ~100%", () => {
    render(<MatchupPredictor />);
    const odds = screen.getByTestId("wdl-odds").textContent || "";
    const nums = (odds.match(/\d+/g) || []).map(Number);
    const total = nums.reduce((a, b) => a + b, 0);
    expect(total).toBeGreaterThanOrEqual(98);
    expect(total).toBeLessThanOrEqual(102);
  });

  it("updates when the away team changes", () => {
    render(<MatchupPredictor />);
    const before = screen.getByTestId("wdl-odds").textContent;
    const away = screen.getByLabelText("Away team") as HTMLSelectElement;
    const other = Array.from(away.options).find((o) => o.value !== away.value)!;
    fireEvent.change(away, { target: { value: other.value } });
    expect(screen.getByTestId("wdl-odds").textContent).not.toEqual(before);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `site/`): `npx vitest run src/components/MatchupPredictor.test.tsx`
Expected: FAIL — cannot find `./MatchupPredictor`.

- [ ] **Step 3: Implement `site/src/components/MatchupPredictor.tsx`**

```tsx
import { useMemo, useState } from "react";
import ratings from "../data/ratings.json";
import type { Ratings } from "../data/types";
import { scoreMatrix, predictResult, mostLikelyScore } from "../lib/dixonColes";
import { ScorelineGrid } from "./ScorelineGrid";

const R = ratings as Ratings;
const TEAMS = [...R.teams].sort();

export function MatchupPredictor() {
  const [home, setHome] = useState("Brazil");
  const [away, setAway] = useState("Croatia");

  const { matrix, result, top } = useMemo(() => {
    const m = scoreMatrix(home, away, R, true);
    return { matrix: m, result: predictResult(m), top: mostLikelyScore(m) };
  }, [home, away]);

  const pct = (x: number) => Math.round(x * 100);

  return (
    <div>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 20 }}>
        <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
          Home team
          <select aria-label="Home team" value={home} onChange={(e) => setHome(e.target.value)}>
            {TEAMS.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
          Away team
          <select aria-label="Away team" value={away} onChange={(e) => setAway(e.target.value)}>
            {TEAMS.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
      </div>

      <div data-testid="wdl-odds" style={{ display: "flex", gap: 20, marginBottom: 8, fontSize: 16 }}>
        <span><strong>{pct(result.homeWin)}%</strong> {home}</span>
        <span><strong>{pct(result.draw)}%</strong> draw</span>
        <span><strong>{pct(result.awayWin)}%</strong> {away}</span>
      </div>
      <p style={{ fontSize: 14, color: "#666", marginTop: 0 }}>
        Most likely score: {home} {top.home}–{top.away} {away} ({(top.prob * 100).toFixed(1)}%)
      </p>

      <ScorelineGrid matrix={matrix} homeName={home} awayName={away} display={7} />
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `site/`): `npx vitest run src/components/MatchupPredictor.test.tsx`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add site/src/components/MatchupPredictor.tsx site/src/components/MatchupPredictor.test.tsx
git commit -m "feat: interactive MatchupPredictor (model in the browser)"
```

---

## Task 8: TitleOddsChart component

**Files:**
- Create: `site/src/components/TitleOddsChart.tsx`
- Test: `site/src/components/TitleOddsChart.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { TitleOddsChart } from "./TitleOddsChart";

describe("TitleOddsChart", () => {
  it("renders the top team's name", () => {
    render(<TitleOddsChart topN={5} />);
    expect(screen.getByText("Brazil")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `site/`): `npx vitest run src/components/TitleOddsChart.test.tsx`
Expected: FAIL — cannot find `./TitleOddsChart`.

- [ ] **Step 3: Implement `site/src/components/TitleOddsChart.tsx`**

```tsx
import { BarChart, Bar, XAxis, YAxis, Cell, ResponsiveContainer, LabelList } from "recharts";
import simulation from "../data/simulation.json";
import type { Simulation } from "../data/types";

const SIM = simulation as Simulation;

export function TitleOddsChart({ topN = 12 }: { topN?: number }) {
  const data = SIM.teams.slice(0, topN).map((t) => ({
    team: t.team, pct: Math.round(t.p_champion * 1000) / 10,
  }));
  return (
    <ResponsiveContainer width="100%" height={topN * 34 + 40}>
      <BarChart data={data} layout="vertical" margin={{ left: 12, right: 36 }}>
        <XAxis type="number" hide domain={[0, "dataMax"]} />
        <YAxis type="category" dataKey="team" width={92} tickLine={false} axisLine={false}
               tick={{ fontSize: 13 }} />
        <Bar dataKey="pct" radius={[2, 2, 2, 2]}>
          {data.map((_, i) => <Cell key={i} fill="#1a7f5a" />)}
          <LabelList dataKey="pct" position="right" formatter={(v: number) => `${v}%`}
                     style={{ fontSize: 12, fill: "#444" }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `site/`): `npx vitest run src/components/TitleOddsChart.test.tsx`
Expected: PASS. (If Recharts' ResponsiveContainer warns about 0 width in jsdom, the test still finds the YAxis tick text "Brazil"; the warning is harmless.)

- [ ] **Step 5: Commit**

```bash
git add site/src/components/TitleOddsChart.tsx site/src/components/TitleOddsChart.test.tsx
git commit -m "feat: TitleOddsChart (Recharts)"
```

---

## Task 9: Content sections (Hero, HowItWorks, Validation, Limitations, About)

**Files:**
- Create: `site/src/components/Hero.tsx`, `HowItWorks.tsx`, `Validation.tsx`, `Limitations.tsx`, `About.tsx`
- Test: `site/src/components/sections.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Hero } from "./Hero";
import { Validation } from "./Validation";
import { Limitations } from "./Limitations";
import { About } from "./About";

describe("content sections", () => {
  it("Hero shows the headline question", () => {
    render(<Hero />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/2026 World Cup/i);
  });
  it("Validation shows the model and baseline RPS", () => {
    render(<Validation />);
    expect(screen.getByText(/RPS/i)).toBeInTheDocument();
  });
  it("Limitations mentions South America", () => {
    render(<Limitations />);
    expect(screen.getByText(/South American/i)).toBeInTheDocument();
  });
  it("About links to the GitHub repo", () => {
    render(<About />);
    const link = screen.getByRole("link", { name: /github/i });
    expect(link).toHaveAttribute("href", expect.stringContaining("github.com/Kojur/WDC26"));
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `site/`): `npx vitest run src/components/sections.test.tsx`
Expected: FAIL — cannot find `./Hero`.

- [ ] **Step 3: Implement the five components**

`site/src/components/Hero.tsx`:
```tsx
import simulation from "../data/simulation.json";
import type { Simulation } from "../data/types";

const top = (simulation as Simulation).teams[0];

export function Hero() {
  return (
    <header className="hero">
      <h1>Who will win the 2026 World Cup?</h1>
      <p className="lede">
        A Dixon-Coles statistical model, trained on 150 years of international football and
        simulated 10,000 times — explained simply, and yours to play with.
      </p>
      <p className="headline-stat">
        <strong>{Math.round(top.p_champion * 100)}%</strong> {top.team} to lift the cup
      </p>
    </header>
  );
}
```

`site/src/components/HowItWorks.tsx`:
```tsx
export function HowItWorks() {
  return (
    <section className="section">
      <h2>How it works</h2>
      <p>
        Every national team gets two numbers learned from history: an <em>attack</em> rating
        (how many goals it tends to score) and a <em>defense</em> rating (how few it concedes).
        For any matchup, those combine into an expected number of goals for each side.
      </p>
      <p>
        We then treat goals as a Poisson process — turning the two expected-goal numbers into a
        full grid of scoreline probabilities (a 2-1, a 0-0, and so on). Summing the right cells
        of that grid gives the chance of a win, draw, or loss.
      </p>
      <p>
        To predict the whole tournament, we simulate all 104 matches — group stage through final
        — <strong>ten thousand times</strong>, and count how often each team lifts the trophy.
      </p>
    </section>
  );
}
```

`site/src/components/Validation.tsx`:
```tsx
import meta from "../data/meta.json";
import type { Meta } from "../data/types";

const M = meta as Meta;

export function Validation() {
  const b = M.backtest;
  return (
    <section className="section">
      <h2>Does it actually work?</h2>
      <p>
        Tested honestly — trained only on matches <em>before</em> each World Cup, then asked to
        predict it ({b.n_matches} matches across {b.wc_years.join(" & ")}). It beats a sensible
        baseline on every measure:
      </p>
      <table className="metrics">
        <thead><tr><th></th><th>Model</th><th>Baseline</th></tr></thead>
        <tbody>
          <tr><td>RPS (lower better)</td><td>{b.model.rps}</td><td>{b.baseline.rps}</td></tr>
          <tr><td>Log-loss (lower better)</td><td>{b.model.log_loss}</td><td>{b.baseline.log_loss}</td></tr>
          <tr><td>Accuracy</td><td>{Math.round(b.model.accuracy * 100)}%</td><td>{Math.round(b.baseline.accuracy * 100)}%</td></tr>
        </tbody>
      </table>
      <p>
        As a sanity check, the model's learned strength correlates {M.ratings_sanity.fifa_rank_corr}
        with the official FIFA ranking — without ever being shown it.
      </p>
    </section>
  );
}
```

`site/src/components/Limitations.tsx`:
```tsx
export function Limitations() {
  return (
    <section className="section">
      <h2>The catch</h2>
      <p>
        The model rates South American sides unusually highly — five of its top six. That isn't a
        bug so much as a property of learning purely from results: CONMEBOL teams play each other
        constantly in a brutal round-robin, while Europe's giants rack up lopsided qualifying wins
        that the model can't fully weigh. It has no notion of confederation strength.
      </p>
      <p>
        It also can't see what isn't in the scoreline data: injuries, squad turnover, form, or
        tactics. Treat the numbers as an informed baseline, not a crystal ball.
      </p>
    </section>
  );
}
```

`site/src/components/About.tsx`:
```tsx
import meta from "../data/meta.json";
import type { Meta } from "../data/types";

const M = meta as Meta;

export function About() {
  return (
    <section className="section about">
      <h2>Behind the scenes</h2>
      <p>
        Built from {M.dataset.n_matches.toLocaleString()} international matches
        ({M.dataset.date_min} to {M.dataset.date_max}). The model (Dixon-Coles + Monte Carlo) is
        written in Python; this page is React, with the model re-implemented in TypeScript so the
        predictor runs in your browser.
      </p>
      <p>
        <a href="https://github.com/Kojur/WDC26">Source &amp; notebooks on GitHub</a>
      </p>
    </section>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run (from `site/`): `npx vitest run src/components/sections.test.tsx`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add site/src/components/Hero.tsx site/src/components/HowItWorks.tsx site/src/components/Validation.tsx site/src/components/Limitations.tsx site/src/components/About.tsx site/src/components/sections.test.tsx
git commit -m "feat: content sections (hero, method, validation, limitations, about)"
```

---

## Task 10: Compose App + editorial styling

**Files:**
- Modify: `site/src/App.tsx`
- Create/replace: `site/src/index.css`
- Test: `site/src/App.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "./App";

describe("App", () => {
  it("renders the hero and the matchup predictor", () => {
    render(<App />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/2026 World Cup/i);
    expect(screen.getByLabelText("Home team")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `site/`): `npx vitest run src/App.test.tsx`
Expected: FAIL — App doesn't render the predictor yet.

- [ ] **Step 3: Replace `site/src/App.tsx`**

```tsx
import { Hero } from "./components/Hero";
import { TitleOddsChart } from "./components/TitleOddsChart";
import { HowItWorks } from "./components/HowItWorks";
import { MatchupPredictor } from "./components/MatchupPredictor";
import { Validation } from "./components/Validation";
import { Limitations } from "./components/Limitations";
import { About } from "./components/About";
import "./index.css";

export default function App() {
  return (
    <main className="page">
      <Hero />
      <section className="section">
        <h2>The title race</h2>
        <p>Each team's chance of winning the tournament, across 10,000 simulations:</p>
        <TitleOddsChart topN={12} />
      </section>
      <HowItWorks />
      <section className="section">
        <h2>Try it yourself</h2>
        <p>Pick any two teams. The model runs right here in your browser.</p>
        <MatchupPredictor />
      </section>
      <Validation />
      <Limitations />
      <About />
    </main>
  );
}
```

- [ ] **Step 4: Replace `site/src/index.css`** (editorial theme)

```css
:root {
  --ink: #1a1a1a;
  --muted: #666;
  --accent: #1a7f5a;
  --rule: #e7e7e2;
  --bg: #fdfdfb;
  --serif: Georgia, "Times New Roman", serif;
  --sans: system-ui, "Segoe UI", Helvetica, Arial, sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans);
       line-height: 1.7; }
.page { max-width: 720px; margin: 0 auto; padding: 32px 20px 80px; }
h1 { font-family: var(--serif); font-size: 40px; line-height: 1.15; margin: 0 0 12px; }
h2 { font-family: var(--serif); font-size: 26px; margin: 0 0 12px; }
.hero { padding: 40px 0; border-bottom: 1px solid var(--rule); margin-bottom: 24px; }
.lede { font-size: 18px; color: var(--muted); max-width: 60ch; }
.headline-stat { font-family: var(--serif); font-size: 22px; margin-top: 16px; }
.headline-stat strong { font-size: 34px; color: var(--accent); }
.section { padding: 28px 0; border-bottom: 1px solid var(--rule); }
a { color: var(--accent); }
select { font: inherit; padding: 6px 8px; border: 1px solid var(--rule); border-radius: 6px;
         background: #fff; margin-top: 4px; }
.metrics { border-collapse: collapse; margin: 12px 0; font-size: 15px; }
.metrics th, .metrics td { text-align: left; padding: 6px 18px 6px 0; border-bottom: 1px solid var(--rule); }
.about { border-bottom: none; }
@media (max-width: 520px) { h1 { font-size: 30px; } .page { padding: 20px 16px 60px; } }
```

- [ ] **Step 5: Run test + full suite + build**

Run (from `site/`):
```bash
npx vitest run
npm run build
```
Expected: all Vitest tests pass; `npm run build` succeeds (static output in `site/dist/`).

- [ ] **Step 6: Commit**

```bash
git add site/src/App.tsx site/src/index.css site/src/App.test.tsx
git commit -m "feat: compose page and apply editorial styling"
```

---

## Task 11: GitHub Pages deployment

**Files:**
- Create: `.github/workflows/deploy-site.yml`
- Modify: `README.md` (add a link to the live site)

- [ ] **Step 1: Create `.github/workflows/deploy-site.yml`**

```yaml
name: Deploy site
on:
  push:
    branches: [main]
    paths: ["site/**", ".github/workflows/deploy-site.yml"]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: true
jobs:
  build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: site
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
          cache-dependency-path: site/package-lock.json
      - run: npm ci
      - run: npm run test
      - run: npm run build
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site/dist
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Add a live-site link to `README.md`**

Add near the top of `README.md`, after the title line:
```markdown
**🔗 Live showcase: https://kojur.github.io/WDC26/**
```

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/deploy-site.yml README.md
git commit -m "ci: deploy site to GitHub Pages"
git push origin main
```

- [ ] **Step 4: Enable Pages (one-time, manual)**

In the GitHub repo: Settings → Pages → Build and deployment → Source = **GitHub Actions**. Then re-run the workflow (Actions tab → Deploy site → Run workflow) if it didn't trigger. Verify the site loads at `https://kojur.github.io/WDC26/`.

---

## Task 12: Final verification

- [ ] **Step 1: Full Python suite still green**

Run (from repo root): `.venv/Scripts/python -m pytest -q`
Expected: all tests pass (includes the new `test_export_site_data.py`).

- [ ] **Step 2: Full site suite + build**

Run (from `site/`): `npx vitest run && npm run build`
Expected: all tests pass; build succeeds.

- [ ] **Step 3: Manual smoke (local preview)**

Run (from `site/`): `npm run preview`
Open the printed URL; confirm: hero stat shows ~28% Brazil; title-odds chart renders; switching teams in the predictor updates the grid and odds; the validation table shows the model beating the baseline.

- [ ] **Step 4: Confirm deployment**

Verify `https://kojur.github.io/WDC26/` is live and matches the local preview.

---

## Self-review (completed by plan author)

**Spec coverage:**
- §3 narrative (7 sections) → Hero/TitleOdds (Task 8/10), HowItWorks (9), MatchupPredictor (7), Validation/Limitations/About (9). ✓
- §4 editorial visual design → Task 10 `index.css`. ✓
- §5 architecture (export script, src/data JSON, dixonColes.ts port, components, Recharts, deploy) → Tasks 1–3, 5, 6–9, 8, 11. ✓
- §5 data contracts (ratings/simulation/groups/meta + mean_defense) → Task 1 payloads + Task 4 types. ✓
- §6 testing (Vitest parity vs Python fixtures, data-shape, build check) → Task 5 parity, Task 1 exporter tests, Tasks 10/12 build. ✓
- §7 risks (parity drift → parity test; base path → `base:'/WDC26/'` + bundled imports; staleness → re-run exporter; scope → fixed 7 sections). ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. The only manual non-code step (Task 11 Step 4, enabling Pages) is an unavoidable one-time GitHub UI action, fully described.

**Type/name consistency:** `Ratings`, `Simulation`, `TeamOdds`, `Meta`, `ParityCase` (Task 4) are used consistently in Tasks 5–10. `scoreMatrix`/`predictResult`/`expectedGoals`/`mostLikelyScore` (Task 5) match their call sites (Tasks 6–7). Exporter payload keys (Task 1) match the TS types (Task 4) and the `meta.json`/`simulation.json`/`ratings.json` field reads in Tasks 8–9. `mean_defense` is produced (Task 1) and consumed (Task 5).
