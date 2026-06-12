# WC2026 Showcase Site — Design Spec

**Date:** 2026-06-12
**Status:** Approved design, pending implementation plan
**Type:** Portfolio web project (front-end + a Python data-export step)

## 1. Goal & scope

Publish the existing WDC26 World Cup prediction analysis as a polished, public-facing
**single-page website** that (a) tells the story and methodology in plain English and
(b) lets readers **play with the model** via an interactive matchup predictor. It is a
**portfolio piece** aimed at recruiters and technically-curious readers.

The site is fully **static**: a Python script exports the model's outputs to JSON, and a
React app consumes them. The Dixon-Coles model is re-implemented in TypeScript so the
matchup predictor runs entirely in the reader's browser — no backend.

Non-goals (YAGNI): backend/server, live-updating-during-the-tournament tracker, separate
interactive "ratings explorer" / "bracket explorer" tools (those results are shown as
*static* content), user accounts, a custom domain (can be added later).

## 2. Decisions (locked in during brainstorming)

| Decision | Choice |
|----------|--------|
| Primary audience | Portfolio / recruiters (plus curious readers) |
| Reader experience | Interactive explorer; the **matchup predictor** is the one interactive feature |
| Front-end stack | **React + Vite**, static-exported |
| Visual direction | **Editorial** — clean white, serif headlines, generous whitespace, charts as the stars |
| Hosting | GitHub Pages, same repo, served at `https://kojur.github.io/WDC26/` |
| Model in browser | Dixon-Coles ported to TypeScript (live matchup compute) |

## 3. Page structure & narrative (top to bottom)

1. **Hero** — "Who will win the 2026 World Cup?" + one-line hook + the headline answer
   (Brazil ~28%) + scroll cue.
2. **The title race** — editorial title-odds bar chart (top contenders) + narrative lead-in.
3. **How it works** — plain-English method: each team has *attack* & *defense* ratings
   learned from ~49k matches → goals as Poisson → a scoreline grid → 10,000 simulated
   tournaments. Includes a small static scoreline-grid illustration.
4. **Try it yourself — Matchup Predictor** *(interactive centerpiece)* — two team pickers →
   live scoreline grid + win/draw/loss odds + most-likely score, computed in-browser.
5. **Does it actually work?** — validation: walk-forward backtest beats the baseline (metrics
   table) and −0.92 correlation with FIFA rankings.
6. **The catch** — honest limitations: the South-American bias explained, plus what the model
   can't see (injuries, squad changes, no confederation adjustment).
7. **About / behind the scenes** — one-line tech summary; links to the GitHub repo, the
   notebooks, and the author's name/portfolio.

## 4. Visual design (editorial direction)

- Light/white background, dark near-black text, a single restrained accent colour (deep
  green, evoking the pitch) used sparingly for chart bars and links.
- **Serif** display headlines (e.g. a Georgia-class or Google "Source Serif"/"Lora"); clean
  sans for body and UI. Large type scale, generous line-height, comfortable measure (~65ch).
- Generous vertical whitespace between sections; charts and the scoreline grid are the focal
  visual elements. No heavy chrome, gradients, or shadows.
- Fully responsive (mobile-first); the matchup predictor and charts reflow on small screens.

## 5. Technical architecture

### Repo layout (additions only; existing `src/`, `notebooks/`, `tests/` untouched)
```
WDC26/
├── scripts/export_site_data.py     # model → JSON (run manually; output committed)
├── site/                            # React + Vite app
│   ├── public/data/                 # ratings.json, simulation.json, groups.json, meta.json
│   ├── src/
│   │   ├── lib/dixonColes.ts         # model ported to TS
│   │   ├── data/types.ts             # TS types for the JSON
│   │   ├── components/               # Hero, TitleOddsChart, HowItWorks,
│   │   │                             # MatchupPredictor, ScorelineGrid,
│   │   │                             # Validation, Limitations, About
│   │   ├── App.tsx · main.tsx
│   ├── index.html · vite.config.ts · package.json · tsconfig.json
└── .github/workflows/deploy-site.yml
```

### Data export — `scripts/export_site_data.py`
Reuses the existing `src/` modules. Fits the model on 2010+ data (`xi=0.0019`), runs (or
loads) the simulation, and writes four JSON files into `site/public/data/`:

- **`ratings.json`** — everything the in-browser model needs:
  ```json
  { "teams": ["Argentina","Brazil", ...],
    "attack": {"Brazil": 0.55, ...}, "defense": {"Brazil": -0.50, ...},
    "home_adv": 0.284, "rho": -0.139, "max_goals": 10,
    "fit": {"min_date":"2010-01-01","xi":0.0019,"n_matches":15817} }
  ```
- **`simulation.json`** — per-team odds:
  ```json
  { "n_sims":10000, "seed":2026,
    "teams":[ {"team":"Brazil","p_r32":0.9925,"p_r16":0.9415,"p_qf":0.7692,
               "p_sf":0.5967,"p_final":0.4272,"p_champion":0.2827}, ... ] }
  ```
- **`groups.json`** — `{ "A": ["Mexico", ...], ..., "L": [...] }`.
- **`meta.json`** — dataset stats, the FIFA-rank correlation, and backtest metrics
  (model vs baseline, `best_xi`) for the validation & about sections.

### The model in the browser — `src/lib/dixonColes.ts`
A faithful port of the Python: `poissonPmf(k, λ)`, `dcTau(x,y,λ,μ,ρ)`,
`expectedGoals(home, away, ratings, neutral=true)`, `scoreMatrix(...)` (returns an
`(max_goals+1)²` matrix, tau-corrected, clipped non-negative, normalized), and
`predictResult(matrix)` → `{ homeWin, draw, awayWin }`. Unseen teams fall back to the
global-average rating, mirroring the Python `_params`.

### Components
- **Hero** — headline, hook, top result, scroll cue.
- **TitleOddsChart** — Recharts horizontal bar of top-N title odds (`simulation.json`).
- **HowItWorks** — methodology prose + a static `ScorelineGrid` example.
- **MatchupPredictor** — two `<select>`s → `scoreMatrix` in-browser → `ScorelineGrid` +
  W/D/L odds + most-likely score.
- **ScorelineGrid** — reusable colored grid (home-win / draw / away-win regions), props are
  either a matrix or two team names + ratings.
- **Validation** — backtest metrics table + FIFA-rank correlation (`meta.json`).
- **Limitations** — prose (South-American bias, blind spots).
- **About** — tech summary + links.

### Charts
Recharts for the editorial bar chart; `ScorelineGrid` is a bespoke CSS-grid component
(not a chart-lib chart).

### Deployment
`.github/workflows/deploy-site.yml` builds `site/` and publishes to GitHub Pages on push to
`main`. `vite.config.ts` sets `base: '/WDC26/'`. Served at `https://kojur.github.io/WDC26/`.

## 6. Testing

- **Vitest parity tests** for `dixonColes.ts`: load the committed `ratings.json` and assert
  the TS model reproduces the Python model's numbers within tolerance (e.g. Brazil v Croatia
  ≈ {home 0.70, draw 0.22, away 0.08}); `scoreMatrix` sums to 1; the stronger team is
  favoured. A tiny `parity_fixtures.json` (exported by the Python script) holds the expected
  values so the test is self-contained.
- **Data-shape smoke test** — the four JSON files load and have the expected keys.
- **Build check** — `vite build` succeeds (run in CI before deploy).

## 7. Risks & gotchas

- **TS/Python parity drift** — guarded by the Vitest parity test against exported fixtures.
- **GitHub Pages base path** — wrong `base` 404s all assets; set `base: '/WDC26/'` and use
  relative data fetches (`import.meta.env.BASE_URL + 'data/...'`).
- **Data staleness** — `export_site_data.py` must be re-run (and JSON re-committed) whenever
  the model or simulation changes; the site never recomputes the simulation.
- **Scope creep** — keep to the seven sections and the single interactive feature.
```
