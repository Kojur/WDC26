# FIFA-Calibrated Strength Blend — Design Spec

**Date:** 2026-06-13
**Status:** Approved design, pending implementation plan
**Type:** Modeling change (Python core + data export; showcase copy)

## 1. Problem

The Monte Carlo champion odds are driven entirely by each team's learned **net strength**
(`attack − defense`) from a single global Dixon-Coles fit. That fit over-rates the CONMEBOL
(South American) pool, producing an implausible top order:

| Rank | Team | Net strength | Champion odds |
|------|------|-------------|---------------|
| 1 | Brazil | 2.38 | 28.3% |
| 2 | Argentina | 2.25 | 20.9% |
| 3 | **Colombia** | **2.02** | **10.6%** |
| 4 | Spain | 1.87 | 8.4% |
| 5 | **Uruguay** | **1.75** | 5.0% |
| 6 | **Ecuador** | **1.62** | 3.5% |
| 7 | France | 1.61 | 3.4% |

So the model believes Colombia > Spain and Uruguay/Ecuador > France. This is a faithful
report of a **biased rating**, not a simulation bug.

**Root cause — closed-pool / confederation miscalibration.** CONMEBOL is a 10-team pool that
plays a long home-and-away qualifying round-robin plus Copa América — a high volume of
competitive intra-pool matches. The top of the pool (Brazil, Argentina) is genuinely elite,
and Colombia/Uruguay/Ecuador regularly take points off them, so the model infers they are
nearly elite too. There are relatively few cross-confederation games to anchor the pool's
*absolute* level back down against Europe, so the whole pool floats up. Contributing factors:
friendlies are weighted equally with competitive games (Europe's giants rest players in
friendlies), and a single global home-advantage term can't model altitude, so altitude wins
are absorbed as raw "team strength."

The README's −0.92 FIFA correlation is a *global* rank correlation dominated by the
minnows-vs-giants spread; it looks great while hiding this top-end miscalibration.

## 2. Goal & success criteria

Rebalance the evaluation so the strongest sides surface sensibly, judged by **both**:

1. **Accuracy preserved** — the change must not hurt the walk-forward backtest (log-loss /
   RPS on 2018 + 2022 World Cups, vs the base-rate baseline).
2. **Sane top order** — the top ~6 should resemble real title contenders (Brazil, Argentina,
   Spain, France, England, Portugal), not South-American-heavy.

This rules out hand-tuning numbers to taste: every change is validated on the held-out
backtest.

## 3. Decisions (locked in during brainstorming)

| Decision | Choice |
|----------|--------|
| Success standard | Both: backtest accuracy preserved **and** sane top order |
| Narrative | **Allow an external prior** — FIFA ranking becomes an explicit, documented ingredient; reframe story as "data-driven rating, calibrated against FIFA" |
| Approach | **Post-fit FIFA blend** (shrinkage on net strength), α tuned by backtest |
| Prior signal | FIFA `total_points` (continuous), as-of aware |

Approaches considered and rejected: (2) prior baked into the MLE — more principled but
invasive to the core fit + Python↔TS parity tests, harder to explain; (3) confederation
recalibration + match-importance weighting — root-cause but needs a full confederation map,
leans on a small/noisy cross-confederation sample, and may only partly close the gap.

## 4. The prior signal

Use FIFA `total_points` (e.g. France 1837, Spain 1730, Colombia 1669, Ecuador 1518) rather
than ordinal rank — gaps between teams carry real information.

Extend `src/data.py` with `latest_points(rankings, as_of=None)` returning
`{country_full: total_points}` from the most recent snapshot on/before `as_of`, mirroring the
existing `latest_rankings`. The `as_of` parameter is **critical**: the backtest must use the
FIFA ranking as it stood *before* each past World Cup, so the prior introduces no leakage.

Data note: the FIFA CSV snapshot ends **2024-06-20**, ~2 years stale for a 2026 prediction.
This is a disclosed data limitation, not a blocker.

## 5. The blend — new module `src/calibrate.py`

A pure function (no side effects on the input model):

```
calibrate(model, fifa_points: dict, alpha: float) -> DixonColesModel  # calibrated copy
```

1. **Reference set** = teams present in **both** the model and `fifa_points`. Non-FIFA junk
   entries ("Yorkshire", "Sealand", etc.) get no prior and are left untouched.
2. Compute `net_t = attack_t − defense_t` for the reference set.
3. Standardize both signals over the reference set:
   `z_model = (net − mean_net)/std_net`, `z_fifa = (pts − mean_pts)/std_pts`.
4. Blend: `z_final = α·z_model + (1−α)·z_fifa`.
5. Map back to the model's **native net scale**: `net_final = mean_net + std_net·z_final`
   (so Poisson goal rates stay sane).
6. Apply per-team change `Δ_t = net_final_t − net_t`, split evenly:
   `attack_t += Δ_t/2`, `defense_t −= Δ_t/2`.
7. Return a calibrated copy; `home_adv`, `rho`, `max_goals` unchanged. Originals untouched.

α = 1 recovers today's model exactly; α = 0 reproduces the FIFA ordering on the reference set.
The even attack/defense split is a deliberate modeling choice (keeps total goals stable while
shifting overall strength); documented as such.

## 6. Choosing α (the adjudication step)

Grid-search α ∈ {0.0, 0.1, …, 1.0}. For each α, run the existing `walk_forward_worldcups`
on 2018 + 2022 **with the blend applied using as-of FIFA points**, and record log-loss + RPS.

Decision rule:

- Pick the α minimizing backtest **log-loss** (primary; RPS as secondary tiebreak).
- **If several α are within noise** (log-loss differences tiny — likely, since the blend is
  gentle), pick the one inside that indifferent band with the sanest top order. Report the
  full α-vs-loss curve so the choice is transparent.
- **Contingency:** if the backtest *strictly* prefers α = 1 (pure model) and blending
  measurably hurts, that is a genuine accuracy-vs-plausibility tension — surface the curve and
  the trade-off to the user rather than silently picking. (Expectation from the design preview:
  the curve is flat-to-improved around α ≈ 0.4–0.6.)

Design preview (net-strength reordering only, no full sim) at α = 0.5 gives a top-8 of
**Argentina, Brazil, France, Spain, England, Belgium, Colombia, Portugal** — France and Spain
above Colombia, Ecuador/Uruguay out of the top tier, Colombia retained as a respectable #7.

## 7. Code touch-points

- `src/data.py` — add `latest_points(as_of=…)`.
- `src/calibrate.py` — **new**, the blend (section 5).
- `src/evaluate.py` — let `walk_forward_worldcups` optionally apply the blend (accept FIFA
  points + α), used for α-tuning.
- `scripts/export_site_data.py` — fit → `calibrate(α*)` → simulate; export chosen α*, the
  α-vs-loss curve, and a before/after top-12 into `meta.json`.
- **TS side: no logic change.** The browser predictor reads `attack`/`defense` from
  `ratings.json`; since we export *calibrated* ratings, the matchup predictor and the
  Python↔TS parity fixtures stay in sync automatically (same numbers on both sides).
- `tests/` — new unit tests for `calibrate`: α=1 is identity; α=0 reproduces FIFA ordering on
  the reference set; monotonicity in α between a high-FIFA/low-model team and the reverse;
  junk (non-FIFA) teams untouched; originals not mutated.
- Showcase copy — `README.md`, `HowItWorks`, `Limitations`, `Validation`: reframe to
  "data-driven rating, calibrated against FIFA," document α and the data-staleness caveat, and
  add the before/after table as a selling point.

## 8. Validation / what we report for sign-off

- The α-vs-backtest-loss curve and the chosen α*.
- Before/after top-12 champion-odds table.
- Confirmation that the backtest model still beats the base-rate baseline at α*.
- Full Python test suite green; TS parity fixtures regenerated and passing.

## 9. Risks & caveats

- **FIFA inertia** — e.g. Belgium rides a generous #3; at moderate α it is tempered, but it is
  a disclosed consequence of trusting FIFA.
- **Stale snapshot** — FIFA data ends 2024-06-20.
- **Even attack/defense split** is a modeling choice, not estimated.
- **Two-stage adjustment**, not a joint estimate (accepted trade-off for transparency and for
  keeping the core fit + parity tests intact).

## 10. Non-goals (YAGNI)

- No change to the MLE objective or the core Dixon-Coles fit.
- No confederation map / match-importance reweighting (Approach 3).
- No new external data source (World Football Elo, bookmaker odds); FIFA points only.
- No dual "results-only vs calibrated" toggle in the showcase — single calibrated narrative.
