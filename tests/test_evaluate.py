import numpy as np
import pandas as pd
from src.evaluate import result_outcome, rps, log_loss, evaluate
from src.evaluate import (base_rate_probs, rank_baseline_probs,
                          calibration_curve, walk_forward_worldcups)
from src.dixon_coles import DixonColesModel


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


def _fifa_fixture():
    return pd.DataFrame({
        "rank_date": pd.to_datetime(["2014-01-01"] * 3),
        "country_full": ["Strong", "Medium", "Weak"],
        "total_points": [1500.0, 1300.0, 1100.0],
    })


def test_walk_forward_calibrated_runs(synthetic_matches):
    m = synthetic_matches.copy()
    m.loc[m["date"].dt.year == 2015, "tournament"] = "FIFA World Cup"
    base, _ = walk_forward_worldcups(
        m, wc_years=[2015], model_factory=DixonColesModel, xi=0.0)
    preds, outs = walk_forward_worldcups(
        m, wc_years=[2015], model_factory=DixonColesModel, xi=0.0,
        fifa_rankings=_fifa_fixture(), alpha=0.5)
    assert len(preds) == len(outs) > 0
    assert all(abs(sum(p) - 1.0) < 1e-6 for p in preds)
    # calibration must actually move the predictions (FIFA fixture inverts the model order)
    assert not np.allclose(np.array(preds), np.array(base))


def test_walk_forward_alpha_one_matches_uncalibrated(synthetic_matches):
    m = synthetic_matches.copy()
    m.loc[m["date"].dt.year == 2015, "tournament"] = "FIFA World Cup"
    base, _ = walk_forward_worldcups(
        m, wc_years=[2015], model_factory=DixonColesModel, xi=0.0)
    same, _ = walk_forward_worldcups(
        m, wc_years=[2015], model_factory=DixonColesModel, xi=0.0,
        fifa_rankings=_fifa_fixture(), alpha=1.0)
    np.testing.assert_allclose(base, same, atol=1e-9)


def test_alpha_backtest_curve_shape(synthetic_matches):
    from src.evaluate import alpha_backtest_curve
    m = synthetic_matches.copy()
    m.loc[m["date"].dt.year == 2015, "tournament"] = "FIFA World Cup"
    curve = alpha_backtest_curve(
        m, wc_years=[2015], model_factory=DixonColesModel, xi=0.0,
        fifa_rankings=_fifa_fixture(), alphas=[0.5, 1.0])
    assert [c["alpha"] for c in curve] == [0.5, 1.0]
    assert all({"alpha", "log_loss", "rps"} <= set(c) for c in curve)
    assert all(c["log_loss"] > 0 and 0 <= c["rps"] <= 1 for c in curve)


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
