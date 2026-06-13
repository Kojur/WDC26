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
