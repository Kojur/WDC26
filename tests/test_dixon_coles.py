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


def test_fit_recovers_strength_ordering(fitted_model):
    m = fitted_model
    i_strong = m.team_index["Strong"]
    i_weak = m.team_index["Weak"]
    assert m.attack[i_strong] > m.attack[i_weak]
    assert abs(m.attack.mean()) < 1e-6  # identifiability: mean attack == 0
    assert isinstance(m.home_adv, float)


def test_fit_sets_all_teams(fitted_model):
    assert set(fitted_model.teams) == {"Strong", "Medium", "Weak"}
