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
