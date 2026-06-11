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
