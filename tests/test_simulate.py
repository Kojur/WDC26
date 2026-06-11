import numpy as np
from src.simulate import simulate_group, rank_group
from src.simulate import select_best_thirds, simulate_knockout_match
from src.simulate import simulate_tournament, monte_carlo


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
