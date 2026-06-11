import numpy as np
from src.simulate import simulate_group, rank_group


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
