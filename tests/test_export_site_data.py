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
