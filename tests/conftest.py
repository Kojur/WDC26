import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_matches():
    """300 friendly matches among three teams of known relative strength."""
    rng = np.random.default_rng(0)
    teams = ["Strong", "Medium", "Weak"]
    base = {"Strong": 1.9, "Medium": 1.1, "Weak": 0.6}
    dates = pd.date_range("2010-01-01", periods=300, freq="W")
    rows = []
    for d in dates:
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append({
            "date": d, "home_team": h, "away_team": a,
            "home_score": int(rng.poisson(base[h])),
            "away_score": int(rng.poisson(base[a])),
            "tournament": "Friendly", "neutral": True,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_model(synthetic_matches):
    from src.dixon_coles import DixonColesModel
    return DixonColesModel().fit(synthetic_matches, xi=0.0)


class StubModel:
    """Deterministic-strength model for simulation tests (no fitting needed)."""
    def __init__(self, strengths):
        self.strengths = strengths

    def _params(self, team):
        return self.strengths.get(team, 0.0), 0.0

    def sample_scoreline(self, home, away, rng, neutral=True):
        hg = rng.poisson(max(0.1, 1.0 + self.strengths.get(home, 0.0)))
        ag = rng.poisson(max(0.1, 1.0 + self.strengths.get(away, 0.0)))
        return int(hg), int(ag)


@pytest.fixture
def stub_model_factory():
    return StubModel
