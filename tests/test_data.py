import pandas as pd
from src.data import clean_results, DEFAULT_ALIASES


def test_clean_results_normalizes_aliases_and_types():
    raw = pd.DataFrame({
        "date": ["1990-06-08", "2002-06-01"],
        "home_team": ["West Germany", "USA"],
        "away_team": ["USA", "Korea Republic"],
        "home_score": [4.0, 1.0],
        "away_score": [1.0, 1.0],
        "tournament": ["FIFA World Cup", "FIFA World Cup"],
        "neutral": [True, True],
        "extra_col": ["x", "y"],
    })
    out = clean_results(raw)
    assert list(out.columns) == [
        "date", "home_team", "away_team", "home_score",
        "away_score", "tournament", "neutral",
    ]
    assert out.loc[0, "home_team"] == "Germany"
    assert out.loc[0, "away_team"] == "United States"
    assert out.loc[1, "away_team"] == "South Korea"
    assert pd.api.types.is_datetime64_any_dtype(out["date"])
    assert out["home_score"].dtype == int
