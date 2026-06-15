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


def test_latest_points_picks_most_recent_on_or_before():
    import pandas as pd
    from src.data import latest_points
    rankings = pd.DataFrame({
        "rank_date": pd.to_datetime(["2014-01-01", "2014-01-01", "2018-06-01"]),
        "country_full": ["France", "Brazil", "France"],
        "total_points": [1500.0, 1600.0, 1837.0],
    })
    # as_of before the 2018 snapshot -> uses the 2014 snapshot
    early = latest_points(rankings, as_of=pd.Timestamp("2015-01-01"))
    assert early == {"France": 1500.0, "Brazil": 1600.0}
    # no as_of -> latest snapshot only (2018, France only)
    assert latest_points(rankings) == {"France": 1837.0}
