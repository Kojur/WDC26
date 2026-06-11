"""Load and clean the international results dataset and FIFA rankings."""
import pandas as pd

DEFAULT_ALIASES = {
    "West Germany": "Germany",
    "East Germany": "Germany",
    "USA": "United States",
    "Soviet Union": "Russia",
    "Czechoslovakia": "Czechia",
    "Yugoslavia": "Serbia",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
}

_COLS = ["date", "home_team", "away_team", "home_score",
         "away_score", "tournament", "neutral"]


def load_results(path):
    """Read results.csv with parsed dates."""
    return pd.read_csv(path, parse_dates=["date"])


def clean_results(df, aliases=None):
    """Normalize team names, coerce types, drop incomplete rows, select columns."""
    aliases = DEFAULT_ALIASES if aliases is None else aliases
    out = df.copy()
    out["home_team"] = out["home_team"].replace(aliases)
    out["away_team"] = out["away_team"].replace(aliases)
    if not pd.api.types.is_datetime64_any_dtype(out["date"]):
        out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=["home_score", "away_score"])
    out["home_score"] = out["home_score"].astype(int)
    out["away_score"] = out["away_score"].astype(int)
    out["neutral"] = out["neutral"].astype(bool)
    return out[_COLS].reset_index(drop=True)


def load_rankings(path):
    """Read the FIFA ranking CSV (Kaggle 'cashncarry/fifaworldranking')."""
    return pd.read_csv(path, parse_dates=["rank_date"])


def latest_rankings(rankings, as_of=None):
    """Return {country: rank} from the most recent snapshot on/before as_of."""
    df = rankings if as_of is None else rankings[rankings["rank_date"] <= as_of]
    snap = df[df["rank_date"] == df["rank_date"].max()]
    return dict(zip(snap["country_full"], snap["rank"]))
