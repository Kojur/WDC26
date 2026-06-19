"""Append the current (pre-2026-World-Cup) FIFA ranking snapshot to the raw CSV.

The Kaggle FIFA-ranking dataset (`data/raw/fifa_ranking.csv`, gitignored) ends at
2024-06-20 — *before* Spain won Euro 2024 — so it badly under-rates Spain and a few
others for a 2026 prediction. This script appends the official **11 June 2026**
pre-tournament release (the last update before the World Cup; the next was 20 July
2026, mid-tournament, so this snapshot is leak-free for a "going-in" forecast).

Only the top 20 are published openly, which covers every realistic title contender;
teams outside the top 20 are carried forward from the prior snapshot (their points
barely move and they have negligible championship odds). Team names use the model's
canonical spelling (e.g. "United States", "Iran") so the FIFA calibration matches.

Idempotent: running twice does not double-append. After running, re-run
`scripts/export_site_data.py` to regenerate the site data with the fresh snapshot.

Source: FIFA/Coca-Cola Men's World Ranking, 11 June 2026 release.
"""
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "fifa_ranking.csv"
SNAPSHOT_DATE = pd.Timestamp("2026-06-11")

# rank, team (model-canonical name), total_points -- 11 June 2026 release, top 20.
CURRENT = [
    (1, "Argentina", 1877.27), (2, "Spain", 1874.71), (3, "France", 1870.70),
    (4, "England", 1828.02), (5, "Portugal", 1767.85), (6, "Brazil", 1765.86),
    (7, "Morocco", 1755.10), (8, "Netherlands", 1753.57), (9, "Belgium", 1742.24),
    (10, "Germany", 1735.77), (11, "Croatia", 1714.87), (12, "Italy", 1704.73),
    (13, "Colombia", 1698.35), (14, "Mexico", 1687.48), (15, "Senegal", 1684.07),
    (16, "Uruguay", 1673.07), (17, "United States", 1671.23), (18, "Japan", 1661.58),
    (19, "Switzerland", 1650.06), (20, "Iran", 1619.58),
]
# Teams whose FIFA-CSV spelling differs from the model's canonical name -> add fresh rows.
_ABBR_CONF = {"United States": ("USA", "CONCACAF"), "Iran": ("IRN", "AFC")}


def main():
    df = pd.read_csv(CSV, parse_dates=["rank_date"])
    if (df["rank_date"] == SNAPSHOT_DATE).any():
        print(f"snapshot {SNAPSHOT_DATE.date()} already present; nothing to do")
        return

    latest = df[df["rank_date"] == df["rank_date"].max()].copy()
    snap = latest.copy()
    snap["rank_date"] = SNAPSHOT_DATE
    present = set(snap["country_full"])
    extra = []
    for rank, team, pts in CURRENT:
        if team in present:
            m = snap["country_full"] == team
            snap.loc[m, ["rank", "total_points", "previous_points", "rank_change"]] = \
                [rank, pts, pts, 0]
        else:
            abbr, conf = _ABBR_CONF[team]
            extra.append({"rank": rank, "country_full": team, "country_abrv": abbr,
                          "total_points": pts, "previous_points": pts, "rank_change": 0,
                          "confederation": conf, "rank_date": SNAPSHOT_DATE})
    snap = pd.concat([snap, pd.DataFrame(extra)], ignore_index=True)
    pd.concat([df, snap], ignore_index=True).to_csv(CSV, index=False)
    print(f"appended {len(snap)} rows dated {SNAPSHOT_DATE.date()} "
          f"({len(CURRENT)} refreshed, {len(extra)} added)")


if __name__ == "__main__":
    main()
