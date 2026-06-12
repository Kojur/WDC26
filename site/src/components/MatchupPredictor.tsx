import { useMemo, useState } from "react";
import ratings from "../data/ratings.json";
import type { Ratings } from "../data/types";
import { scoreMatrix, predictResult, mostLikelyScore } from "../lib/dixonColes";
import { ScorelineGrid } from "./ScorelineGrid";

const R = ratings as Ratings;
const TEAMS = [...R.teams].sort();

export function MatchupPredictor() {
  const [home, setHome] = useState("Brazil");
  const [away, setAway] = useState("Croatia");

  const { matrix, result, top } = useMemo(() => {
    const m = scoreMatrix(home, away, R, true);
    return { matrix: m, result: predictResult(m), top: mostLikelyScore(m) };
  }, [home, away]);

  const pct = (x: number) => Math.round(x * 100);

  return (
    <div>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 20 }}>
        <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
          Home team
          <select aria-label="Home team" value={home} onChange={(e) => setHome(e.target.value)}>
            {TEAMS.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
          Away team
          <select aria-label="Away team" value={away} onChange={(e) => setAway(e.target.value)}>
            {TEAMS.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
      </div>

      <div data-testid="wdl-odds" style={{ display: "flex", gap: 20, marginBottom: 8, fontSize: 16 }}>
        <span><strong>{pct(result.homeWin)}%</strong> {home}</span>
        <span><strong>{pct(result.draw)}%</strong> draw</span>
        <span><strong>{pct(result.awayWin)}%</strong> {away}</span>
      </div>
      <p style={{ fontSize: 14, color: "#666", marginTop: 0 }}>
        Most likely score: {home} {top.home}–{top.away} {away} ({(top.prob * 100).toFixed(1)}%)
      </p>

      <ScorelineGrid matrix={matrix} homeName={home} awayName={away} display={7} />
    </div>
  );
}
