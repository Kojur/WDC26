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
          Team 1
          <select aria-label="Team 1" value={home} onChange={(e) => setHome(e.target.value)}>
            {TEAMS.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
          Team 2
          <select aria-label="Team 2" value={away} onChange={(e) => setAway(e.target.value)}>
            {TEAMS.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
      </div>

      <p style={{ fontSize: 12, color: "#888", margin: "0 0 16px" }}>
        Modelled at a neutral venue (no home advantage), as at a World Cup.
      </p>

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
