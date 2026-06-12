import { BarChart, Bar, XAxis, YAxis, Cell, ResponsiveContainer, LabelList } from "recharts";
import simulation from "../data/simulation.json";
import type { Simulation } from "../data/types";

const SIM = simulation as Simulation;

export function TitleOddsChart({ topN = 12 }: { topN?: number }) {
  const data = SIM.teams.slice(0, topN).map((t) => ({
    team: t.team, pct: Math.round(t.p_champion * 1000) / 10,
  }));
  return (
    <ResponsiveContainer width="100%" height={topN * 34 + 40}>
      <BarChart data={data} layout="vertical" margin={{ left: 12, right: 36 }}>
        <XAxis type="number" hide domain={[0, "dataMax"]} />
        <YAxis type="category" dataKey="team" width={92} tickLine={false} axisLine={false}
               tick={{ fontSize: 13 }} />
        <Bar dataKey="pct" radius={[2, 2, 2, 2]}>
          {data.map((_, i) => <Cell key={i} fill="#1a7f5a" />)}
          <LabelList dataKey="pct" position="right" formatter={(v: number) => `${v}%`}
                     style={{ fontSize: 12, fill: "#444" }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
