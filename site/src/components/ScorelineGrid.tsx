interface Props {
  matrix: number[][];
  homeName: string;
  awayName: string;
  display?: number;
}

const REGION = {
  home: { bg: "#e6f1eb", fg: "#1a7f5a" },
  draw: { bg: "#f4efe2", fg: "#8a6d1a" },
  away: { bg: "#f6eae7", fg: "#a8553a" },
};

export function ScorelineGrid({ matrix, homeName, awayName, display = 6 }: Props) {
  const n = Math.min(display, matrix.length);
  const cols = `36px repeat(${n}, 1fr)`;
  const cells = [];
  // header row
  cells.push(<div key="corner" />);
  for (let y = 0; y < n; y++)
    cells.push(<div key={`h${y}`} style={{ textAlign: "center", fontSize: 12, color: "#888" }}>{y}</div>);
  for (let x = 0; x < n; x++) {
    cells.push(<div key={`r${x}`} style={{ display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, color: "#888" }}>{x}</div>);
    for (let y = 0; y < n; y++) {
      const region = x > y ? REGION.home : x === y ? REGION.draw : REGION.away;
      const pct = matrix[x][y] * 100;
      cells.push(
        <div key={`${x}-${y}`} data-testid="grid-cell"
          style={{ background: region.bg, color: region.fg, borderRadius: 4, height: 34,
                   display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12 }}>
          {pct >= 0.5 ? Math.round(pct) : ""}
        </div>,
      );
    }
  }
  return (
    <div>
      <div style={{ fontSize: 13, color: "#666", marginBottom: 6 }}>
        {awayName} goals across (→), {homeName} goals down (↓)
      </div>
      <div data-testid="scoreline-grid"
        style={{ display: "grid", gridTemplateColumns: cols, gap: 3, maxWidth: 480 }}>
        {cells}
      </div>
    </div>
  );
}
