import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ScorelineGrid } from "./ScorelineGrid";

describe("ScorelineGrid", () => {
  it("renders a cell for each scoreline up to display size", () => {
    const m = Array.from({ length: 3 }, () => [0.2, 0.1, 0.0]);
    render(<ScorelineGrid matrix={m} homeName="A" awayName="B" display={3} />);
    expect(screen.getByTestId("scoreline-grid")).toBeInTheDocument();
    // 3x3 = 9 probability cells
    expect(screen.getAllByTestId("grid-cell")).toHaveLength(9);
  });
});
