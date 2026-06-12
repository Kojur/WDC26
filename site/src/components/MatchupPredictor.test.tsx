import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MatchupPredictor } from "./MatchupPredictor";

describe("MatchupPredictor", () => {
  it("shows win/draw/loss odds that sum to ~100%", () => {
    render(<MatchupPredictor />);
    const odds = screen.getByTestId("wdl-odds").textContent || "";
    const nums = (odds.match(/\d+/g) || []).map(Number);
    const total = nums.reduce((a, b) => a + b, 0);
    expect(total).toBeGreaterThanOrEqual(98);
    expect(total).toBeLessThanOrEqual(102);
  });

  it("updates when the away team changes", () => {
    render(<MatchupPredictor />);
    const before = screen.getByTestId("wdl-odds").textContent;
    const away = screen.getByLabelText("Team 2") as HTMLSelectElement;
    const other = Array.from(away.options).find((o) => o.value !== away.value)!;
    fireEvent.change(away, { target: { value: other.value } });
    expect(screen.getByTestId("wdl-odds").textContent).not.toEqual(before);
  });
});
