import { vi } from "vitest";
vi.mock("recharts", async (importOriginal) => {
  const actual = await importOriginal<typeof import("recharts")>();
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <actual.ResponsiveContainer width={800} height={400}>{children}</actual.ResponsiveContainer>
    ),
  };
});

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { TitleOddsChart } from "./TitleOddsChart";

describe("TitleOddsChart", () => {
  it("renders the top team's name", () => {
    render(<TitleOddsChart topN={5} />);
    expect(screen.getAllByText("Brazil").length).toBeGreaterThan(0);
  });
});
