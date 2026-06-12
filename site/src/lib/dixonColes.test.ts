import { describe, it, expect } from "vitest";
import ratings from "../data/ratings.json";
import parity from "../data/parity_fixtures.json";
import type { Ratings, ParityCase } from "../data/types";
import { scoreMatrix, predictResult, expectedGoals } from "./dixonColes";

const R = ratings as Ratings;

describe("dixonColes parity with Python", () => {
  it("reproduces predict_result for every fixture", () => {
    (parity as ParityCase[]).forEach((c) => {
      const res = predictResult(scoreMatrix(c.home, c.away, R, c.neutral));
      expect(res.homeWin).toBeCloseTo(c.home_win, 6);
      expect(res.draw).toBeCloseTo(c.draw, 6);
      expect(res.awayWin).toBeCloseTo(c.away_win, 6);
    });
  });

  it("score matrix sums to 1", () => {
    const m = scoreMatrix(R.teams[0], R.teams[1], R);
    const sum = m.flat().reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 9);
  });

  it("stronger expected goals for the better team", () => {
    const [lam, mu] = expectedGoals("Brazil", "Croatia", R);
    expect(lam).toBeGreaterThan(mu);
  });
});
