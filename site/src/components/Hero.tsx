import simulation from "../data/simulation.json";
import type { Simulation } from "../data/types";

const top = (simulation as Simulation).teams[0];

export function Hero() {
  return (
    <header className="hero">
      <h1>Who will win the 2026 World Cup?</h1>
      <p className="lede">
        A Dixon-Coles statistical model, trained on 150 years of international football and
        simulated 10,000 times — explained simply, and yours to play with.
      </p>
      <p className="headline-stat">
        <strong>{Math.round(top.p_champion * 100)}%</strong> {top.team} to lift the cup
      </p>
    </header>
  );
}
