import { Hero } from "./components/Hero";
import { TitleOddsChart } from "./components/TitleOddsChart";
import { HowItWorks } from "./components/HowItWorks";
import { MatchupPredictor } from "./components/MatchupPredictor";
import { Validation } from "./components/Validation";
import { Limitations } from "./components/Limitations";
import { About } from "./components/About";
import "./index.css";

export default function App() {
  return (
    <main className="page">
      <Hero />
      <section className="section">
        <h2>The title race</h2>
        <p>Each team's chance of winning the tournament, across 10,000 simulations:</p>
        <TitleOddsChart topN={12} />
      </section>
      <HowItWorks />
      <section className="section">
        <h2>Try it yourself</h2>
        <p>Pick any two teams. The model runs right here in your browser.</p>
        <MatchupPredictor />
      </section>
      <Validation />
      <Limitations />
      <About />
    </main>
  );
}
