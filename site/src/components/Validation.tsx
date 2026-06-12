import meta from "../data/meta.json";
import type { Meta } from "../data/types";

const M = meta as Meta;

export function Validation() {
  const b = M.backtest;
  return (
    <section className="section">
      <h2>Does it actually work?</h2>
      <p>
        Tested honestly — trained only on matches <em>before</em> each World Cup, then asked to
        predict it ({b.n_matches} matches across {b.wc_years.join(" & ")}). It beats a sensible
        baseline on every measure:
      </p>
      <table className="metrics">
        <thead><tr><th></th><th>Model</th><th>Baseline</th></tr></thead>
        <tbody>
          <tr><td>RPS (lower better)</td><td>{b.model.rps}</td><td>{b.baseline.rps}</td></tr>
          <tr><td>Log-loss (lower better)</td><td>{b.model.log_loss}</td><td>{b.baseline.log_loss}</td></tr>
          <tr><td>Accuracy</td><td>{Math.round(b.model.accuracy * 100)}%</td><td>{Math.round(b.baseline.accuracy * 100)}%</td></tr>
        </tbody>
      </table>
      <p>
        As a sanity check, the model's learned strength correlates {M.ratings_sanity.fifa_rank_corr}
        with the official FIFA ranking — without ever being shown it.
      </p>
    </section>
  );
}
