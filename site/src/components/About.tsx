import meta from "../data/meta.json";
import type { Meta } from "../data/types";

const M = meta as Meta;

export function About() {
  return (
    <section className="section about">
      <h2>Behind the scenes</h2>
      <p>
        Built from {M.dataset.n_matches.toLocaleString()} international matches
        ({M.dataset.date_min} to {M.dataset.date_max}). The model (Dixon-Coles + Monte Carlo) is
        written in Python; this page is React, with the model re-implemented in TypeScript so the
        predictor runs in your browser.
      </p>
      <p>
        <a href="https://github.com/Kojur/WDC26">Source &amp; notebooks on GitHub</a>
      </p>
    </section>
  );
}
