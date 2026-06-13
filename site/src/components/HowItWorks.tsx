export function HowItWorks() {
  return (
    <section className="section">
      <h2>How it works</h2>
      <p>
        Every national team gets two numbers learned from history: an <em>attack</em> rating
        (how many goals it tends to score) and a <em>defense</em> rating (how few it concedes).
        For any matchup, those combine into an expected number of goals for each side.
      </p>
      <p>
        We then treat goals as a Poisson process — turning the two expected-goal numbers into a
        full grid of scoreline probabilities (a 2-1, a 0-0, and so on). Summing the right cells
        of that grid gives the chance of a win, draw, or loss.
      </p>
      <p>
        One adjustment first: a model trained only on results over-rates regions that
        mostly play among themselves — South America especially. So we gently calibrate
        each team's strength toward the FIFA ranking, by a blend weight chosen on the
        backtest, before simulating. The data still does the heavy lifting; FIFA just
        anchors the regions to each other.
      </p>
      <p>
        To predict the whole tournament, we simulate all 104 matches — group stage through
        final — <strong>ten thousand times</strong>, and count how often each team lifts
        the trophy.
      </p>
    </section>
  );
}
