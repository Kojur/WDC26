export function Limitations() {
  return (
    <section className="section">
      <h2>The catch</h2>
      <p>
        The model rates South American sides unusually highly — five of its top six. That isn't a
        bug so much as a property of learning purely from results: CONMEBOL teams play each other
        constantly in a brutal round-robin, while Europe's giants rack up lopsided qualifying wins
        that the model can't fully weigh. It has no notion of confederation strength.
      </p>
      <p>
        It also can't see what isn't in the scoreline data: injuries, squad turnover, form, or
        tactics. Treat the numbers as an informed baseline, not a crystal ball.
      </p>
    </section>
  );
}
