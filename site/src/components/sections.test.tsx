import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Hero } from "./Hero";
import { Validation } from "./Validation";
import { Limitations } from "./Limitations";
import { About } from "./About";

describe("content sections", () => {
  it("Hero shows the headline question", () => {
    render(<Hero />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/2026 World Cup/i);
  });
  it("Validation shows the model and baseline RPS", () => {
    render(<Validation />);
    expect(screen.getByText(/RPS/i)).toBeInTheDocument();
  });
  it("Limitations mentions South America", () => {
    render(<Limitations />);
    expect(screen.getByText(/South American/i)).toBeInTheDocument();
  });
  it("About links to the GitHub repo", () => {
    render(<About />);
    const link = screen.getByRole("link", { name: /github/i });
    expect(link).toHaveAttribute("href", expect.stringContaining("github.com/Kojur/WDC26"));
  });
});
