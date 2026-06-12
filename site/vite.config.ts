/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/WDC26/",
  plugins: [react()],
  test: { environment: "jsdom", globals: true, setupFiles: "./src/setupTests.ts", passWithNoTests: true },
});
