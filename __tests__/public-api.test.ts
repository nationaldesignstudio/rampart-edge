import { describe, expect, it } from "vitest";

import { BLOCK_THRESHOLD, createValidatorGate, Gate, MODEL_FILES, type GateDecision } from "../index";
import { loadLocalGate } from "./node-gate";

const gate: Gate = await loadLocalGate();

const cases: ReadonlyArray<{ text: string; verdict: GateDecision["verdict"]; basis: GateDecision["basis"] }> = [
  { text: "My SSN is 452-78-1934 and I need to fix my tax return.", verdict: "BLOCK", basis: "validator-and-model" },
  { text: "My landlord John Smith won't fix the heat, who do I report him to?", verdict: "BLOCK", basis: "model" },
  { text: "How much is a passport renewal and how long does it take?", verdict: "ALLOW", basis: "clear" },
  { text: "Who is Jerome Powell?", verdict: "ALLOW", basis: "clear" },
];

describe("public API: createGate-shaped gate over the committed model", () => {
  it("exports the pinned model files and the default threshold", () => {
    expect(Object.keys(MODEL_FILES).sort()).toEqual(["model.rmpw", "vocab.txt"]);
    expect(gate.threshold).toBe(BLOCK_THRESHOLD);
  });

  it.each(cases)("$verdict/$basis: $text", async ({ text, verdict, basis }) => {
    const decision = await gate.check(text);
    expect(decision.verdict).toBe(verdict);
    expect(decision.basis).toBe(basis);
  });

  it("falls back to validators only when the model is unavailable", async () => {
    const validatorsOnly = createValidatorGate();
    const ssn = await validatorsOnly.check(cases[0].text);
    expect(ssn).toMatchObject({ verdict: "BLOCK", basis: "validator-only-model-unavailable", model: null });
    const name = await validatorsOnly.check(cases[1].text);
    expect(name).toMatchObject({ verdict: "ALLOW", basis: "clear-model-unavailable" });
  });
});
