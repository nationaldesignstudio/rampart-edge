import { describe, expect, it } from "vitest";

import reference from "./reference.fixture.json";
import { createValidatorGate, Gate } from "../src/gate";
import { labelTokens } from "../src/spans";
import {
  assertThreshold,
  BLOCK_THRESHOLD,
  CONTENT_TOKENS,
  STRIDE,
  combineWindows,
  contentWindows,
  decide,
  modelBlocks,
  sigmoid,
  withSpecialTokens,
} from "../src/scoring";
import { gate } from "../src/validators/index";

const ids = (n: number) => Array.from({ length: n }, (_, i) => i + 100);

describe("contentWindows", () => {
  it("keeps a text that fits in one window", () => {
    expect(contentWindows(ids(382))).toEqual([ids(382)]);
    expect(contentWindows([])).toEqual([[]]);
  });

  it("starts the second window where Hugging Face stride=96 would", () => {
    const all = ids(383);
    const windows = contentWindows(all);
    expect(windows).toHaveLength(2);
    expect(windows[0]).toEqual(all.slice(0, 382));
    expect(windows[1]).toEqual(all.slice(286));
    expect(windows[1]).toHaveLength(97);
    expect(CONTENT_TOKENS - STRIDE).toBe(286);
  });

  it("covers every token of a long input with overlapping windows", () => {
    const all = ids(1000);
    const windows = contentWindows(all);
    expect(windows.map((w) => w.length)).toEqual([382, 382, 382, 142]);
    expect(windows.map((w) => all.indexOf(w[0]))).toEqual([0, 286, 572, 858]);
    expect(new Set(windows.flat()).size).toBe(1000);
  });

  it("matches the Python reference window sizes for the long sample", () => {
    const long = reference.samples.long;
    expect(long.contentTokens).toBe(427);
    expect(contentWindows(ids(long.contentTokens)).map((w) => w.length)).toEqual(
      long.windows.map((w) => w.tokens),
    );
  });

  it("wraps a window in [CLS] and [SEP]", () => {
    expect(withSpecialTokens([5, 6], 2, 3)).toEqual([2, 5, 6, 3]);
  });
});

describe("combineWindows", () => {
  it("takes the maximum BLOCK probability across windows", () => {
    expect(combineWindows([0.01, 0.98, 0.2]).probability).toBe(0.98);
    expect(combineWindows([0.3]).probability).toBe(0.3);
  });

  it("matches the Python reference for the two-window sample", () => {
    const probabilities = reference.samples.long.windows.map((w) => sigmoid(w.logit));
    expect(probabilities[0]).toBeCloseTo(reference.samples.long.windows[0].prob, 6);
    expect(combineWindows(probabilities).probability).toBeCloseTo(
      reference.samples.long.maxProb,
      6,
    );
  });

  it("refuses empty or out-of-range input", () => {
    expect(() => combineWindows([])).toThrow("a model score needs at least one window");
    expect(() => combineWindows([1.5])).toThrow("window probability out of range: 1.5");
    expect(() => combineWindows([Number.NaN])).toThrow("window probability out of range: NaN");
  });
});

describe("threshold", () => {
  it("blocks at exactly the production threshold and allows one ulp below", () => {
    const below = BLOCK_THRESHOLD - Number.EPSILON * BLOCK_THRESHOLD;
    expect(BLOCK_THRESHOLD).toBe(0.417801);
    expect(modelBlocks(combineWindows([BLOCK_THRESHOLD]))).toBe(true);
    expect(below).toBeLessThan(BLOCK_THRESHOLD);
    expect(modelBlocks(combineWindows([below]))).toBe(false);
  });
});

describe("decide (validators OR model)", () => {
  const clear = gate("When was Abraham Lincoln elected president?");
  const ssn = gate("My SSN is 452-78-1934");

  it("allows when neither fires", () => {
    const d = decide(clear, combineWindows([0.02]));
    expect(d.verdict).toBe("ALLOW");
    expect(d.basis).toBe("clear");
  });

  it("blocks on the model alone", () => {
    const d = decide(clear, combineWindows([0.9]));
    expect(d.verdict).toBe("BLOCK");
    expect(d.basis).toBe("model");
  });

  it("blocks on a validator alone, even when the model is confident it is fine", () => {
    const d = decide(ssn, combineWindows([0.001]));
    expect(d.verdict).toBe("BLOCK");
    expect(d.basis).toBe("validator");
  });

  it("reports when both agree", () => {
    expect(decide(ssn, combineWindows([0.99])).basis).toBe("validator-and-model");
  });

  it("falls back to validators only while the model is unavailable", () => {
    expect(decide(ssn, null)).toMatchObject({
      verdict: "BLOCK",
      basis: "validator-only-model-unavailable",
    });
    expect(decide(clear, null)).toMatchObject({
      verdict: "ALLOW",
      basis: "clear-model-unavailable",
    });
  });
});

describe("threshold validation", () => {
  const bad = [Number.NaN, -1, 0, 2, Number.POSITIVE_INFINITY, Number.parseFloat("")];
  const message = /threshold must be a finite number in \(0, 1\]/;
  const clear = { block: false, hard: [], soft: [] };

  it("accepts thresholds in (0, 1]", () => {
    expect(assertThreshold(BLOCK_THRESHOLD)).toBe(BLOCK_THRESHOLD);
    expect(assertThreshold(1)).toBe(1);
    expect(new Gate(null, 0.5).threshold).toBe(0.5);
  });

  for (const t of bad) {
    it(`rejects ${String(t)} everywhere a threshold is accepted`, () => {
      expect(() => assertThreshold(t)).toThrow(RangeError);
      expect(() => new Gate(null, t)).toThrow(message);
      expect(() => createValidatorGate(t)).toThrow(message);
      expect(() => decide(clear, null, t)).toThrow(message);
      expect(() => decide(clear, { windows: [0.9], probability: 0.9 }, t)).toThrow(message);
      expect(() => modelBlocks({ windows: [0.9], probability: 0.9 }, t)).toThrow(message);
      expect(() => labelTokens([[0.1, 0.9]], t)).toThrow(/span threshold must be a finite number/);
    });
  }
});
