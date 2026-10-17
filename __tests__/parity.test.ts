/**
 * Parity gate: the shipped engine against the archived output of onnxruntime-web 1.30 (WASM EP, one thread) on
 * `model_q4.onnx` in Chromium, the canonical result for this model. The references hold token ids and logits only, no
 * text (schema `rampart-edge-parity/1`, archived by rampart-2 `tools/model/edge-parity`).
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { loadLocalModel } from "./node-gate";
import { sigmoid } from "../src/scoring";

type Window = { ids: number[]; gate: number; argmax: number[]; tok?: number[] };
type ParityFile = { schema: string; set: string; threshold: number; counts: { rows: number; windows: number; tokens: number }; rows: { id: string; windows: Window[] }[] };

const MAX_GATE_DELTA = 1e-4;
const sets = ["fixture", "dev-1000", "near-threshold"] as const;
const model = await loadLocalModel();
const engine = model.engine;
const labels = model.spanHead!.labels.length;

function argmaxRows(flat: Float32Array, n: number): number[] {
  const out = new Array<number>(n);
  for (let t = 0; t < n; t++) {
    let best = 0;
    for (let j = 1; j < labels; j++) if (flat[t * labels + j] > flat[t * labels + best]) best = j;
    out[t] = best;
  }
  return out;
}

describe("parity with the archived browser onnxruntime-web output", () => {
  for (const set of sets) {
    it(`${set}: max |Δ gate| ≤ ${MAX_GATE_DELTA}, every token argmax equal, no decision flips`, { timeout: 120_000 }, () => {
      const ref = JSON.parse(readFileSync(new URL(`./fixtures/parity/${set}.json`, import.meta.url), "utf8")) as ParityFile;
      expect(ref.schema).toBe("rampart-edge-parity/1");
      expect(ref.rows).toHaveLength(ref.counts.rows);
      let windows = 0;
      let tokens = 0;
      let maxDelta = 0;
      let argmaxMismatches = 0;
      let flips = 0;
      let maxTokDelta = 0;
      for (const row of ref.rows) {
        let refMax = -Infinity;
        let ourMax = -Infinity;
        for (const w of row.windows) {
          const out = engine.run(w.ids);
          windows++;
          tokens += w.ids.length;
          maxDelta = Math.max(maxDelta, Math.abs(out.gateLogit - w.gate));
          refMax = Math.max(refMax, sigmoid(w.gate));
          ourMax = Math.max(ourMax, sigmoid(out.gateLogit));
          const ours = argmaxRows(out.tokenLogits, w.ids.length);
          for (let t = 0; t < w.ids.length; t++) if (ours[t] !== w.argmax[t]) argmaxMismatches++;
          if (w.tok) for (let i = 0; i < w.tok.length; i++) maxTokDelta = Math.max(maxTokDelta, Math.abs(out.tokenLogits[i] - w.tok[i]));
        }
        if (refMax >= ref.threshold !== ourMax >= ref.threshold) flips++;
      }
      expect(windows).toBe(ref.counts.windows);
      expect(tokens).toBe(ref.counts.tokens);
      expect(maxDelta).toBeLessThanOrEqual(MAX_GATE_DELTA);
      expect(argmaxMismatches).toBe(0);
      expect(flips).toBe(0);
      if (set === "fixture") expect(maxTokDelta).toBeLessThanOrEqual(1e-3);
    });
  }
});
