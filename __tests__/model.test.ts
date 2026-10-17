import { describe, expect, it } from "vitest";

import reference from "./reference.fixture.json";
import { loadLocalFiles, loadLocalGate } from "./node-gate";
import { EngineError, type GateEngine } from "../src/engine";
import { createGateModel, createGateModelFromWeights, fromEngineError, GateModelError, verifyPinned } from "../src/model";

const gate = await loadLocalGate();
const engine = gate.model!.engine;

describe("gate model (RMPW weights on the inlined wasm engine)", () => {
  expect(Object.keys(reference.samples).length).toBeGreaterThan(5);
  for (const [id, sample] of Object.entries(reference.samples)) {
    it(`reproduces the Python reference probability for ${id}`, async () => {
      const decision = await gate.check(sample.text);
      expect(decision.model?.windows).toHaveLength(sample.windows.length);
      expect(decision.model?.probability).toBeCloseTo(sample.maxProb, 3);
    });
  }

  it("blocks obvious PII and allows a benign government question", async () => {
    expect((await gate.check(reference.samples.ssn.text)).verdict).toBe("BLOCK");
    expect((await gate.check(reference.samples.fee.text)).verdict).toBe("ALLOW");
    expect((await gate.check(reference.samples["public-figure"].text)).verdict).toBe("ALLOW");
  });

  it("exposes the engine it runs on", () => {
    expect(engine.loadMs).toBeGreaterThan(0);
    expect(engine.memoryPages).toBeGreaterThan(100);
    expect(gate.model!.tokenizer.vocabSize).toBe(23_910);
  });

  it("refuses bytes that do not match the pin", async () => {
    const tampered = new Uint8Array([1, 2, 3]);
    await expect(verifyPinned("x", tampered, { bytes: 3, sha256: "0".repeat(64) })).rejects.toMatchObject({
      name: "GateModelError",
      stage: "integrity",
    });
  });

  it("refuses weights whose payload was altered, as an integrity failure", async () => {
    const { weights, vocab } = await loadLocalFiles();
    const flipped = weights.slice();
    flipped[flipped.length - 1] ^= 0xff;
    const error = await createGateModelFromWeights({ weights: flipped, vocab }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(GateModelError);
    expect((error as GateModelError).stage).toBe("integrity");
    expect((error as GateModelError).cause).toBeInstanceOf(EngineError);
    expect(((error as GateModelError).cause as EngineError).stage).toBe("integrity");
  });

  it("refuses weights that are not RMPW v1, as a graph failure", async () => {
    const { weights, vocab } = await loadLocalFiles();
    const other = weights.slice();
    other[0] = 0x58;
    await expect(createGateModelFromWeights({ weights: other, vocab })).rejects.toMatchObject({ name: "GateModelError", stage: "graph" });
  });

  it("refuses an engine whose span head does not have the configured label count", () => {
    const narrow: GateEngine = {
      run: (ids) => ({ gateLogit: 0, tokenLogits: new Float32Array(ids.length * 5) }),
      memoryPages: 1,
      loadMs: 1,
      kernel: "unrolled",
    };
    const error = (() => {
      try {
        return createGateModel(narrow, "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n");
      } catch (e) {
        return e;
      }
    })();
    expect(error).toBeInstanceOf(GateModelError);
    expect((error as GateModelError).stage).toBe("graph");
  });

  it("maps every engine failure stage onto the documented GateModelError stage", () => {
    const expected = { wasm: "runtime", format: "graph", integrity: "integrity", init: "session", range: "inference", numeric: "inference" } as const;
    for (const [from, to] of Object.entries(expected)) {
      const mapped = fromEngineError(new EngineError(from as EngineError["stage"], from));
      expect(mapped.stage).toBe(to);
      expect(mapped.message).toBe(from);
    }
    expect(fromEngineError(new TypeError("boom"), "session")).toMatchObject({ stage: "session", message: "boom" });
    const passthrough = new GateModelError("download", "x");
    expect(fromEngineError(passthrough)).toBe(passthrough);
  });

  it("turns a window the engine rejects into GateModelError('inference')", async () => {
    const tooLong = "word ".repeat(500);
    const ids = gate.model!.tokenizer.encode(tooLong);
    expect(ids.length).toBeGreaterThan(384);
    expect(() => engine.run(ids)).toThrow(EngineError);
    expect((await gate.check(tooLong)).model?.windows.length).toBeGreaterThan(1);
  });
});
