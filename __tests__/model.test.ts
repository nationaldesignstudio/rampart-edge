import { describe, expect, it } from "vitest";
import * as ort from "onnxruntime-web";

import reference from "./reference.fixture.json";
import { loadLocalGate } from "./node-gate";
import { createGateModel, GateModelError, verifyPinned } from "../src/model";

const gate = await loadLocalGate();

describe("gate model (onnxruntime-web, wasm)", () => {
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

  it("refuses bytes that do not match the pin", async () => {
    const tampered = new Uint8Array([1, 2, 3]);
    await expect(verifyPinned("x", tampered, { bytes: 3, sha256: "0".repeat(64) })).rejects.toMatchObject({
      name: "GateModelError",
      stage: "integrity",
    });
  });

  it("refuses a graph with the wrong input and output names", async () => {
    const fake = {
      InferenceSession: {
        create: async () => ({ inputNames: ["x"], outputNames: ["y"], run: async () => ({}) }),
      },
      Tensor: ort.Tensor as never,
    };
    const vocab = gate.model!.tokenizer;
    const error = await createGateModel(fake, new Uint8Array(), "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n").catch((e) => e);
    expect(error).toBeInstanceOf(GateModelError);
    expect(error.stage).toBe("graph");
    expect(vocab.vocabSize).toBe(23_910);
  });
});
