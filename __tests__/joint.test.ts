import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import reference from "./fixtures/joint-reference.json";
import { Gate } from "../src/gate";
import type { GateEngine } from "../src/engine";
import { createGateModel, GateModelError, sha256Hex, type ModelConfig } from "../src/model";
import { createRedactor } from "../src/redactor";
import { decodeSpans, deterministicUnion } from "../src/spans";
import { createWorkerGate, registerGateWorker, type WorkerPort } from "../src/worker";

const vocab = readFileSync(new URL("../model/vocab.txt", import.meta.url));
const vocabText = new TextDecoder().decode(vocab);
const config: ModelConfig = {
  inputs: ["input_ids", "attention_mask"],
  output: ["gate_logit", "token_logits"],
  max_tokens: reference.maxTokens,
  stride: reference.stride,
  span_head: "typed",
  span_labels: reference.labels,
  span_threshold: reference.threshold,
};
const L = reference.labels.length;

/** Replays the reference export's logits per window, keyed by the window's input ids. */
const replay = new Map<string, { gate: number; tokens: number[][] }>();
for (const s of Object.values(reference.samples)) {
  s.windows.forEach((w, i) => replay.set(s.ids.slice(w.start, w.end).join(","), { gate: s.gateLogits[i], tokens: s.tokenLogits[i] }));
}
/** An engine that answers each window with the archived reference logits; `labels` widens or narrows the token head. */
function replayEngine(labels = L): GateEngine {
  return {
    run: (input) => {
      const ids = Array.from(input, Number);
      const hit = ids.length === 2 ? { gate: -8, tokens: [] } : replay.get(ids.slice(1, -1).join(","));
      if (!hit) throw new Error("window not in the reference fixture");
      const logits = new Float32Array(ids.length * labels);
      if (labels === L) hit.tokens.forEach((row, t) => logits.set(row, (t + 1) * L));
      return { gateLogit: hit.gate, tokenLogits: logits };
    },
    memoryPages: 1,
    loadMs: 0,
    kernel: "unrolled",
  };
}

const model = createGateModel(replayEngine(), vocabText, { config });
const gate = new Gate(model, 0.5);
const key = (spans: readonly { start: number; end: number; type: string }[]) => spans.map((s) => [s.start, s.end, s.type]);
/** The Python reference reports code points; the package reports UTF-16 offsets. */
function toUtf16(text: string): (cp: number) => number {
  const at = [0];
  for (const ch of text) at.push(at[at.length - 1] + ch.length);
  return (cp) => at[cp];
}
const utf16Spans = (text: string, spans: readonly { start: number; end: number; type: string }[]) => {
  const u = toUtf16(text);
  return spans.map((s) => ({ ...s, start: u(s.start), end: u(s.end) }));
};

describe("joint gate + span head", () => {
  it("tokenizes with the reference vocabulary", async () => {
    expect(await sha256Hex(vocab)).toBe(reference.vocabSha256);
    expect(Object.keys(reference.samples).length).toBeGreaterThan(8);
  });

  for (const [id, sample] of Object.entries(reference.samples)) {
    it(`reproduces the Python reference gate probability, windows and spans for ${id}`, async () => {
      const { score, tokens } = await model.analyze(sample.text);
      expect(tokens?.ids).toEqual(sample.ids);
      const u = toUtf16(sample.text);
      expect(tokens?.offsets).toEqual(sample.offsets.map(([a, b]) => [u(a), u(b)]));
      expect(score.windows).toHaveLength(sample.gateLogits.length);
      expect(score.probability).toBeCloseTo(sample.probability, 6);
      const { spans, windows } = decodeSpans(sample.text, tokens!, reference.labels, reference.threshold);
      expect(windows).toEqual(sample.windows);
      expect(key(spans)).toEqual(key(utf16Spans(sample.text, sample.spans)));
      spans.forEach((s, i) => expect(s.probability).toBeCloseTo(sample.spans[i].score, 4));
    });
  }

  it("maps multi-byte, accented and astral tokens back to their UTF-16 offsets", () => {
    const pieces = vocabText.split("\n");
    for (const id of ["accents", "astral", "cjk-emoji"] as const) {
      const { text } = reference.samples[id];
      const { ids, offsets } = model.tokenizer.encodeWithOffsets(text);
      expect(ids).toEqual(reference.samples[id].ids);
      ids.forEach((tokenId, t) => {
        const surface = text.slice(offsets[t][0], offsets[t][1]).toLowerCase().normalize("NFD").replace(/\p{Mn}/gu, "");
        const piece = pieces[tokenId].replace(/^##/, "");
        expect(piece === "[UNK]" || surface === piece).toBe(true);
      });
    }
  });

  it("keeps validator spans in the UTF-16 offsets of validators.hard after astral characters", () => {
    const text = "😀😀 My SSN is 452-78-1934";
    expect(deterministicUnion(text, [])).toEqual([expect.objectContaining({ start: 15, end: 26, type: "GOV_ID" })]);
  });

  it("adds validator-unioned spans to the decision and keeps the verdict rule", async () => {
    const sample = reference.samples["ssn-routing"];
    const decision = await gate.check(sample.text);
    expect(decision.verdict).toBe("BLOCK");
    expect(decision.basis).toMatch(/^validator/);
    expect(decision.spans?.filter((s) => s.source === "validator").map((s) => s.type)).toContain("GOV_ID");
    expect((await gate.check("")).spans).toEqual([]);
  });

  it("keeps a typed model span that no validator matched", () => {
    const text = "Reach me at five five five, oh one four eight";
    const span = { start: 12, end: text.length, type: "PHONE", probability: 0.9, source: "model" } as const;
    expect(deterministicUnion(text, [span])).toEqual([span]);
  });

  it("redacts with the gate's span head and downloads nothing", async () => {
    const redactor = await createRedactor({ gate, modelBaseUrl: "http://127.0.0.1:9/unreachable/" });
    const sample = reference.samples["pdf-form"];
    expect((await redactor.redact(sample.text)).spans).toEqual((await gate.check(sample.text)).spans);
    expect(redactor.threshold).toBe(reference.threshold);
  });

  it("carries spans through the worker exactly like the in-process gate", async () => {
    const main: WorkerPort<any, any> & { terminate: () => void } = {
      onmessage: null,
      postMessage: (message) => queueMicrotask(() => worker.onmessage?.({ data: structuredClone(message) })),
      terminate: () => {},
    };
    const worker: WorkerPort<any, any> = {
      onmessage: null,
      postMessage: (message) => queueMicrotask(() => main.onmessage?.({ data: structuredClone(message) })),
    };
    registerGateWorker(worker, async () => gate);
    const remote = await createWorkerGate(main);
    for (const id of ["name-address", "astral", "long-letter"] as const) {
      const text = reference.samples[id].text;
      const decision = await remote.check(text);
      expect(decision.spans?.length).toBeGreaterThan(0);
      expect(decision).toEqual(await gate.check(text));
    }
  });

  it("refuses an engine without the span head and a config without O first", () => {
    const attempt = (engine: GateEngine, c: ModelConfig) => {
      try {
        createGateModel(engine, vocabText, { config: c });
        return null;
      } catch (e) {
        return e as GateModelError;
      }
    };
    const missing = attempt(replayEngine(0), config);
    expect(missing).toBeInstanceOf(GateModelError);
    expect(missing?.stage).toBe("graph");
    expect(attempt(replayEngine(), { ...config, output: "gate_logit" })).toMatchObject({ stage: "graph" });
    expect(attempt(replayEngine(), { ...config, span_labels: ["B-NAME", "O", "I-NAME"] })).toMatchObject({ stage: "graph" });
  });
});
