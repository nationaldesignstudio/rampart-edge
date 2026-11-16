import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import reference from "./fixtures/joint-reference.json";
import { Gate } from "../src/gate";
import type { GateEngine } from "../src/engine";
import { createGateModel, GateModelError, sha256Hex, type ModelConfig } from "../src/model";
import { createRedactor } from "../src/redactor";
import { decodeSpans, deterministicUnion, type RedactedSpanType } from "../src/spans";
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

  describe("identifier widening", () => {
    const at = (text: string, part: string) => [text.indexOf(part), text.indexOf(part) + part.length] as const;
    const model = (text: string, part: string, type: RedactedSpanType) => {
      const [start, end] = at(text, part);
      return { start, end, type, probability: 0.8, source: "model" } as const;
    };
    const spanOf = (text: string, part: string) => {
      const [start, end] = at(text, part);
      return expect.objectContaining({ start, end });
    };

    it("widens a partial identifier span to the whole identifier", () => {
      const text = "Account reference CU-4147-4452 was closed.";
      expect(deterministicUnion(text, [model(text, "4147", "OTHER_ID")])).toEqual([spanOf(text, "CU-4147-4452")]);
    });

    it("widens a partial profile URL but leaves wrapping brackets and sentence punctuation", () => {
      const text = "See the profile (https://www.facebook.com/dana.whitfield.272).";
      const out = deterministicUnion(text, [model(text, "dana.whitfield", "URL_HANDLE")]);
      expect(out).toEqual([spanOf(text, "https://www.facebook.com/dana.whitfield.272")]);
    });

    it("merges fragments of one token into a single span", () => {
      const text = "Case No. 1:23-cv-04417-JMF is set for hearing";
      const out = deterministicUnion(text, [model(text, "23-cv", "OTHER_ID"), model(text, "JMF", "OTHER_ID")]);
      expect(out).toEqual([spanOf(text, "1:23-cv-04417-JMF")]);
    });

    it("does not cross whitespace", () => {
      const text = "posts as @dwhitfield_88 daily";
      expect(deterministicUnion(text, [model(text, "whitfield", "URL_HANDLE")])).toEqual([spanOf(text, "@dwhitfield_88")]);
    });

    it("widens a profile URL to the URL the validators found, without angle brackets", () => {
      const text = "see <https://facebook.com/dana.w.272> ok";
      expect(deterministicUnion(text, [model(text, "dana", "URL_HANDLE")])).toEqual([spanOf(text, "https://facebook.com/dana.w.272")]);
    });

    it("stops at a glued label, list separators, quotes and brackets", () => {
      for (const text of [
        "Claim:CU-4147-4452;Agency:FEMA;Status:Closed",
        '{"case":"CU-4147-4452","agency":"FEMA","status":"closed"}',
        "ids: CU-4147-4452,CU-9999-0000,FEMA-PUBLIC-LINE",
        "see [CU-4147-4452] in the file",
        "Hamilton/CU-4147-4452 filed",
        "X\u200bCU-4147-4452\u200bY",
      ]) {
        expect(deterministicUnion(text, [model(text, "4147", "OTHER_ID")]), text).toEqual([spanOf(text, "CU-4147-4452")]);
      }
    });

    it("never widens into text a validator owns, or takes its label", () => {
      const text = "My SSN:452-78-1934 and ref REF-452-78-1934-B";
      const out = deterministicUnion(text, [model(text, "REF", "OTHER_ID")]);
      expect(out.map((s) => text.slice(s.start, s.end))).not.toContain("SSN:452-78-1934");
      expect(out.every((s) => !text.slice(s.start, s.end).includes("SSN"))).toBe(true);
    });

    it("leaves a span inside a long unbroken run as the model marked it", () => {
      const text = `data ${"ab1".repeat(100)} end`;
      const span = { start: 50, end: 54, type: "OTHER_ID", probability: 0.8, source: "model" } as const;
      expect(deterministicUnion(text, [span])).toEqual([span]);
    });

    it("completes an id or handle the head labelled PHONE or EMAIL, without taking a glued label", () => {
      const none = { block: false, hard: [], soft: [] };
      const phone = "Call FEMA:1-800-621-3362 today";
      expect(deterministicUnion(phone, [model(phone, "800-621", "PHONE")], none)).toEqual([spanOf(phone, "1-800-621-3362")]);
      const handle = "posts as @wokafor_92 daily";
      expect(deterministicUnion(handle, [model(handle, "ka", "EMAIL")])).toEqual([spanOf(handle, "@wokafor_92")]);
    });

    it("treats a date as a wall", () => {
      const text = "Tax number 4471209935_03/14/1999 on file";
      expect(deterministicUnion(text, [model(text, "44712", "GOV_ID")])).toEqual([spanOf(text, "4471209935")]);
    });

    it("does not grow over a plain word, only over a run that has a digit", () => {
      for (const [text, part, type] of [
        ["From: Whitfield, Dana J", "Whit", "EMAIL"],
        ["the Grant application", "G", "GOV_ID"],
        ["met Anne-Marie today", "-", "EMAIL"],
      ] as const) {
        const span = model(text, part, type);
        expect(deterministicUnion(text, [span]), text).toEqual([span]);
      }
      const mixed = "SSN 5512 884301 - Kreditkarten";
      const span = model(mixed, "884301 - K", "FINANCIAL");
      expect(deterministicUnion(mixed, [span], { block: false, hard: [], soft: [] })).toEqual([span]);
    });

    it("still completes a handle that has no digit", () => {
      const at = "follow @janedoe please";
      expect(deterministicUnion(at, [model(at, "aned", "EMAIL")])).toEqual([spanOf(at, "@janedoe")]);
      const bare = "posts as janedoe daily";
      expect(deterministicUnion(bare, [model(bare, "aned", "URL_HANDLE")])).toEqual([spanOf(bare, "janedoe")]);
    });

    it("stops at an ordinary word glued after the identifier with no space", () => {
      const none = { block: false, hard: [], soft: [] };
      for (const [text, part, whole] of [
        ["my case number is 73015289nDear Sir", "7301", "73015289"],
        ["Loan number 8841-4147Wells Fargo said", "8841", "8841-4147"],
      ] as const) {
        expect(deterministicUnion(text, [model(text, part, "OTHER_ID")], none), text).toEqual([spanOf(text, whole)]);
      }
    });

    it("keeps letters that belong to the identifier", () => {
      for (const [text, part, whole] of [
        ["policy KAH9180689 is active", "9180", "KAH9180689"],
        ["docket 1:23-cv-04417-JMF today", "04417", "1:23-cv-04417-JMF"],
        ["hash a3f9c21d7e on file", "3f9c", "a3f9c21d7e"],
        ["token 7d19f3cabcd issued", "7d19", "7d19f3cabcd"],
        ["session 4c1e-9a2f-77b0dead ok", "4c1e", "4c1e-9a2f-77b0dead"],
        ["ref CU4147 ok", "4147", "CU4147"],
        ["unit 4147b ok", "414", "4147b"],
      ] as const) {
        expect(deterministicUnion(text, [model(text, part, "OTHER_ID")]), text).toEqual([spanOf(text, whole)]);
      }
    });

    it("completes a username whatever part of it the head marked", () => {
      for (const [text, part, type, whole] of [
        ["My personal Instagram username is maplequill47.", "47", "OTHER_ID", "maplequill47"],
        ["login is jsmith1985 on the portal", "1985", "OTHER_ID", "jsmith1985"],
        ["account acct44719 was closed", "447", "OTHER_ID", "acct44719"],
        ["follow @meitanaka75 now", "75", "EMAIL", "@meitanaka75"],
      ] as const) {
        expect(deterministicUnion(text, [model(text, part, type)]), text).toEqual([spanOf(text, whole)]);
      }
    });

    it("counts a digit in any script", () => {
      const text = "ref CU-\u0664\u0661\u0664\u0667 noted";
      expect(deterministicUnion(text, [model(text, "\u0664\u0661", "OTHER_ID")])).toEqual([spanOf(text, "CU-\u0664\u0661\u0664\u0667")]);
    });

    it("can be switched off", () => {
      const text = "Account reference CU-4147-4452 was closed.";
      const span = model(text, "4147", "OTHER_ID");
      expect(deterministicUnion(text, [span], undefined, false)).toEqual([span]);
    });

    it("leaves multi-word types alone", () => {
      const text = "Plaintiff Dana-Marie Whitfield appeared";
      const span = model(text, "Marie Whitfield", "NAME");
      expect(deterministicUnion(text, [span])).toEqual([span]);
    });
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
