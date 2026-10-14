import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import reference from "./fixtures/joint-reference.json";
import * as publicApi from "../index";
import { loadLocalGate } from "./node-gate";
import { loadLocalRedactor } from "./node-redactor";
import { applyRedactions, REDACTED_PLACEHOLDER, Redactor, spanWindows } from "../src/redactor";

const redactor = await loadLocalRedactor();
const key = (spans: readonly { start: number; end: number; type: string }[]) => spans.map((s) => [s.start, s.end, s.type]);
/** The Python reference reports code points; the package reports UTF-16 offsets. */
function utf16Key(text: string, spans: readonly { start: number; end: number; type: string }[]) {
  const at = [0];
  for (const ch of text) at.push(at[at.length - 1] + ch.length);
  return spans.map((s) => [at[s.start], at[s.end], s.type]);
}

describe("span redactor (gate model's span head, onnxruntime-web, wasm)", () => {
  it("decodes the reference spans at the model's threshold", async () => {
    expect(redactor.threshold).toBe(reference.threshold);
    for (const sample of Object.values(reference.samples)) {
      expect(key((await redactor.model.spans(sample.text)).spans)).toEqual(utf16Key(sample.text, sample.spans));
    }
  });

  it("redacts with [REDACTED] using UTF-16 offsets", async () => {
    const text = reference.samples.astral.text;
    const result = await redactor.redact(text);
    expect(result.text).toBe(`🙂🙂 Thanks! Wire it to card ${REDACTED_PLACEHOLDER} for ${REDACTED_PLACEHOLDER} 👍`);
    expect(result.spans[0]).toMatchObject({ type: "FINANCIAL", source: "validator", rule: "card" });
    expect(text.slice(result.spans[1].start, result.spans[1].end)).toBe("Priya Ramanathan");
    expect(applyRedactions(text, [])).toBe(text);
  });

  it("returns gate spans that slice the input directly after astral characters", async () => {
    const gate = await loadLocalGate();
    const text = "😀😀 My landlord Marcus Delgado has my SSN 452-78-1934";
    const decision = await gate.check(text);
    expect(decision.validators.hard.find((d) => d.label === "SSN")).toMatchObject({ start: 43, end: 54 });
    expect(decision.spans?.map((s) => [text.slice(s.start, s.end), s.type])).toEqual([
      ["Marcus Delgado", "NAME"],
      ["452-78-1934", "GOV_ID"],
    ]);
  });

  it("rejects a span threshold outside (0, 1] and accepts null (argmax)", () => {
    for (const threshold of [Number.NaN, -1, 0, 2]) {
      expect(() => new Redactor(redactor.model, { threshold })).toThrow(/span threshold must be a finite number in \(0, 1\]/);
    }
    expect(new Redactor(redactor.model, { threshold: null }).threshold).toBeNull();
  });

  it("redacts a keyed date of birth whole and leaves a bare date alone", async () => {
    const form = await redactor.redact("APPLICANT: Jane Q. Testperson\nDOB: 07/22/1983\nSSN: 123-45-6789");
    expect(form.text).toContain("DOB: [REDACTED]\n");
    expect(form.spans.find((s) => s.type === "DOB_KEYED")).toBeDefined();
    const hearing = "The hearing is set for 03/14/2026 at the county courthouse.";
    expect((await redactor.redact(hearing)).text).toBe(hearing);
  });

  it("leaves .gov/.mil emails and redacts other named emails", async () => {
    const result = await redactor.redact("Email jane.doe@va.gov or j.doe@army.mil, not jane.doe@gmail.com");
    expect(result.text).toBe("Email jane.doe@va.gov or j.doe@army.mil, not [REDACTED]");
  });

  it("returns nothing for empty text", async () => {
    expect(await redactor.redact("")).toEqual({ text: "", spans: [], windows: 0, threshold: reference.threshold });
  });

  it("windows long inputs like the Python reference", () => {
    expect(spanWindows(10)).toEqual([{ start: 0, end: 10, coreStart: 0, coreEnd: 10 }]);
    expect(spanWindows(766).map((w) => [w.start, w.end])).toEqual([[0, 382], [286, 668], [384, 766]]);
  });

  it("stays out of the main entry", () => {
    expect(Object.keys(publicApi).some((name) => /redact/i.test(name))).toBe(false);
    expect(readFileSync(new URL("../index.ts", import.meta.url), "utf8")).not.toContain("src/redactor");
  });
});
