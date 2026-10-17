import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  createEngine,
  EngineError,
  engineWasmBytes,
  ENGINE_SHAPE,
  parseWeightHeader,
  weightPayloadDigest,
  type GateEngine,
} from "../src/engine";
import { ENGINE_WASM_BYTES, ENGINE_WASM_SHA256 } from "../src/engine-wasm";

const wasm = new Uint8Array(readFileSync(new URL("../runtime/rampart-engine.wasm", import.meta.url)));
const weights = new Uint8Array(readFileSync(new URL("../model/model.rmpw", import.meta.url)));
const pin = JSON.parse(readFileSync(new URL("../model/model.rmpw.json", import.meta.url), "utf8")) as { bytes: number; sha256: string };

type RefWindow = { ids: number[]; gateLogit: number; tokenArgmax: number[]; tokenLogits: number[] };
type Reference = { ortVersion: string; browser: string; samples: Record<string, { windows: RefWindow[] }> };
const reference = JSON.parse(readFileSync(new URL("./fixtures/engine-reference.json", import.meta.url), "utf8")) as Reference;

const sha256 = (b: Uint8Array): string => createHash("sha256").update(b).digest("hex");
const argmax19 = (a: ArrayLike<number>, t: number): number => {
  let best = 0;
  for (let l = 1; l < 19; l++) if (a[t * 19 + l]! > a[t * 19 + best]!) best = l;
  return best;
};
const stage = async (p: Promise<unknown>): Promise<string> => {
  try {
    await p;
    return "ok";
  } catch (e) {
    return e instanceof EngineError ? e.stage : `not-EngineError:${String(e)}`;
  }
};

let shared: Promise<GateEngine> | undefined;
const engine = (): Promise<GateEngine> => (shared ??= createEngine(wasm, weights));

describe("engine artifacts", () => {
  it("committed wasm matches the generated inline module, imports nothing, fits the budget", () => {
    expect(wasm.byteLength).toBe(ENGINE_WASM_BYTES);
    expect(sha256(wasm)).toBe(ENGINE_WASM_SHA256);
    expect(sha256(engineWasmBytes())).toBe(ENGINE_WASM_SHA256);
    expect(wasm.byteLength).toBeLessThanOrEqual(48 * 1024);
    const mod = new WebAssembly.Module(wasm);
    expect(WebAssembly.Module.imports(mod)).toEqual([]);
    const exports = WebAssembly.Module.exports(mod).map((e) => e.name).sort();
    expect(exports).toEqual(["arena_alloc", "forward", "ids_ptr", "init", "mask_ptr", "memory", "tok_ptr", "version"]);
  });

  it("model.rmpw is the pinned golden file", () => {
    expect(weights.byteLength).toBe(1_331_672);
    expect(weights.byteLength).toBe(pin.bytes);
    expect(sha256(weights)).toBe(pin.sha256);
  });
});

describe("parseWeightHeader", () => {
  it("reads the RMPW v1 header fields", () => {
    expect(parseWeightHeader(weights)).toEqual({
      version: 1,
      hidden: 256,
      ffn: 1024,
      heads: 8,
      layers: 4,
      vocab: 23910,
      rank: 48,
      maxPos: 384,
      labels: 19,
      tensors: 42,
    });
    expect(parseWeightHeader(weights)).toEqual(ENGINE_SHAPE);
    expect(weightPayloadDigest(weights)).toBe(sha256(weights.subarray(928)));
  });

  it("rejects bad magic, truncation and shape mismatches with EngineError(format)", () => {
    const bad = weights.slice();
    bad[0] = 0x58;
    expect(() => parseWeightHeader(bad)).toThrow(EngineError);
    expect(() => parseWeightHeader(bad)).toThrow(expect.objectContaining({ stage: "format" }));
    expect(() => parseWeightHeader(weights.subarray(0, 100))).toThrow(expect.objectContaining({ stage: "format" }));
    const v2 = weights.slice();
    new DataView(v2.buffer).setUint32(4, 2, true);
    expect(() => parseWeightHeader(v2)).toThrow(expect.objectContaining({ stage: "format" }));
    const hid = weights.slice();
    new DataView(hid.buffer).setUint32(8, 512, true);
    expect(() => parseWeightHeader(hid)).toThrow(expect.objectContaining({ stage: "format" }));
  });
});

describe("createEngine", () => {
  it("loads from raw bytes and from a precompiled WebAssembly.Module", async () => {
    const e1 = await engine();
    expect(e1.loadMs).toBeGreaterThan(0);
    expect(e1.memoryPages).toBeGreaterThan(0);
    const e2 = await createEngine(await WebAssembly.compile(wasm), weights);
    const ids = [2, 1032, 1176, 1009, 3];
    expect(e2.run(ids).gateLogit).toBe(e1.run(ids).gateLogit);
  });

  it("rejects a flipped payload byte with EngineError(integrity)", async () => {
    const flipped = weights.slice();
    flipped[200_000] ^= 0x01;
    expect(await stage(createEngine(wasm, flipped))).toBe("integrity");
    const last = weights.slice();
    last[last.length - 1] ^= 0x80;
    expect(await stage(createEngine(wasm, last))).toBe("integrity");
  });

  it("maps bad magic / truncated weights / bad wasm to EngineError stages", async () => {
    const bad = weights.slice();
    bad[1] = 0;
    expect(await stage(createEngine(wasm, bad))).toBe("format");
    expect(await stage(createEngine(wasm, weights.subarray(0, 64)))).toBe("format");
    expect(await stage(createEngine(wasm, weights.subarray(0, 1_000_000)))).toBe("format");
    expect(await stage(createEngine(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 0xff]), weights))).toBe("wasm");
  });

  it("enforces 1 <= n <= 384 and valid token ids with EngineError(range)", async () => {
    const e = await engine();
    const bad = (ids: ArrayLike<number>, mask?: ArrayLike<number>): string => {
      try {
        e.run(ids, mask);
        return "ok";
      } catch (err) {
        return err instanceof EngineError ? err.stage : `not-EngineError:${String(err)}`;
      }
    };
    expect(bad([])).toBe("range");
    expect(bad(new Array<number>(385).fill(5))).toBe("range");
    expect(bad([2, 23910, 3])).toBe("range");
    expect(bad([2, -1, 3])).toBe("range");
    expect(bad([2, 5, 3], [1, 1])).toBe("range");
    expect(bad(new Array<number>(384).fill(5))).toBe("ok");
    expect(bad([2])).toBe("ok");
  });

  it("returns a copy of the token logits and honours the attention mask", async () => {
    const e = await engine();
    const a = e.run([2, 1032, 1176, 3]);
    expect(a.tokenLogits).toHaveLength(4 * 19);
    const before = a.tokenLogits[0];
    e.run([2, 9960, 65, 14404, 3]);
    expect(a.tokenLogits[0]).toBe(before);
    const padded = e.run([2, 1032, 1176, 3, 0, 0], [1, 1, 1, 1, 0, 0]);
    expect(Math.abs(padded.gateLogit - a.gateLogit)).toBeLessThan(1e-4);
  });
});

describe(`parity with browser onnxruntime-web ${reference.ortVersion} (${reference.browser})`, () => {
  const windows = Object.entries(reference.samples).flatMap(([name, s]) => s.windows.map((w, i) => [`${name}#${i}`, w] as const));
  it("archived fixture covers every joint-reference window", () => {
    expect(windows.length).toBeGreaterThanOrEqual(14);
  });
  it.each(windows)("%s: |dGate| <= 1e-4 and identical token argmax", async (_name, w) => {
    const e = await engine();
    const out = e.run(w.ids);
    expect(Math.abs(out.gateLogit - w.gateLogit)).toBeLessThanOrEqual(1e-4);
    expect(out.tokenLogits).toHaveLength(w.ids.length * 19);
    let maxTok = 0;
    for (let i = 0; i < out.tokenLogits.length; i++) maxTok = Math.max(maxTok, Math.abs(out.tokenLogits[i]! - w.tokenLogits[i]!));
    expect(maxTok).toBeLessThanOrEqual(1e-3);
    for (let t = 0; t < w.ids.length; t++) expect(argmax19(out.tokenLogits, t)).toBe(w.tokenArgmax[t]);
  });
});
