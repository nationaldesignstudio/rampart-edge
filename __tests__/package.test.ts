/** The shipped artifacts agree with each other: inlined engine = `runtime/rampart-engine.wasm` = `config.engine` pin; file pins = files. */
import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import config from "../model/config.json";
import manifest from "../model/model.rmpw.json";
import pkg from "../package.json";
import { engineWasmBytes, parseWeightHeader } from "../src/engine";
import { MODEL_ENGINE, MODEL_FILES } from "../src/gate";
import { sha256Hex } from "../src/model";

const root = new URL("../", import.meta.url);

describe("shipped artifacts", () => {
  it("inlines the same engine wasm that is shipped and pinned", async () => {
    const inlined = engineWasmBytes();
    const shipped = new Uint8Array(readFileSync(new URL("runtime/rampart-engine.wasm", root)));
    expect(inlined.byteLength).toBe(MODEL_ENGINE.wasm_bytes);
    expect(await sha256Hex(inlined)).toBe(MODEL_ENGINE.wasm_sha256);
    expect(await sha256Hex(shipped)).toBe(MODEL_ENGINE.wasm_sha256);
    expect(MODEL_ENGINE).toMatchObject({ format: "rmpw", version: 1 });
  });

  it("pins the committed model.rmpw and vocab.txt", async () => {
    const weights = new Uint8Array(readFileSync(new URL("model/model.rmpw", root)));
    const vocab = new Uint8Array(readFileSync(new URL("model/vocab.txt", root)));
    expect(MODEL_FILES["model.rmpw"]).toEqual({ bytes: manifest.bytes, sha256: manifest.sha256 });
    expect(weights.byteLength).toBe(MODEL_FILES["model.rmpw"].bytes);
    expect(await sha256Hex(weights)).toBe(MODEL_FILES["model.rmpw"].sha256);
    expect(vocab.byteLength).toBe(MODEL_FILES["vocab.txt"].bytes);
    expect(await sha256Hex(vocab)).toBe(MODEL_FILES["vocab.txt"].sha256);
    const header = parseWeightHeader(weights);
    expect(header).toMatchObject({ version: MODEL_ENGINE.version, vocab: config.architecture.vocab_size, hidden: config.architecture.hidden_size, layers: config.architecture.layers, rank: config.architecture.embedding_rank, maxPos: config.max_tokens, labels: config.span_labels.length });
  });

  it("ships every file the package lists and no runtime dependency", () => {
    for (const entry of pkg.files) if (entry !== "dist") expect(existsSync(new URL(entry, root)), entry).toBe(true);
    expect("dependencies" in pkg || "peerDependencies" in pkg).toBe(false);
  });
});
