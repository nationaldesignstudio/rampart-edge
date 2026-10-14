import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import type { AllowlistOptions } from "../src/allowlist";
import { Gate } from "../src/gate";
import { createGateModel, verifyPinned, type ModelConfig, type PinnedFile } from "../src/model";

/** Loads a model directory (default the committed `model/`), verified against its `config.json` pins, for Bun/Node. */
export async function loadLocalGate(
  threshold?: number,
  dir: URL = new URL("../model/", import.meta.url),
  allowlist?: AllowlistOptions,
): Promise<Gate> {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  const config = JSON.parse(readFileSync(new URL("config.json", dir), "utf8")) as ModelConfig & {
    block_threshold: number;
    files: Record<"onnx/model_q4.onnx" | "vocab.txt", PinnedFile>;
  };
  const modelBytes = new Uint8Array(readFileSync(new URL("onnx/model_q4.onnx", dir)));
  const vocabBytes = new Uint8Array(readFileSync(new URL("vocab.txt", dir)));
  await verifyPinned(`${dir.pathname}onnx/model_q4.onnx`, modelBytes, config.files["onnx/model_q4.onnx"]);
  await verifyPinned(`${dir.pathname}vocab.txt`, vocabBytes, config.files["vocab.txt"]);
  const model = await createGateModel(ort, modelBytes, new TextDecoder().decode(vocabBytes), { config });
  return new Gate(model, threshold ?? config.block_threshold, allowlist);
}
