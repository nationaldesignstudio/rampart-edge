import { readFileSync } from "node:fs";
import type { AllowlistOptions } from "../src/allowlist";
import { Gate } from "../src/gate";
import { createGateModelFromWeights, verifyPinned, type GateModel, type ModelConfig, type PinnedFile } from "../src/model";

type LocalConfig = ModelConfig & { block_threshold: number; files: Record<"model.rmpw" | "vocab.txt", PinnedFile> };

/** Reads a model directory (default the committed `model/`) and verifies `model.rmpw` and `vocab.txt` against its `config.json` pins. */
export async function loadLocalFiles(dir: URL = new URL("../model/", import.meta.url)): Promise<{ config: LocalConfig; weights: Uint8Array; vocab: string }> {
  const config = JSON.parse(readFileSync(new URL("config.json", dir), "utf8")) as LocalConfig;
  const weights = new Uint8Array(readFileSync(new URL("model.rmpw", dir)));
  const vocabBytes = new Uint8Array(readFileSync(new URL("vocab.txt", dir)));
  await verifyPinned(`${dir.pathname}model.rmpw`, weights, config.files["model.rmpw"]);
  await verifyPinned(`${dir.pathname}vocab.txt`, vocabBytes, config.files["vocab.txt"]);
  return { config, weights, vocab: new TextDecoder().decode(vocabBytes) };
}

/** The committed model on the inlined engine, for Bun/Node. */
export async function loadLocalModel(dir?: URL): Promise<GateModel> {
  const { config, weights, vocab } = await loadLocalFiles(dir);
  return createGateModelFromWeights({ weights, vocab, config });
}

/** Loads a model directory (default the committed `model/`), verified against its `config.json` pins, as a gate for Bun/Node. */
export async function loadLocalGate(threshold?: number, dir?: URL, allowlist?: AllowlistOptions): Promise<Gate> {
  const { config, weights, vocab } = await loadLocalFiles(dir);
  const model = await createGateModelFromWeights({ weights, vocab, config });
  return new Gate(model, threshold ?? config.block_threshold, allowlist);
}
