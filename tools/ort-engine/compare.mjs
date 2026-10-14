// A/B: run model_q4.onnx through two onnxruntime-web engines (stock vs custom) in Node.
// usage: node compare.mjs --model <onnx> --tokenizer <dir> --a <wasm dir> --b <wasm dir> [--threads N] [--count 200] [--opt all|extended|basic|disabled]
import { env as hfEnv, AutoTokenizer } from "@huggingface/transformers";
import { buildTexts } from "./texts.mjs";
import { fork } from "node:child_process";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
  if (a.startsWith("--")) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith("--") ? arr[i + 1] : "1"]);
  return acc;
}, []));

if (args.worker) {
  // child: load one engine, run all inputs, send back logits + timings
  const ort = await import("onnxruntime-web");
  ort.env.wasm.wasmPaths = args.wasm.endsWith("/") ? args.wasm : args.wasm + "/";
  ort.env.wasm.numThreads = Number(args.threads ?? 1);
  ort.env.wasm.proxy = false;
  const inputs = JSON.parse(await import("node:fs").then((m) => m.readFileSync(args.inputs, "utf8")));
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(args.model, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: args.opt ?? "all",
  });
  const loadMs = performance.now() - t0;
  const results = [];
  for (const enc of inputs) {
    const n = enc.input_ids.length;
    const feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(enc.input_ids.map(BigInt)), [1, n]),
      attention_mask: new ort.Tensor("int64", BigInt64Array.from(enc.attention_mask.map(BigInt)), [1, n]),
      token_type_ids: new ort.Tensor("int64", BigInt64Array.from(enc.token_type_ids.map(BigInt)), [1, n]),
    };
    const s = performance.now();
    const out = await session.run(feeds);
    const ms = performance.now() - s;
    const data = Array.from(out.logits.data);
    if (!data.every(Number.isFinite)) throw new Error(`non-finite logits from ${args.wasm} on input ${results.length}`);
    results.push({ ms, dims: out.logits.dims, data });
  }
  // warm-up excluded: rerun first 50 for stable timing
  const timings = [];
  for (const enc of inputs.slice(0, 50)) {
    const n = enc.input_ids.length;
    const feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(enc.input_ids.map(BigInt)), [1, n]),
      attention_mask: new ort.Tensor("int64", BigInt64Array.from(enc.attention_mask.map(BigInt)), [1, n]),
      token_type_ids: new ort.Tensor("int64", BigInt64Array.from(enc.token_type_ids.map(BigInt)), [1, n]),
    };
    const s = performance.now();
    await session.run(feeds);
    timings.push(performance.now() - s);
  }
  await session.release();
  await new Promise((r) => process.send({ loadMs, results, timings }, r));
  process.exit(0);
}

hfEnv.allowRemoteModels = false;
hfEnv.allowLocalModels = true;
const tokenizer = await AutoTokenizer.from_pretrained(resolve(args.tokenizer), { local_files_only: true });
const texts = buildTexts(Number(args.count ?? 200));
const encoded = texts.map((t) => {
  const e = tokenizer(t, { add_special_tokens: true });
  const ids = Array.from(e.input_ids.data, Number);
  return {
    input_ids: ids,
    attention_mask: Array.from(e.attention_mask.data, Number),
    token_type_ids: e.token_type_ids ? Array.from(e.token_type_ids.data, Number) : ids.map(() => 0),
  };
});
const fs = await import("node:fs");
const { tmpdir } = await import("node:os");
const workDir = fs.mkdtempSync(resolve(tmpdir(), "ort-compare-"));
const inputsPath = resolve(workDir, "inputs.json");
fs.writeFileSync(inputsPath, JSON.stringify(encoded));
const lens = encoded.map((e) => e.input_ids.length);
console.log(`texts=${texts.length} tokens min/median/max = ${Math.min(...lens)}/${lens.sort((a, b) => a - b)[lens.length >> 1]}/${Math.max(...lens)}`);

function runEngine(wasm) {
  return new Promise((res, rej) => {
    const child = fork(fileURLToPath(import.meta.url), [
      "--worker", "--wasm", wasm, "--model", resolve(args.model), "--inputs", inputsPath,
      "--threads", args.threads ?? "1", "--opt", args.opt ?? "all",
    ], { stdio: ["inherit", "inherit", "inherit", "ipc"] });
    child.on("message", res);
    child.on("exit", (c) => c !== 0 && rej(new Error(`engine ${wasm} exited ${c}`)));
  });
}
const pct = (arr, p) => { const s = [...arr].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const A = await runEngine(resolve(args.a));
const B = await runEngine(resolve(args.b));
fs.rmSync(workDir, { recursive: true, force: true });
let maxAbs = 0, argmaxAgree = 0, tokens = 0, labelsA = 0, nonFinite = 0;
for (let i = 0; i < A.results.length; i++) {
  const a = A.results[i], b = B.results[i];
  if (a.data.length !== b.data.length) throw new Error(`shape mismatch at ${i}`);
  const C = a.dims[2];
  for (let t = 0; t < a.dims[1]; t++) {
    let am = 0, bm = 0;
    for (let c = 0; c < C; c++) {
      const av = a.data[t * C + c], bv = b.data[t * C + c];
      // JSON IPC turns NaN/±Infinity into null; never let that read as agreement.
      if (!Number.isFinite(av) || !Number.isFinite(bv)) { nonFinite++; maxAbs = Infinity; }
      const d = Math.abs(av - bv);
      if (d > maxAbs) maxAbs = d;
      if (a.data[t * C + c] > a.data[t * C + am]) am = c;
      if (b.data[t * C + c] > b.data[t * C + bm]) bm = c;
    }
    tokens++; if (am === bm) argmaxAgree++; if (am !== 0) labelsA++;
  }
}
const fmt = (e) => `load ${e.loadMs.toFixed(0)} ms; run p50 ${pct(e.timings, 0.5).toFixed(2)} ms, p95 ${pct(e.timings, 0.95).toFixed(2)} ms (first-pass p50 ${pct(e.results.map((r) => r.ms), 0.5).toFixed(2)} ms)`;
console.log(`threads=${args.threads ?? 1} opt=${args.opt ?? "all"}`);
console.log(`A (${args.a}): ${fmt(A)}`);
console.log(`B (${args.b}): ${fmt(B)}`);
if (nonFinite > 0) argmaxAgree = 0;
console.log(`tokens=${tokens} non-O argmax in A=${labelsA} max|Δlogit|=${maxAbs} argmax agreement=${argmaxAgree}/${tokens} (${(100 * argmaxAgree / tokens).toFixed(3)}%)${nonFinite ? ` NON-FINITE LOGITS: ${nonFinite}` : ""}`);
if (nonFinite > 0) process.exit(1);
