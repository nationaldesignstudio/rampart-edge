// Size table for ORT wasm engine builds: raw, gzip -9 and brotli -q 11 bytes of
// ort-wasm-simd-threaded.{wasm,mjs}, with deltas against a reference (stock) dir.
// usage: node measure.mjs --stock <dir> [--label <dir>]... [--json out.json]
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { brotliCompressSync, constants, gzipSync } from "node:zlib";

const argv = process.argv.slice(2);
const dirs = [];
let jsonOut = null;
for (let i = 0; i < argv.length; i += 2) {
  if (argv[i] === "--json") jsonOut = argv[i + 1];
  else dirs.push({ label: argv[i].replace(/^--/, ""), dir: argv[i + 1] });
}
if (dirs.length === 0 || dirs[0].label !== "stock") {
  console.error("first argument must be --stock <dir>");
  process.exit(2);
}
const FILES = ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"];
const sizes = (buf) => ({
  raw: buf.length,
  gzip: gzipSync(buf, { level: 9 }).length,
  brotli: brotliCompressSync(buf, {
    params: {
      [constants.BROTLI_PARAM_QUALITY]: 11,
      [constants.BROTLI_PARAM_SIZE_HINT]: buf.length,
    },
  }).length,
});
const rows = dirs.map(({ label, dir }) => ({
  label,
  files: Object.fromEntries(FILES.map((f) => [f, sizes(readFileSync(join(dir, f)))])),
}));
const ref = rows[0];
const fmt = (n) => n.toLocaleString("en-US");
const mb = (n) => (n / 1_000_000).toFixed(2) + " MB";
const delta = (n, r) => (n === r ? "—" : `${n < r ? "−" : "+"}${fmt(Math.abs(n - r))} (${((100 * (n - r)) / r).toFixed(1)}%)`);
for (const f of FILES) {
  console.log(`\n### ${f}\n`);
  console.log("| build | raw | gzip -9 | brotli -11 | Δ raw vs stock | Δ gzip | Δ brotli |");
  console.log("|---|---:|---:|---:|---:|---:|---:|");
  for (const row of rows) {
    const s = row.files[f], r = ref.files[f];
    console.log(
      `| ${row.label} | ${fmt(s.raw)} (${mb(s.raw)}) | ${fmt(s.gzip)} (${mb(s.gzip)}) | ${fmt(s.brotli)} (${mb(s.brotli)}) | ${delta(s.raw, r.raw)} | ${delta(s.gzip, r.gzip)} | ${delta(s.brotli, r.brotli)} |`,
    );
  }
}
if (jsonOut) writeFileSync(jsonOut, JSON.stringify(rows, null, 2));
