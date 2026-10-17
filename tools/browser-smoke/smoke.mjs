#!/usr/bin/env bun
// Playwright smoke of the built dist/ in a plain page and a Worker over HTTP without COOP/COEP. See README.md.
import { createServer } from "node:http";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const pwDir = process.env.PLAYWRIGHT_DIR;
const { chromium, webkit } = await import(pwDir ? pathToFileURL(join(pwDir, "index.mjs")).href : "playwright");
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".txt": "text/plain", ".rmpw": "application/octet-stream", ".wasm": "application/wasm", ".map": "application/json" };

if (!existsSync(join(root, "dist", "index.js"))) throw new Error("run `bun run build` first");
const server = createServer((req, res) => {
  const path = req.url === "/" ? "/tools/browser-smoke/page.html" : decodeURIComponent(req.url.split("?")[0]);
  try {
    const body = readFileSync(join(root, path));
    res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream", "cache-control": "no-store" });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const origin = `http://127.0.0.1:${server.address().port}`;

const results = { origin, browsers: {} };
for (const [name, type] of [["chromium", chromium], ["webkit", webkit]]) {
  let browser;
  try {
    browser = await type.launch();
  } catch (error) {
    results.browsers[name] = { skipped: String(error.message).split("\n")[0] };
    continue;
  }
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`${origin}/`);
  await page.waitForFunction(() => window.__smoke !== undefined, null, { timeout: 120_000 });
  const smoke = await page.evaluate(() => window.__smoke);
  results.browsers[name] = { version: browser.version(), ...smoke, pageErrors: errors };
  await browser.close();
}
server.close();
writeFileSync(join(root, "tools", "browser-smoke", "results.json"), JSON.stringify(results, null, 2) + "\n");
console.log(JSON.stringify(results, null, 2));
for (const r of Object.values(results.browsers)) if (r.error || (r.pageErrors && r.pageErrors.length)) process.exitCode = 1;
