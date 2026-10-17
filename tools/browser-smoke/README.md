# Browser smoke

Loads the built `dist/` in a real browser (Playwright Chromium, and WebKit when installed) from a plain HTTP server with
no COOP/COEP headers, fetches `model.rmpw` + `vocab.txt` over HTTP, and runs the gate on the main thread and in a Worker.
Reports engine load time, first `check`, p50 latency at 32 and 128 tokens, and wasm memory pages.

```bash
bun run build
PLAYWRIGHT_DIR=/path/to/node_modules/playwright bun run smoke:browser      # writes tools/browser-smoke/results.json
```

Playwright is not a dependency of this package; point `PLAYWRIGHT_DIR` at an installed copy (any project with
`playwright` installed and `bunx playwright install chromium webkit` run).
