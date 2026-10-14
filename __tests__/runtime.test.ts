import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { applyBundledRuntime, BUNDLED_RUNTIME_ORT_VERSION, bundledRuntimePaths } from "../src/runtime";

describe("bundled ORT Web engine", () => {
  it("resolves to the shipped wasm and mjs files", () => {
    const paths = bundledRuntimePaths();
    expect(existsSync(fileURLToPath(paths.wasm))).toBe(true);
    expect(existsSync(fileURLToPath(paths.mjs))).toBe(true);
  });

  it("points a matching onnxruntime-web at the bundled engine", () => {
    const env = { versions: { web: BUNDLED_RUNTIME_ORT_VERSION }, wasm: {} as { wasmPaths?: unknown } };
    expect(applyBundledRuntime(env)).toBe(true);
    expect(env.wasm.wasmPaths).toEqual(bundledRuntimePaths());
  });

  it("keeps the host's own wasmPaths or wasmBinary", () => {
    const withPaths = { versions: { web: BUNDLED_RUNTIME_ORT_VERSION }, wasm: { wasmPaths: "/ort/" } };
    const withBinary = { versions: { web: BUNDLED_RUNTIME_ORT_VERSION }, wasm: { wasmBinary: new Uint8Array(1) } };
    expect(applyBundledRuntime(withPaths)).toBe(false);
    expect(withPaths.wasm.wasmPaths).toBe("/ort/");
    expect(applyBundledRuntime(withBinary)).toBe(false);
    expect(withBinary.wasm).not.toHaveProperty("wasmPaths");
  });

  it("leaves other onnxruntime-web versions on their stock engine", () => {
    const env = { versions: { web: "1.26.0" }, wasm: {} as { wasmPaths?: unknown } };
    expect(applyBundledRuntime(env)).toBe(false);
    expect(env.wasm.wasmPaths).toBeUndefined();
    expect(applyBundledRuntime(undefined)).toBe(false);
  });
});
