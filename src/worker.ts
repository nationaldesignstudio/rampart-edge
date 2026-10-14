/**
 * Web Worker host for the gate.
 *
 * Model inference takes ~30 ms and would freeze the page while it runs, so the gate can live on a
 * worker thread. The main thread talks to it through {@link createWorkerGate}, which gives back the
 * same `check(text)` contract as {@link Gate}, so the rest of an app does not care which thread
 * decides. Validators also run locally on the main thread via `validators(text)` for synchronous use.
 *
 * Bundle this module as the worker entry and call `registerGateWorker(self)` (see
 * `examples/gate-worker.ts`); create the client with `createWorkerGate(new Worker(new URL(...)))`.
 */

import { compileAllowlist, type CompiledAllowlist } from "./allowlist.js";
import { createGate, Gate, type GateOptions } from "./gate.js";
import { GateModelError } from "./model.js";
import type { GateDecision } from "./scoring.js";
import { gate as validatorGate, type GateResult } from "./validators/index.js";

/** `GateOptions` minus the parts that cannot cross a `postMessage` boundary (`ort`, `signal`). */
export type WorkerGateOptions = Omit<GateOptions, "ort" | "signal" | "modelBaseUrl"> & {
  readonly modelBaseUrl?: string;
};

type InitMessage = { readonly kind: "init"; readonly options: WorkerGateOptions };
type CheckMessage = { readonly kind: "check"; readonly id: number; readonly text: string };
type Inbound = InitMessage | CheckMessage;

type ReadyMessage = { readonly kind: "ready"; readonly threshold: number };
type ResultMessage = { readonly kind: "result"; readonly id: number; readonly decision: GateDecision };
type ErrorMessage = { readonly kind: "error"; readonly id?: number; readonly code?: GateModelError["stage"]; readonly message: string };
type Outbound = ReadyMessage | ResultMessage | ErrorMessage;

/** The subset of `Worker` / `DedicatedWorkerGlobalScope` the two sides use. */
export type WorkerPort<In, Out> = {
  // Method-style signature so a DOM `Worker`, whose handler takes a full `MessageEvent`, is accepted under `strict`.
  onmessage: { handle(event: { data: In }): void }["handle"] | null;
  postMessage: (message: Out) => void;
};

function toErrorMessage(error: unknown, id?: number): ErrorMessage {
  if (error instanceof GateModelError) return { kind: "error", id, code: error.stage, message: error.message };
  return { kind: "error", id, message: error instanceof Error ? error.message : String(error) };
}

// --- Worker side ---

/**
 * Register the message handler inside the worker. `load` builds the gate from the client's options;
 * the default downloads and verifies the pinned model like `createGate`.
 */
export function registerGateWorker(
  scope: WorkerPort<Inbound, Outbound>,
  load: (options: WorkerGateOptions) => Promise<Gate> = (options) => createGate(options),
): void {
  let gatePromise: Promise<Gate> | null = null;
  scope.onmessage = async (event) => {
    const message = event.data;
    if (message.kind === "init") {
      try {
        gatePromise = load(message.options);
        const gate = await gatePromise;
        scope.postMessage({ kind: "ready", threshold: gate.threshold });
      } catch (error) {
        gatePromise = null;
        scope.postMessage(toErrorMessage(error));
      }
      return;
    }
    try {
      if (gatePromise === null) throw new GateModelError("runtime", "gate worker is not initialized");
      const gate = await gatePromise;
      scope.postMessage({ kind: "result", id: message.id, decision: await gate.check(message.text) });
    } catch (error) {
      scope.postMessage(toErrorMessage(error, message.id));
    }
  };
}

// --- Main-thread side ---

/** A gate whose model runs in a worker. Same `check` contract as {@link Gate}. */
export class WorkerGate {
  readonly threshold: number;
  readonly #allowlist: CompiledAllowlist;
  readonly #worker: WorkerPort<Outbound, Inbound> & { terminate?: () => void };
  readonly #pending = new Map<number, { resolve: (d: GateDecision) => void; reject: (e: Error) => void }>();
  #nextId = 0;

  constructor(worker: WorkerPort<Outbound, Inbound> & { terminate?: () => void }, threshold: number, allowlist: CompiledAllowlist = compileAllowlist()) {
    this.#worker = worker;
    this.threshold = threshold;
    this.#allowlist = allowlist;
    worker.onmessage = (event) => this.#receive(event.data);
  }

  /** Deterministic validators only, run here on the calling thread. Synchronous. */
  validators(text: string): GateResult {
    return this.#allowlist.applyToValidators(validatorGate(text));
  }

  check(text: string): Promise<GateDecision> {
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#worker.postMessage({ kind: "check", id, text });
    });
  }

  /** Stops the worker; every in-flight `check` rejects. */
  terminate(): void {
    this.#worker.terminate?.();
    for (const [id, entry] of this.#pending) {
      this.#pending.delete(id);
      entry.reject(new GateModelError("runtime", "gate worker terminated"));
    }
  }

  #receive(data: Outbound): void {
    if (data.kind === "ready" || data.id === undefined) return;
    const entry = this.#pending.get(data.id);
    if (entry === undefined) return;
    this.#pending.delete(data.id);
    if (data.kind === "result") entry.resolve(data.decision);
    else entry.reject(data.code ? new GateModelError(data.code, data.message) : new Error(data.message));
  }
}

/**
 * Sends `init` to a worker running {@link registerGateWorker} and resolves with a {@link WorkerGate}
 * once the model is loaded and verified there. Rejects with `GateModelError` if loading fails, so
 * callers can fall back to `createValidatorGate()`.
 */
export function createWorkerGate(
  worker: WorkerPort<Outbound, Inbound> & { terminate?: () => void },
  options: WorkerGateOptions = {},
): Promise<WorkerGate> {
  return new Promise((resolve, reject) => {
    worker.onmessage = (event) => {
      const data = event.data;
      if (data.kind === "ready") resolve(new WorkerGate(worker, data.threshold, compileAllowlist(options.allowlist)));
      else if (data.kind === "error" && data.id === undefined)
        reject(data.code ? new GateModelError(data.code, data.message) : new Error(data.message));
    };
    worker.postMessage({ kind: "init", options });
  });
}
