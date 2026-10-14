/**
 * Worker entry: bundle this file and hand its URL to `new Worker(...)`.
 *
 *   // main thread
 *   import { createWorkerGate, createValidatorGate } from "@nationaldesignstudio/rampart-edge";
 *   const worker = new Worker(new URL("./gate-worker.ts", import.meta.url), { type: "module" });
 *   const gate = await createWorkerGate(worker, { modelBaseUrl: "/models/rampart-edge/" }).catch(() => createValidatorGate());
 *   const decision = await gate.check(draft);   // same shape as the main-thread gate
 */
import { registerGateWorker } from "@nationaldesignstudio/rampart-edge/worker";

registerGateWorker(self as unknown as Parameters<typeof registerGateWorker>[0]);
