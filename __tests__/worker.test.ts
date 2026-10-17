import { describe, expect, it } from "vitest";

import { GateModelError } from "../src/model";
import { createWorkerGate, registerGateWorker, type WorkerPort } from "../src/worker";
import { loadLocalGate } from "./node-gate";

/** Two in-memory ports wired like a Worker and its global scope: a message posted on one arrives on the other. */
function channel(): { main: WorkerPort<any, any> & { terminate: () => void }; worker: WorkerPort<any, any> } {
  const main: WorkerPort<any, any> & { terminate: () => void } = {
    onmessage: null,
    postMessage: (message) => queueMicrotask(() => worker.onmessage?.({ data: message })),
    terminate: () => {},
  };
  const worker: WorkerPort<any, any> = {
    onmessage: null,
    postMessage: (message) => queueMicrotask(() => main.onmessage?.({ data: message })),
  };
  return { main, worker };
}

// Compile-time check: `createWorkerGate(new Worker(...))` from the README type-checks.
export const acceptsDomWorker = (worker: Worker) => createWorkerGate(worker);

describe("worker gate", () => {
  it("decides over the channel exactly like the in-process gate", async () => {
    const { main, worker } = channel();
    registerGateWorker(worker, () => loadLocalGate());
    const remote = await createWorkerGate(main);
    const local = await loadLocalGate();
    expect(remote.threshold).toBe(local.threshold);
    for (const text of ["my ssn is 452-78-1934", "how much is a passport renewal", "My landlord John Smith won't fix the heat"]) {
      expect(await remote.check(text)).toEqual(await local.check(text));
    }
    expect(remote.validators("call me at 312-206-0199").hard.map((d) => d.label)).toEqual(["PHONE"]);
  });

  it("rejects init with the worker's GateModelError so callers can fall back", async () => {
    const { main, worker } = channel();
    registerGateWorker(worker, async () => {
      throw new GateModelError("integrity", "model/model.rmpw: sha256 mismatch");
    });
    const error = await createWorkerGate(main).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(GateModelError);
    expect((error as GateModelError).stage).toBe("integrity");
  });

  it("rejects in-flight checks on terminate", async () => {
    const { main, worker } = channel();
    registerGateWorker(worker, () => loadLocalGate());
    const remote = await createWorkerGate(main);
    const pending = remote.check("hello");
    remote.terminate();
    await expect(pending).rejects.toMatchObject({ name: "GateModelError", stage: "runtime" });
  });
});
