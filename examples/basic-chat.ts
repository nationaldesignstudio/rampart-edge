/**
 * Gate a chat message in the browser before it is sent.
 *
 *   bun add @nationaldesignstudio/rampart-edge onnxruntime-web
 *
 * Self-host `onnx/model_q4.onnx` and `vocab.txt` (1.45 MB model + 173 KB vocab; this package's `model/` folder or the
 * Hugging Face repo root) next to your app and point `modelBaseUrl` at them; the files are checked against
 * pinned SHA-256s before they are used. Root-relative paths like the one below resolve in a browser only.
 */
import { createGate, createValidatorGate, GateModelError, type Gate } from "@nationaldesignstudio/rampart-edge";

let gate: Gate;
try {
  gate = await createGate({ modelBaseUrl: "/models/rampart-edge/" });
} catch (error) {
  if (!(error instanceof GateModelError)) throw error;
  console.warn(`Rampart Edge model unavailable (${error.stage}); validators only`, error);
  gate = createValidatorGate();
}

export async function send(message: string): Promise<void> {
  const decision = await gate.check(message);
  if (decision.verdict === "BLOCK") {
    // Nothing was sent. Ask the user to remove the personal details and try again.
    console.log(`blocked (${decision.basis}): p=${decision.model?.probability.toFixed(3) ?? "n/a"}`);
    return;
  }
  await fetch("/api/chat", { method: "POST", body: JSON.stringify({ message }) });
}

await send("The passport renewal fee is $130 and processing takes 6 to 8 weeks."); // sent
await send("My SSN is 452-78-1934 and I need to fix my tax return."); // blocked (validator-and-model)
