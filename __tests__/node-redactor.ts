import { redactorModelFromGate, Redactor, type RedactOptions } from "../src/redactor";
import { loadLocalGate } from "./node-gate";

/** Loads the committed `model/` (verified against `model/config.json`) as a redactor for Bun/Node. */
export async function loadLocalRedactor(options?: RedactOptions): Promise<Redactor> {
  return new Redactor(redactorModelFromGate((await loadLocalGate()).model!), options);
}
