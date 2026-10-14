/**
 * Redact text extracted from a PDF before it is sent anywhere: the gate says BLOCK, then the redactor
 * reuses the gate's model (no second download) and replaces the PII it finds with [REDACTED].
 */
import { createGate } from "@nationaldesignstudio/rampart-edge";
import { createRedactor } from "@nationaldesignstudio/rampart-edge/redactor";

const modelBaseUrl = "/models/rampart-edge/";
const pageText = "APPLICANT INFORMATION\nName: Tomas Elling\nAddress: 77 Birch Hollow Rd Apt 3, Dayton OH 45402\nPhone: 937-555-0148";

const gate = await createGate({ modelBaseUrl });
const decision = await gate.check(pageText);

if (decision.verdict === "BLOCK") {
  const redactor = await createRedactor({ gate });
  const result = await redactor.redact(pageText);
  console.log(result.text);
  // APPLICANT INFORMATION
  // Name: [REDACTED]
  // Address: [REDACTED]
  // Phone: [REDACTED]
  for (const span of result.spans) console.log(span.start, span.end, span.type, span.probability.toFixed(3));
}
