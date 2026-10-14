/**
 * Deterministic validators of the Rampart Edge gate: structured identifiers (SSN, ITIN, cards, phones,
 * emails, addresses, ...) with checksums, allowlists and overlap resolution. A hard hit BLOCKs on its
 * own; the classifier covers everything the validators cannot see.
 */
import type { Detection, Norm } from "./types.js";
import { normalize } from "./normalize.js";
import { ALL_RULES, digitCandidates } from "./rules.js";
import { resolveOverlaps } from "./resolve.js";
import { applyAllowlists } from "./allowlist.js";

export type { Detection, Label } from "./types.js";

/** All raw candidates from every rule (may overlap). */
export function detectAll(text: string): Detection[] {
  const n: Norm = normalize(text);
  const cands = digitCandidates(n);
  const out: Detection[] = [];
  for (const r of ALL_RULES) out.push(...r.fn(n, cands));
  return out;
}

const NON_US_IDS = new Set([
  "ES_DNI",
  "FR_NIR",
  "DE_STEUER_ID",
  "IT_CODICE_FISCALE",
  "PT_NIF",
  "NL_BSN",
  "BR_CPF",
]);
const US_NUMERIC_IDS = new Set(["SSN", "ITIN", "EIN", "ROUTING_NUMBER"]);
/** A cued, checksum-valid non-US id (soft) owns its span: the same digits must not resurface as a US id. */
function suppressUsCollisions(dets: Detection[]): Detection[] {
  const foreign = dets.filter((d) => NON_US_IDS.has(d.label) && d.evidence.includes(";kw"));
  if (foreign.length === 0) return dets;
  return dets.filter(
    (d) =>
      !(US_NUMERIC_IDS.has(d.label) && foreign.some((f) => d.start < f.end && f.start < d.end)),
  );
}

/** Non-overlapping detections after overlap resolution and public-entity allowlisting. */
export function detect(text: string): Detection[] {
  return applyAllowlists(resolveOverlaps(suppressUsCollisions(detectAll(text))));
}

export interface GateResult {
  block: boolean;
  hard: Detection[];
  soft: Detection[];
}
/** Deterministic-only gate: block iff any resolved detection is hard. */
export function gate(text: string): GateResult {
  const d = detect(text);
  const hard = d.filter((x) => x.hard),
    soft = d.filter((x) => !x.hard);
  return { block: hard.length > 0, hard, soft };
}
