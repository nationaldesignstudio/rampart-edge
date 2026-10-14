import type { Detection, Label } from "./types.js";

/**
 * Overlap resolution. Deterministic greedy: hard beats soft; then longer span; then label priority
 * (checksum-backed / more specific first); then earlier start.
 * Consequences: an IPv4 (hard) is never also an SSN; a 16-digit card (hard) beats PHONE+ZIP pieces;
 * IPv6 beats a MAC-looking fragment; E.164 (with +) beats bare NANP; MRZ line beats passport/date pieces.
 */
export const LABEL_PRIORITY: Label[] = [
  "MRZ",
  "IBAN",
  "CREDIT_CARD",
  "IPV6",
  "IPV4",
  "MAC",
  "EMAIL",
  "URL",
  "PHONE_E164",
  "FR_NIR",
  "IT_CODICE_FISCALE",
  "VIN",
  "MEDICARE_MBI",
  "USCIS_RECEIPT",
  "USCIS_A_NUMBER",
  "DE_STEUER_ID",
  "BR_CPF",
  "ES_DNI",
  "ITIN",
  "SSN",
  "EIN",
  "ROUTING_NUMBER",
  "NL_BSN",
  "PT_NIF",
  "PASSPORT",
  "DRIVERS_LICENSE",
  "PHONE",
  "BANK_ACCOUNT",
  "BIC",
  "COORDINATES",
  "DOB",
  "DATE",
  "ZIP",
  "LICENSE_PLATE",
];
const RANK = new Map(LABEL_PRIORITY.map((l, i) => [l, i]));

export function resolveOverlaps(all: Detection[]): Detection[] {
  const sorted = [...all].sort((a, b) => {
    if (a.hard !== b.hard) return a.hard ? -1 : 1;
    const la = a.end - a.start,
      lb = b.end - b.start;
    if (la !== lb) return lb - la;
    const ra = RANK.get(a.label) ?? 99,
      rb = RANK.get(b.label) ?? 99;
    if (ra !== rb) return ra - rb;
    return a.start - b.start;
  });
  const out: Detection[] = [];
  for (const d of sorted) {
    if (out.some((o) => d.start < o.end && o.start < d.end)) continue;
    out.push(d);
  }
  return out.sort((a, b) => a.start - b.start);
}
