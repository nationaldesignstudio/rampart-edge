import type { Detection, Label } from "./types.js";

/**
 * Fixed allowlists that demote a hard detection to soft:
 *   E1 — format-only examples, RFC documentation IPs and fictional phone numbers that cannot identify
 *      or contact a person. Complete SSNs, emails and cards are excluded.
 *   E2 — published generic mailboxes of public bodies outside .gov/.mil (those always allow, `rules.ts`).
 * Names, phone numbers and addresses are applied by the gate's allowlist (`src/allowlist.ts`).
 */
const KEYPAD: Record<string, string> = {
  ABC: "2",
  DEF: "3",
  GHI: "4",
  JKL: "5",
  MNO: "6",
  PQRS: "7",
  TUV: "8",
  WXYZ: "9",
};
function keypad(s: string): string {
  return s.toUpperCase().replace(/[A-Z]/g, (ch) => {
    const entry = Object.entries(KEYPAD).find(([letters]) => letters.includes(ch));
    return entry ? entry[1] : ch;
  });
}
export function canonPhone(v: string): string {
  let d = keypad(v).replace(/\D/g, "");
  if (d.length === 11 && d[0] === "1") d = d.slice(1);
  return d;
}
export function canonId(v: string): string {
  return v.toUpperCase().replace(/[^A-Z0-9]/g, "");
}

const NUMERIC: Set<Label> = new Set([
  "SSN",
  "ITIN",
  "EIN",
  "PHONE",
  "PHONE_E164",
  "ROUTING_NUMBER",
  "ZIP",
  "CREDIT_CARD",
  "BANK_ACCOUNT",
]);

/** `example_ssn` rows: complete example SSNs block like any other SSN. */
const EXAMPLE_SSNS = [
  "000-00-0000",
  "078-05-1120",
  "123-45-6789",
  "219-09-9999",
  "666-12-3456",
  "900-00-0000",
  "987-65-4320",
  "987-65-4321",
  "987-65-4329",
];
const EXAMPLE_PHONES = [
  "(555) 555-5555",
  "123-456-7890",
  "202-555-0100",
  "202-555-0123",
  "212-555-0100",
  "305-555-0142",
  "312-555-0199",
  "415-555-0150",
  "555-0100",
  "555-0199",
  "555-555-5555",
  "617-555-0188",
  "713-555-0177",
];
const EXAMPLE_IDS = [
  "20500-0001",
  "21235-0001",
  "60680-4120",
  "73301-0002",
  "192.0.2.1",
  "198.51.100.1",
  "2001:db8::1",
  "203.0.113.1",
  "A-123-456-789",
  "A123456789",
  "A-XXX-XXX-XXX",
  "RSSMRA85T10A562S",
  "00000000T",
  "12345678Z",
  "X1234567L",
  "00-0000000",
  "12-3456789",
  "XX-XXXXXXX",
  "9XX-7X-XXXX",
  "BR18 0036 0305 0000 1000 9795 493C 1",
  "DE89 3704 0044 0532 0130 00",
  "ES91 2100 0418 4502 0005 1332",
  "FR14 2004 1010 0505 0001 3M02 606",
  "GB82 WEST 1234 5698 7654 32",
  "IT60 X054 2811 1010 0000 0123 456",
  "NL91 ABNA 0417 1643 00",
  "PT50 0002 0123 1234 5678 9015 4",
  "1EG4-TE5-MK73",
  "1EG4TE5MK73",
  "EAC1234567890",
  "IOE0912345678",
  "011000015",
  "021000021",
  "026009593",
  "031176110",
  "121000248",
];
const ROLE_MAILBOXES = [
  "311@cityofchicago.org",
  "311@denvergov.org",
  "311@sfgov.org",
  "contact@service-public.fr",
  "help@id.me",
  "info@belastingdienst.nl",
  "info@inps.it",
  "noreply@id.me",
];

const BLOCKED_EXAMPLE_SSNS = new Set(EXAMPLE_SSNS.map((v) => v.replace(/\D/g, "")));
const E1 = new Set<string>([...EXAMPLE_PHONES.map(canonPhone), ...EXAMPLE_IDS.map(canonId)]);
const E2 = new Set<string>(ROLE_MAILBOXES);
E1.delete("");

export function isBlockedExampleSsn(value: string): boolean {
  return BLOCKED_EXAMPLE_SSNS.has(value.replace(/\D/g, ""));
}

export function allowlistHit(d: Detection): "E1" | "E2" | null {
  if (d.label === "SSN" && isBlockedExampleSsn(d.value)) return null;
  const key =
    d.label === "PHONE" || d.label === "PHONE_E164"
      ? canonPhone(d.value)
      : NUMERIC.has(d.label)
        ? d.value.replace(/\D/g, "")
        : canonId(d.value);
  if (key.length < 3) return null;
  if (E1.has(key) || (d.label === "EMAIL" && E1.has(canonId(d.value)))) return "E1";
  if (d.label === "EMAIL" && E2.has(d.value.toLowerCase())) return "E2";
  return null;
}

/** Demote allowlisted detections to soft; evidence records which list matched. */
export function applyAllowlists(dets: Detection[]): Detection[] {
  return dets.map((d) => {
    const hit = allowlistHit(d);
    if (!hit) return d;
    return { ...d, hard: false, evidence: `${d.evidence};allowlist=${hit}` };
  });
}
