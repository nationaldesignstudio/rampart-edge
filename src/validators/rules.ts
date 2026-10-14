import type { Detection, Label, Norm } from "./types.js";
import { toOriginal } from "./normalize.js";
import { KW, hasKw, FORMAT_EXAMPLE } from "./keywords.js";
import {
  luhn,
  abaChecksum,
  abaPrefixOk,
  ssnValid,
  itinValid,
  einValid,
  mbiValid,
  ibanValid,
  IBAN_LEN,
  vinCheckDigit,
  mrzCheck,
  dniValid,
  nirValid,
  steuerIdValid,
  bsnValid,
  codiceFiscaleValid,
  nifPtValid,
  cpfValid,
} from "./checksums.js";
import { isBlockedExampleSsn } from "./allowlist.js";

// ---------------------------------------------------------------------------------------------
// Digit-run candidates. A run is a maximal sequence of digits joined by single space/dot/dash.
// Every contiguous window of groups inside a run becomes a candidate, so "123 45 6789 202 555 0147"
// yields an SSN window and a PHONE window; overlap resolution later picks the longest/hardest.
// ---------------------------------------------------------------------------------------------
export interface Cand {
  start: number;
  end: number; // normalized offsets
  raw: string;
  digits: string;
  groups: number[];
  seps: string; // distinct separators used inside the window (a spaced slash or double space counts as "/" or " ")
  prev: string;
  next: string; // char before / after the window ("" at edges)
  attachedAlpha: boolean; // e.g. "I-485", "W-2", "HUD-52675": window glued to a letter token
  money: boolean; // preceded by $, €, £, US$ ...
  pct: boolean; // followed by %
  fullRun: boolean; // window spans the entire digit run (not a sub-run)
}
// Groups are joined by one space/dot/dash or a double space; SSN_RUN_RE also joins across a slash ("858 / 33 / 6165").
const RUN_RE = /\d(?:(?:[ .-]|  )?\d)*/g;
const SSN_RUN_RE = /\d(?:(?:[ .-]| ?\/ ?|  )?\d)*/g;

export function digitCandidates(n: Norm, runRe = RUN_RE): Cand[] {
  const t = n.text;
  const out: Cand[] = [];
  for (const m of t.matchAll(runRe)) {
    const runStart = m.index!;
    const raw = m[0];
    // split run into groups with their offsets
    const groups = [...raw.matchAll(/\d+/g)].map((g) => ({ s: g.index!, e: g.index! + g[0].length }));
    const maxWin = Math.min(groups.length, 8);
    for (let i = 0; i < groups.length; i++) {
      for (let j = i; j < Math.min(groups.length, i + maxWin); j++) {
        const s = runStart + groups[i].s,
          e = runStart + groups[j].e;
        const win = t.slice(s, e);
        const prev = s > 0 ? t[s - 1] : "";
        const next = e < t.length ? t[e] : "";
        const pre2 = t.slice(Math.max(0, s - 4), s);
        const seps = [...new Set(win.split(/\d+/).filter(Boolean).map((sep) => sep.trim() || " "))].join("");
        out.push({
          start: s,
          end: e,
          raw: win,
          digits: win.replace(/\D/g, ""),
          fullRun: i === 0 && j === groups.length - 1,
          groups: groups.slice(i, j + 1).map((g) => g.e - g.s),
          seps,
          prev,
          next,
          attachedAlpha: /[A-Za-z]-?$/.test(pre2) || /^-?[A-Za-z]/.test(t.slice(e, e + 2)),
          money: /[$€£]\s?$/.test(pre2) || /US\$\s?$/.test(pre2),
          pct: /^\s?%/.test(t.slice(e, e + 2)),
        });
      }
    }
  }
  return out;
}

function det(
  n: Norm,
  label: Label,
  s: number,
  e: number,
  value: string,
  hard: boolean,
  evidence: string,
  rule: string,
): Detection {
  const [start, end] = toOriginal(n, s, e);
  return { label, start, end, value, hard, evidence, rule };
}
const eq = (a: number[], b: number[]) => a.length === b.length && a.every((x, i) => x === b[i]);
const fmt = (g: number[]) => g.join("-");

// ------------------------------------------------------------------ US tax / benefit identifiers
export function detectSsn(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  const slashCands = /\d ?\/ ?\d/.test(n.text) ? digitCandidates(n, SSN_RUN_RE).filter((c) => c.seps === "/") : [];
  for (const c of [...cands, ...slashCands]) {
    if (c.digits.length !== 9 || c.money || c.pct || c.attachedAlpha) continue;
    const formatted = eq(c.groups, [3, 2, 4]) && c.seps.length === 1;
    const contiguous = eq(c.groups, [9]);
    if (!formatted && !contiguous) continue;
    const example = isBlockedExampleSsn(c.digits);
    if (!ssnValid(c.digits) && !example) continue;
    const kw = hasKw(n, KW.SSN, c.start, c.end);
    out.push(
      det(
        n,
        "SSN",
        c.start,
        c.end,
        c.digits,
        formatted || kw,
        `${example ? "q21-example" : "ssa-range-ok"};fmt=${fmt(c.groups)}${kw ? ";kw" : ""}`,
        "ssn",
      ),
    );
  }
  return out;
}
export function detectItin(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  for (const c of cands) {
    if (c.digits.length !== 9 || c.money || c.pct || c.attachedAlpha) continue;
    const formatted = eq(c.groups, [3, 2, 4]) && c.seps.length === 1;
    const contiguous = eq(c.groups, [9]);
    if (!formatted && !contiguous) continue;
    if (!itinValid(c.digits)) continue;
    const kw = hasKw(n, KW.ITIN, c.start, c.end) || hasKw(n, KW.SSN, c.start, c.end);
    out.push(
      det(
        n,
        "ITIN",
        c.start,
        c.end,
        c.digits,
        formatted || kw,
        `irs-itin-range;fmt=${fmt(c.groups)}${kw ? ";kw" : ""}`,
        "itin",
      ),
    );
  }
  return out;
}
export function detectEin(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  for (const c of cands) {
    if (c.digits.length !== 9 || c.money || c.pct || c.attachedAlpha) continue;
    const formatted = eq(c.groups, [2, 7]) && c.seps === "-";
    const kw = hasKw(n, KW.EIN, c.start, c.end);
    if (!formatted && !(eq(c.groups, [9]) && kw)) continue;
    if (!einValid(c.digits)) continue;
    out.push(
      det(
        n,
        "EIN",
        c.start,
        c.end,
        c.digits,
        false,
        `irs-prefix-ok;business-id-often-public;fmt=${fmt(c.groups)}${kw ? ";kw" : ""}`,
        "ein",
      ),
    );
  }
  return out;
}
const MBI_SCAN = new RegExp(
  `(?<![A-Za-z0-9])([1-9][AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y0-9]\\d)[- ]?([AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y0-9]\\d)[- ]?([AC-HJKMNP-RT-Y]{2}\\d{2})(?![A-Za-z0-9])`,
  "gi",
);
export function detectMbi(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(MBI_SCAN)) {
    const v = (m[1] + m[2] + m[3]).toUpperCase();
    if (!mbiValid(v)) continue;
    const upper = m[0] === m[0].toUpperCase();
    const kw = hasKw(n, KW.MBI, m.index!, m.index! + m[0].length);
    out.push(
      det(
        n,
        "MEDICARE_MBI",
        m.index!,
        m.index! + m[0].length,
        v,
        upper || kw,
        `cms-position-classes${upper ? ";upper" : ";mixed-case"}${kw ? ";kw" : ""}`,
        "mbi",
      ),
    );
  }
  return out;
}
const RECEIPT_RE =
  /(?<![A-Za-z0-9])(EAC|WAC|LIN|SRC|NBC|MSC|IOE|YSC|NSC|TSC|VSC|CSC|MCT|EEC)[- ]?(\d{10})(?!\d)/gi;
export function detectUscisReceipt(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(RECEIPT_RE)) {
    const kw = hasKw(n, KW.RECEIPT, m.index!, m.index! + m[0].length);
    out.push(
      det(
        n,
        "USCIS_RECEIPT",
        m.index!,
        m.index! + m[0].length,
        (m[1] + m[2]).toUpperCase(),
        true,
        `uscis-prefix=${m[1].toUpperCase()};10digits${kw ? ";kw" : ""}`,
        "uscis_receipt",
      ),
    );
  }
  return out;
}
const ANUM_RE = /(?<![A-Za-z0-9])A([- ]?)(\d{7,9}|\d{3}[- ]\d{3}[- ]\d{2,3})(?![A-Za-z0-9])/g;
const ANUM_KW_RE =
  /(?<![A-Za-z0-9])(?:A ?#|A-?number|alien (?:registration )?number|USCIS ?#)\s*:?\s*A?[- ]?(\d{7,9}|\d{3}[- ]\d{3}[- ]\d{2,3})(?![A-Za-z0-9])/gi;
export function detectANumber(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(ANUM_RE)) {
    const digits = m[2].replace(/\D/g, "");
    const kw = hasKw(n, KW.ANUMBER, m.index!, m.index! + m[0].length);
    if (m[1] === " " && !kw) continue; // "A 12345678" is only an A-number with context
    const hard = digits.length >= 8 || kw;
    out.push(
      det(
        n,
        "USCIS_A_NUMBER",
        m.index!,
        m.index! + m[0].length,
        "A" + digits,
        hard,
        `A-prefix;digits=${digits.length}${kw ? ";kw" : ""}`,
        "a_number",
      ),
    );
  }
  for (const m of n.text.matchAll(ANUM_KW_RE)) {
    const digits = m[1].replace(/\D/g, "");
    const compactPrefix = /^A ?#/i.test(m[0]);
    const s = compactPrefix ? m.index! : m.index! + m[0].length - m[1].length,
      e = m.index! + m[0].length;
    if (out.some((d) => d.value === "A" + digits)) continue;
    out.push(
      det(
        n,
        "USCIS_A_NUMBER",
        s,
        e,
        "A" + digits,
        true,
        `kw-prefix;digits=${digits.length}`,
        "a_number",
      ),
    );
  }
  return out;
}
/** A format illustration (0000…, 1234…, 9876…) is not anyone's document number. */
function exampleDigits(d: string): boolean {
  return /^(\d)\1+$/.test(d) || "0123456789".includes(d) || "9876543210".includes(d);
}
const PASSPORT_RE = /(?<![A-Za-z0-9])([A-Z]\d{8})(?![A-Za-z0-9])/g;
export function detectPassport(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(PASSPORT_RE)) {
    const s = m.index!,
      e = s + m[0].length;
    const kw = hasKw(n, KW.PASSPORT, s, e);
    const example = exampleDigits(m[1].slice(1)) || hasKw(n, FORMAT_EXAMPLE, s, e, 64, 32);
    out.push(
      det(
        n,
        "PASSPORT",
        s,
        e,
        m[1],
        kw && !example,
        `us-nextgen-1alpha8num${kw ? ";kw" : ""}${example ? ";example" : ""}`,
        "passport",
      ),
    );
  }
  for (const c of cands) {
    if (c.digits.length !== 9 || !eq(c.groups, [9]) || c.money || c.attachedAlpha) continue;
    if (!hasKw(n, KW.PASSPORT, c.start, c.end)) continue;
    const example = exampleDigits(c.digits) || hasKw(n, FORMAT_EXAMPLE, c.start, c.end, 64, 32);
    out.push(
      det(
        n,
        "PASSPORT",
        c.start,
        c.end,
        c.digits,
        !example,
        `us-legacy-9num;kw${example ? ";example" : ""}`,
        "passport",
      ),
    );
  }
  return out;
}
// ICAO 9303 TD3 line 2
const MRZ2_RE =
  /(?<![A-Z0-9<])([A-Z0-9<]{9})(\d)([A-Z<]{3})(\d{6})(\d)([MF<])(\d{6})(\d)([A-Z0-9<]{14})([\d<])(\d)(?![A-Z0-9<])/g;
const MRZ1_RE = /(?<![A-Z0-9<])P[A-Z<][A-Z<]{3}[A-Z<]{39}(?![A-Z0-9<])/g;
export function detectMrz(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(MRZ2_RE)) {
    const ok =
      mrzCheck(m[1], m[2]) &&
      mrzCheck(m[4], m[5]) &&
      mrzCheck(m[7], m[8]) &&
      (m[10] === "<" ? m[9].replace(/</g, "") === "" : mrzCheck(m[9], m[10]));
    const composite = mrzCheck(m[1] + m[2] + m[4] + m[5] + m[7] + m[8] + m[9] + m[10], m[11]);
    if (!ok) continue;
    out.push(
      det(
        n,
        "MRZ",
        m.index!,
        m.index! + m[0].length,
        m[0],
        true,
        `icao9303-td3-line2;checks=ok;composite=${composite ? "ok" : "bad"}`,
        "mrz",
      ),
    );
  }
  for (const m of n.text.matchAll(MRZ1_RE)) {
    if (!m[0].includes("<<")) continue;
    out.push(
      det(
        n,
        "MRZ",
        m.index!,
        m.index! + m[0].length,
        m[0],
        false,
        "icao9303-td3-line1-structural",
        "mrz",
      ),
    );
  }
  return out;
}
// Driver's licenses: 10 most populous states (secondary sources: NTSI / MVR Decoder / Skyhigh DL tables)
const DL_ALPHA: { st: string; re: RegExp }[] = [
  { st: "CA", re: /(?<![A-Za-z0-9])[A-Z]\d{7}(?![A-Za-z0-9])/g },
  {
    st: "FL/MI",
    re: /(?<![A-Za-z0-9])[A-Z](?:\d{12}|[- ]?\d{3}[- ]\d{3}[- ]\d{2}[- ]\d{3}[- ]\d|[- ]\d{3}[- ]\d{3}[- ]\d{3}[- ]\d{3})(?![A-Za-z0-9])/g,
  },
  { st: "IL", re: /(?<![A-Za-z0-9])[A-Z](?:\d{11}|\d{3}[- ]\d{4}[- ]\d{4})(?![A-Za-z0-9])/g },
  { st: "OH", re: /(?<![A-Za-z0-9])[A-Z]{2}\d{6}(?![A-Za-z0-9])/g },
];
export function detectDriversLicense(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  for (const { st, re } of DL_ALPHA) {
    for (const m of n.text.matchAll(re)) {
      const kw = hasKw(n, KW.DL, m.index!, m.index! + m[0].length, 64);
      out.push(
        det(
          n,
          "DRIVERS_LICENSE",
          m.index!,
          m.index! + m[0].length,
          m[0].replace(/[- ]/g, "").toUpperCase(),
          kw,
          `state=${st};alpha-prefixed${kw ? ";kw" : ""}`,
          "drivers_license",
        ),
      );
    }
  }
  // all-digit formats require a DL keyword to be emitted at all (TX 8, PA 8, NY 9, GA 9, NC 12)
  for (const c of cands) {
    if (c.money || c.pct || c.attachedAlpha) continue;
    const L = c.digits.length;
    let st = "";
    if (L === 8 && (eq(c.groups, [8]) || eq(c.groups, [2, 3, 3]))) st = "TX/PA";
    else if (L === 9 && (eq(c.groups, [9]) || eq(c.groups, [3, 3, 3]))) st = "NY/GA";
    else if (L === 12 && eq(c.groups, [12])) st = "NC";
    if (!st) continue;
    if (!hasKw(n, KW.DL, c.start, c.end, 64)) continue;
    out.push(
      det(
        n,
        "DRIVERS_LICENSE",
        c.start,
        c.end,
        c.digits,
        false,
        `state=${st};digits-only;kw`,
        "drivers_license",
      ),
    );
  }
  return out;
}

// ------------------------------------------------------------------ banking / payments
export function detectRouting(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  for (const c of cands) {
    if (c.digits.length !== 9 || !eq(c.groups, [9]) || c.money || c.pct || c.attachedAlpha)
      continue;
    if (!abaPrefixOk(c.digits) || !abaChecksum(c.digits)) continue;
    const kw = hasKw(n, KW.ROUTING, c.start, c.end, 64);
    out.push(
      det(
        n,
        "ROUTING_NUMBER",
        c.start,
        c.end,
        c.digits,
        kw,
        `aba-checksum;prefix-ok${kw ? ";kw" : ""}`,
        "routing",
      ),
    );
  }
  return out;
}
export function detectBankAccount(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  for (const c of cands) {
    const L = c.digits.length;
    if (L < 6 || L > 17 || c.money || c.pct || c.attachedAlpha) continue;
    if (!eq(c.groups, [L])) continue;
    if (!hasKw(n, KW.ACCOUNT, c.start, c.end, 48, 8)) continue;
    out.push(
      det(
        n,
        "BANK_ACCOUNT",
        c.start,
        c.end,
        c.digits,
        false,
        `len=${L};kw;no-checksum`,
        "bank_account",
      ),
    );
  }
  return out;
}
// IIN table from Wikipedia's "Payment card number" article.
const IIN: { brand: string; test: (d: string) => boolean; lens: number[] }[] = [
  { brand: "amex", test: (d) => /^3[47]/.test(d), lens: [15] },
  { brand: "visa", test: (d) => d.startsWith("4"), lens: [13, 16, 19] },
  {
    brand: "mastercard",
    test: (d) =>
      /^5[1-5]/.test(d) || (Number(d.slice(0, 4)) >= 2221 && Number(d.slice(0, 4)) <= 2720),
    lens: [16],
  },
  {
    brand: "discover",
    test: (d) =>
      /^(6011|64[4-9]|65)/.test(d) ||
      (Number(d.slice(0, 6)) >= 622126 && Number(d.slice(0, 6)) <= 622925),
    lens: [16, 17, 18, 19],
  },
  {
    brand: "jcb",
    test: (d) => Number(d.slice(0, 4)) >= 3528 && Number(d.slice(0, 4)) <= 3589,
    lens: [16, 17, 18, 19],
  },
  { brand: "diners", test: (d) => /^(30|36|38|39)/.test(d), lens: [14, 15, 16, 17, 18, 19] },
  { brand: "unionpay", test: (d) => d.startsWith("62"), lens: [16, 17, 18, 19] },
  {
    brand: "maestro",
    test: (d) => /^(5018|5020|5038|5893|6304|6759|676[123])/.test(d),
    lens: [12, 13, 14, 15, 16, 17, 18, 19],
  },
];
export function detectCard(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  for (const c of cands) {
    const L = c.digits.length;
    if (L < 12 || L > 19 || c.money || c.pct || c.attachedAlpha) continue;
    if (/\./.test(c.seps)) continue;
    const grouped = c.groups.length > 1;
    if (grouped && (c.seps.length !== 1 || c.groups.some((g) => g < 3))) continue;
    if (!luhn(c.digits)) continue;
    const iin = IIN.find((b) => b.test(c.digits));
    const brandOk = !!iin && iin.lens.includes(L);
    const kw = hasKw(n, KW.CARD, c.start, c.end, 64);
    if (!c.fullRun && !kw) continue; // a Luhn-valid 16-digit sub-run inside a longer run (e.g. 20-digit tracking no.) is not a card
    if (!brandOk && !grouped && !kw) continue; // luhn-only contiguous digits with no context: too weak to even be a feature
    out.push(
      det(
        n,
        "CREDIT_CARD",
        c.start,
        c.end,
        c.digits,
        brandOk,
        `luhn;iin=${brandOk ? iin!.brand : "unknown"};len=${L};fmt=${fmt(c.groups)}${kw ? ";kw" : ""}`,
        "card",
      ),
    );
  }
  return out;
}
const IBAN_RE =
  /(?<![A-Za-z0-9])([A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?)(?![A-Za-z0-9])/g;
export function detectIban(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(IBAN_RE)) {
    let raw = m[1];
    let v = raw.replace(/ /g, "");
    if (!ibanValid(v)) {
      // the optional trailing 1-4 char group may have swallowed a following word ("... 7880 for"): retry without it
      const cut = raw.lastIndexOf(" ");
      if (cut < 0) continue;
      raw = raw.slice(0, cut);
      v = raw.replace(/ /g, "");
      if (!ibanValid(v)) continue;
    }
    out.push(
      det(
        n,
        "IBAN",
        m.index!,
        m.index! + raw.length,
        v,
        true,
        `mod97;country=${v.slice(0, 2)};len=${v.length}/${IBAN_LEN[v.slice(0, 2)]}`,
        "iban",
      ),
    );
  }
  return out;
}
const ISO_CC = new Set(
  Object.keys(IBAN_LEN).concat([
    "US",
    "CA",
    "MX",
    "AU",
    "NZ",
    "JP",
    "CN",
    "HK",
    "SG",
    "IN",
    "ZA",
    "AR",
    "CL",
    "CO",
    "PE",
    "KR",
    "TW",
    "TH",
    "MY",
    "ID",
    "PH",
    "VN",
    "RU",
    "NG",
    "KE",
    "MA",
    "DZ",
    "PA",
  ]),
);
const BIC_RE = /(?<![A-Za-z0-9])([A-Z]{4})([A-Z]{2})([A-Z0-9]{2})([A-Z0-9]{3})?(?![A-Za-z0-9])/g;
export function detectBic(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(BIC_RE)) {
    if (!ISO_CC.has(m[2])) continue;
    const kw = hasKw(n, KW.BIC, m.index!, m.index! + m[0].length);
    if (/^[A-Z]+$/.test(m[0]) && !m[4] && !kw) continue; // 8 plain letters without BIC/SWIFT context = almost always a word
    if (!kw && /^[A-Z]+$/.test(m[0])) continue;
    out.push(
      det(
        n,
        "BIC",
        m.index!,
        m.index! + m[0].length,
        m[0],
        false,
        `iso9362-structure;cc=${m[2]}${kw ? ";kw" : ""};no-checksum`,
        "bic",
      ),
    );
  }
  return out;
}

// ------------------------------------------------------------------ phone
const TOLL_FREE = new Set(["800", "833", "844", "855", "866", "877", "888"]);
function nanpOk(d10: string): string | null {
  const npa = d10.slice(0, 3),
    nxx = d10.slice(3, 6);
  if (!/^[2-9]/.test(npa) || /^[2-9]11$/.test(npa)) return "bad-npa";
  if (!/^[2-9]/.test(nxx) || /^[2-9]11$/.test(nxx)) return "bad-nxx";
  if (npa[1] === "9") return "bad-npa"; // N9X reserved for expansion
  return null;
}
const PAREN_PHONE_RE =
  /(?:\+?1[ .-]?)?\(\s?([2-9]\d{2})\s?\)[ .-]?([2-9]\d{2})[ .-]?(\d{4})(?!\d)/g;
export function detectPhone(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  const push = (s: number, e: number, d10: string, formatted: boolean, extra: string) => {
    const bad = nanpOk(d10);
    if (bad) return;
    const toll = TOLL_FREE.has(d10.slice(0, 3));
    const fictional = d10.slice(3, 6) === "555";
    const kw = hasKw(n, KW.PHONE, s, e, 40);
    const personal = hasKw(n, KW.PHONE_PERSONAL, s, e, 64, 96);
    const pub = !personal && hasKw(n, KW.PUBLIC_ENTITY, s, e, 96, 48);
    const hard = !fictional && (personal || formatted || kw); // every complete number blocks; allowlisted numbers are demoted by the gate
    out.push(
      det(
        n,
        "PHONE",
        s,
        e,
        d10,
        hard,
        `nanp-npa-nxx-ok;${extra}${toll ? ";toll-free" : ""}${fictional ? ";555" : ""}${kw ? ";kw" : ""}${personal ? ";personal" : ""}${pub ? ";public-entity" : ""}`,
        "phone",
      ),
    );
  };
  for (const m of n.text.matchAll(PAREN_PHONE_RE))
    push(m.index!, m.index! + m[0].length, m[1] + m[2] + m[3], true, "fmt=(3)3-4");
  for (const c of cands) {
    if (c.money || c.pct || c.attachedAlpha) continue;
    let d = c.digits,
      g = c.groups;
    if (d.length === 11 && d[0] === "1" && g[0] === 1) {
      d = d.slice(1);
      g = g.slice(1);
    } else if (d.length === 11 && d[0] === "1" && eq(g, [11])) {
      d = d.slice(1);
      g = [10];
    }
    if (d.length !== 10) continue;
    const formatted = (eq(g, [3, 3, 4]) || eq(g, [3, 7]) || eq(g, [6, 4])) && c.seps.length === 1;
    const contiguous = eq(g, [10]);
    if (!formatted && !contiguous) continue;
    push(c.start, c.end, d, formatted, `fmt=${fmt(c.groups)}`);
  }
  return out;
}
const E164_RE = /(?<![\w+])\+(\d[\d \-().]{5,22}\d)(?![\d\w])/g;
export function detectE164(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(E164_RE)) {
    const d = m[1].replace(/\D/g, "");
    if (d.length < 8 || d.length > 15) continue;
    let hard = true,
      extra = "";
    if (d[0] === "1" && d.length === 11) {
      const bad = nanpOk(d.slice(1));
      if (bad) continue;
      if (TOLL_FREE.has(d.slice(1, 4))) extra = ";toll-free"; // still hard
    }
    out.push(
      det(
        n,
        "PHONE_E164",
        m.index!,
        m.index! + m[0].length,
        "+" + d,
        hard,
        `e164;digits=${d.length}${extra}`,
        "e164",
      ),
    );
  }
  return out;
}

// ------------------------------------------------------------------ network / contact
const EMAIL_RE =
  /(?<![A-Za-z0-9._%+-])([A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@((?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24})(?![A-Za-z0-9-])/g;
// Role mailboxes on any domain and every .gov/.mil address are deterministic ALLOW; other named mailboxes block.
const ROLE_LOCAL =
  /^(info|hello|hi|help|support|contact|contactus|press|media|news|foia|webmaster|postmaster|hostmaster|noreply|no-reply|donotreply|do-not-reply|admin|office|service|services|customerservice|customer-service|privacy|accessibility|feedback|comments|questions|public|publicaffairs|inquiries|enquiries|sales|billing|orders|careers|jobs|hr|recruiting|legal|security|abuse|team|marketing|newsletter|events|editor|tips|outreach|communications|appointments|reservations|scheduling|intake|volunteer|ombudsman|complaints|hola|contacto|prensa|ayuda|soporte|kontakt|presse)$/i;
export const isGovEmail = (d: Detection): boolean => d.label === "EMAIL" && /\.(gov|mil)$/i.test(d.value);
export function detectEmail(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(EMAIL_RE)) {
    const local = m[1],
      domain = m[2].toLowerCase();
    const gov = /\.(gov|mil)$/.test(domain);
    const role = ROLE_LOCAL.test(local);
    out.push(
      det(
        n,
        "EMAIL",
        m.index!,
        m.index! + m[0].length,
        m[0],
        !role && !gov,
        `rfc5322-practical${gov ? ";gov-domain" : ""}${role ? ";role-mailbox" : ""}`,
        "email",
      ),
    );
  }
  return out;
}
const URL_RE =
  /(?<![A-Za-z0-9@.])((?:https?:\/\/|www\.)[^\s<>"'`]+|(?:[a-z0-9-]+\.)+(?:gov|com|org|net|edu|mil|int|us|es|fr|de|it|pt|nl|eu|io|uk|ca|mx|br|be|ch|at|gl|ly|me|co|app|dev|ai|info)(?:\/[^\s<>"'`]*)?)(?![A-Za-z0-9])/gi;
export function detectUrl(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(URL_RE)) {
    let v = m[1].replace(/[.,;:!?)\]]+$/, "");
    if (
      !/^(https?:|www\.)/i.test(v) &&
      !/\//.test(v) &&
      /@/.test(n.text.slice(Math.max(0, m.index! - 1), m.index!))
    )
      continue;
    const hasQuery = /[?#]/.test(v) || /\/[^/]{12,}/.test(v.replace(/^https?:\/\/[^/]+/, ""));
    out.push(
      det(
        n,
        "URL",
        m.index!,
        m.index! + v.length,
        v,
        false,
        `url;scheme=${/^https?:/i.test(v) ? "yes" : "no"}${hasQuery ? ";path-or-query" : ""}`,
        "url",
      ),
    );
  }
  return out;
}
const IPV4_RE =
  /(?<![\d.])((?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})(?![\d.]\d)/g;
function ipv4Scope(ip: string): string {
  const [a, b, c] = ip.split(".").map(Number);
  if (a === 10 || a === 127 || a === 0) return "private";
  if (a === 172 && b >= 16 && b <= 31) return "private";
  if (a === 192 && b === 168) return "private";
  if (a === 169 && b === 254) return "link-local";
  if (a === 100 && b >= 64 && b <= 127) return "shared";
  if (a >= 224) return "multicast/reserved";
  if (
    (a === 192 && b === 0 && c === 2) ||
    (a === 198 && b === 51 && c === 100) ||
    (a === 203 && b === 0 && c === 113)
  )
    return "documentation";
  return "public";
}
export function detectIpv4(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(IPV4_RE)) {
    const scope = ipv4Scope(m[1]);
    const kw = hasKw(n, KW.IP, m.index!, m.index! + m[0].length);
    const octets = m[1].split(".").map(Number);
    const sectionLike = octets[0] < 10 && octets.every((o) => o < 100) && !kw; // "5.3.1.46 – STAR+PLUS FSR Instructions"
    out.push(
      det(
        n,
        "IPV4",
        m.index!,
        m.index! + m[0].length,
        m[1],
        scope === "public" && !sectionLike,
        `ipv4;scope=${scope}${kw ? ";kw" : ""}${sectionLike ? ";section-like" : ""}`,
        "ipv4",
      ),
    );
  }
  return out;
}
const IPV6_RE =
  /(?<![A-Za-z0-9:.])((?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}|(?:[0-9a-f]{1,4}:){1,7}:|(?:[0-9a-f]{1,4}:){1,6}:[0-9a-f]{1,4}|(?:[0-9a-f]{1,4}:){1,5}(?::[0-9a-f]{1,4}){1,2}|(?:[0-9a-f]{1,4}:){1,4}(?::[0-9a-f]{1,4}){1,3}|(?:[0-9a-f]{1,4}:){1,3}(?::[0-9a-f]{1,4}){1,4}|(?:[0-9a-f]{1,4}:){1,2}(?::[0-9a-f]{1,4}){1,5}|[0-9a-f]{1,4}:(?::[0-9a-f]{1,4}){1,6}|:(?::[0-9a-f]{1,4}){1,7}|::1)(?![A-Za-z0-9:])/gi;
export function detectIpv6(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(IPV6_RE)) {
    const v = m[1].toLowerCase();
    const colons = (v.match(/:/g) ?? []).length;
    if (colons < 2) continue;
    if (!/[0-9a-f]/.test(v)) continue;
    if (/^(?:\d{1,2}:){2}\d{1,2}$/.test(v)) continue; // hh:mm:ss
    const special =
      v === "::1" ||
      v.startsWith("fe80:") ||
      v.startsWith("2001:db8") ||
      v.startsWith("fc") ||
      v.startsWith("fd");
    out.push(
      det(
        n,
        "IPV6",
        m.index!,
        m.index! + m[0].length,
        v,
        !special,
        `ipv6;colons=${colons}${special ? ";special-scope" : ""}`,
        "ipv6",
      ),
    );
  }
  return out;
}
const MAC_RE = /(?<![A-Za-z0-9:-])([0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5})(?![A-Za-z0-9:-])/g;
export function detectMac(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(MAC_RE)) {
    out.push(
      det(
        n,
        "MAC",
        m.index!,
        m.index! + m[0].length,
        m[1].toUpperCase(),
        false,
        "mac-48;6-octets;q34-device-id",
        "mac",
      ),
    );
  }
  return out;
}

// ------------------------------------------------------------------ postal / dates / misc
export function detectZip(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  for (const c of cands) {
    if (c.money || c.pct || c.attachedAlpha) continue;
    const plus4 = eq(c.groups, [5, 4]) && c.seps === "-";
    if (!plus4 && !eq(c.groups, [5])) continue;
    const kw = hasKw(n, KW.ZIP, c.start, c.end, 40);
    out.push(
      det(
        n,
        "ZIP",
        c.start,
        c.end,
        c.digits,
        false,
        `${plus4 ? "zip+4" : "zip5"}${kw ? ";kw" : ""}`,
        "zip",
      ),
    );
  }
  return out;
}
const MONTHS =
  "january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre|januar|februar|märz|maerz|juni|juli|oktober|dezember|gennaio|febbraio|aprile|maggio|giugno|luglio|settembre|ottobre|dicembre|janeiro|fevereiro|março|marco|maio|junho|julho|setembro|outubro|dezembro|januari|februari|maart|mei|augustus|december|novembro";
const DATE_NUM_RE = /(?<![\d/.-])(\d{1,2})([/.-])(\d{1,2})\2(\d{4}|\d{2})(?!\d|[/.-]\d)/g;
const DATE_ISO_RE = /(?<![\d/.-])(\d{4})-(\d{2})-(\d{2})(?!\d|[/.:T-]\d)/g;
const DATE_TXT1 = new RegExp(
  `(?<![A-Za-z0-9])(\\d{1,2})(?:st|nd|rd|th|\\.|º|°)?\\s+(?:de\\s+|of\\s+)?(${MONTHS})\\.?,?\\s+(?:de\\s+|du\\s+)?(\\d{4})(?!\\d)`,
  "gi",
);
const DATE_TXT2 = new RegExp(
  `(?<![A-Za-z0-9])(${MONTHS})\\.?\\s+(\\d{1,2})(?:st|nd|rd|th)?,?\\s+(\\d{4})(?!\\d)`,
  "gi",
);
const YEAR_NOW = 2026;
export function detectDate(n: Norm): Detection[] {
  const out: Detection[] = [];
  const push = (s: number, e: number, y: number, a: number, b: number, kind: string) => {
    const dayMonthOk =
      (a >= 1 && a <= 12 && b >= 1 && b <= 31) || (b >= 1 && b <= 12 && a >= 1 && a <= 31);
    if (!dayMonthOk) return;
    if (y < 100) y += y > YEAR_NOW % 100 ? 1900 : 2000;
    if (y < 1850 || y > YEAR_NOW + 15) return;
    const kw = hasKw(n, KW.DOB, s, e, 36, 16);
    const birthPlausible = y >= YEAR_NOW - 110 && y <= YEAR_NOW;
    const isDob = kw && birthPlausible;
    // A bare DOB is ALLOW — soft feature only, never a hard block.
    out.push(
      det(
        n,
        isDob ? "DOB" : "DATE",
        s,
        e,
        `${y}-${String(a).padStart(2, "0")}-${String(b).padStart(2, "0")}`,
        false,
        `${kind};year=${y}${kw ? ";kw" : ""}`,
        "date",
      ),
    );
  };
  for (const m of n.text.matchAll(DATE_NUM_RE))
    push(
      m.index!,
      m.index! + m[0].length,
      Number(m[4]),
      Number(m[1]),
      Number(m[3]),
      `numeric-${m[2] === "/" ? "slash" : m[2] === "." ? "dot" : "dash"}`,
    );
  for (const m of n.text.matchAll(DATE_ISO_RE))
    push(m.index!, m.index! + m[0].length, Number(m[1]), Number(m[2]), Number(m[3]), "iso8601");
  for (const m of n.text.matchAll(DATE_TXT1))
    push(m.index!, m.index! + m[0].length, Number(m[3]), Number(m[1]), 1, "day-month-year-text");
  for (const m of n.text.matchAll(DATE_TXT2))
    push(m.index!, m.index! + m[0].length, Number(m[3]), Number(m[2]), 1, "month-day-year-text");
  return out;
}
const VIN_RE = /(?<![A-Za-z0-9])([A-HJ-NPR-Z0-9]{17})(?![A-Za-z0-9])/gi;
export function detectVin(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(VIN_RE)) {
    const v = m[1].toUpperCase();
    if (!/[A-Z]/.test(v) || !/\d/.test(v)) continue;
    const ok = vinCheckDigit(v);
    const kw = hasKw(n, KW.VIN, m.index!, m.index! + m[0].length);
    if (!ok && !kw) continue;
    // Vehicles are ALLOW — soft feature only.
    out.push(
      det(
        n,
        "VIN",
        m.index!,
        m.index! + m[0].length,
        v,
        false,
        `17char;check-digit=${ok ? "ok" : "bad"}${kw ? ";kw" : ""}`,
        "vin",
      ),
    );
  }
  return out;
}
const PLATE_RE =
  /(?<![A-Za-z0-9])([A-Z]{1,2}-\d{3}-[A-Z]{2}|[A-Z0-9]{2,4}[- ]?[A-Z0-9]{2,4})(?![A-Za-z0-9])/g;
export function detectLicensePlate(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(PLATE_RE)) {
    const v = m[1];
    if (!/[A-Z]/.test(v) || !/\d/.test(v)) continue;
    if (!hasKw(n, KW.PLATE, m.index!, m.index! + v.length, 40, 8)) continue;
    out.push(
      det(
        n,
        "LICENSE_PLATE",
        m.index!,
        m.index! + v.length,
        v.replace(/[- ]/g, ""),
        false,
        "plate-shape;kw;low-precision",
        "license_plate",
      ),
    );
  }
  return out;
}
const COORD_DEC_RE =
  /(?<![\d.-])(-?\d{1,2}\.\d{3,})°?\s*([NS])?\s*,\s*(-?\d{1,3}\.\d{3,})°?\s*([EW])?(?!\d|\.\d)/g;
const COORD_DMS_RE =
  /(\d{1,3})°\s?(\d{1,2})['′]\s?(\d{1,2}(?:\.\d+)?)?["″]?\s?([NSEW])[,\s]+(\d{1,3})°\s?(\d{1,2})['′]\s?(\d{1,2}(?:\.\d+)?)?["″]?\s?([NSEW])/g;
export function detectCoordinates(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(COORD_DEC_RE)) {
    const lat = Number(m[1]),
      lon = Number(m[3]);
    if (Math.abs(lat) > 90 || Math.abs(lon) > 180) continue;
    const prec = Math.min(m[1].split(".")[1].length, m[3].split(".")[1].length);
    const kw = hasKw(n, KW.COORD, m.index!, m.index! + m[0].length);
    out.push(
      det(
        n,
        "COORDINATES",
        m.index!,
        m.index! + m[0].length,
        `${lat},${lon}`,
        false,
        `decimal;precision=${prec}dp${kw ? ";kw" : ""}`,
        "coordinates",
      ),
    );
  }
  for (const m of n.text.matchAll(COORD_DMS_RE)) {
    out.push(
      det(
        n,
        "COORDINATES",
        m.index!,
        m.index! + m[0].length,
        m[0].replace(/\s+/g, ""),
        false,
        "dms",
        "coordinates",
      ),
    );
  }
  return out;
}

// ------------------------------------------------------------------ EU national identifiers
const DNI_RE =
  /(?<![A-Za-z0-9])((?:\d{8}|\d{2}\.\d{3}\.\d{3}|[XYZ][- ]?\d{7})[- ]?[A-Z])(?![A-Za-z0-9])/gi;
// Non-US national identifiers are out of scope — detected as soft features only, never a hard block.
export function detectEsDni(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(DNI_RE)) {
    const v = m[1].replace(/[-. ]/g, "").toUpperCase();
    if (!dniValid(v)) continue;
    const kw = hasKw(n, KW.DNI, m.index!, m.index! + m[0].length);
    out.push(
      det(
        n,
        "ES_DNI",
        m.index!,
        m.index! + m[0].length,
        v,
        false,
        `mod23-letter-ok;${/^[XYZ]/.test(v) ? "nie" : "dni"}${kw ? ";kw" : ""}`,
        "es_dni",
      ),
    );
  }
  return out;
}
const NIR_RE =
  /(?<![A-Za-z0-9])([1-478][ ]?\d{2}[ ]?\d{2}[ ]?\d[0-9AB][ ]?\d{3}[ ]?\d{3}[ ]?\d{2})(?![A-Za-z0-9])/gi;
export function detectFrNir(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(NIR_RE)) {
    const v = m[1].replace(/ /g, "").toUpperCase();
    if (!nirValid(v)) continue;
    const kw = hasKw(n, KW.NIR, m.index!, m.index! + m[0].length);
    out.push(
      det(
        n,
        "FR_NIR",
        m.index!,
        m.index! + m[0].length,
        v,
        false,
        `mod97-key-ok;15chars${kw ? ";kw" : ""}`,
        "fr_nir",
      ),
    );
  }
  return out;
}
export function detectDeSteuerId(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  for (const c of cands) {
    if (c.digits.length !== 11 || c.money || c.pct || c.attachedAlpha) continue;
    const formatted = eq(c.groups, [2, 3, 3, 3]) && c.seps === " ";
    if (!formatted && !eq(c.groups, [11])) continue;
    if (!steuerIdValid(c.digits)) continue;
    const kw = hasKw(n, KW.STEUER, c.start, c.end);
    out.push(
      det(
        n,
        "DE_STEUER_ID",
        c.start,
        c.end,
        c.digits,
        false,
        `iso7064-mod11-10-ok;digit-repeat-rule-ok;fmt=${fmt(c.groups)}${kw ? ";kw" : ""}`,
        "de_steuer_id",
      ),
    );
  }
  return out;
}
const CF_RE =
  /(?<![A-Za-z0-9])([A-Z]{6}[0-9LMNPQRSTUV]{2}[ABCDEHLMPRST][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z])(?![A-Za-z0-9])/gi;
export function detectItCodiceFiscale(n: Norm): Detection[] {
  const out: Detection[] = [];
  for (const m of n.text.matchAll(CF_RE)) {
    const v = m[1].toUpperCase();
    if (!codiceFiscaleValid(v)) continue;
    const kw = hasKw(n, KW.CF, m.index!, m.index! + m[0].length);
    out.push(
      det(
        n,
        "IT_CODICE_FISCALE",
        m.index!,
        m.index! + m[0].length,
        v,
        false,
        `mod26-check-ok;16chars${kw ? ";kw" : ""}`,
        "it_cf",
      ),
    );
  }
  return out;
}
export function detectPtNif(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  for (const c of cands) {
    if (c.digits.length !== 9 || c.money || c.pct || c.attachedAlpha) continue;
    const formatted = eq(c.groups, [3, 3, 3]) && c.seps === " ";
    if (!formatted && !eq(c.groups, [9])) continue;
    if (!nifPtValid(c.digits)) continue;
    const kw = hasKw(n, KW.NIF_PT, c.start, c.end);
    out.push(
      det(
        n,
        "PT_NIF",
        c.start,
        c.end,
        c.digits,
        false,
        `mod11-ok;prefix-ok;fmt=${fmt(c.groups)}${kw ? ";kw" : ""}`,
        "pt_nif",
      ),
    );
  }
  return out;
}
export function detectNlBsn(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  for (const c of cands) {
    if ((c.digits.length !== 9 && c.digits.length !== 8) || c.money || c.pct || c.attachedAlpha)
      continue;
    if (!eq(c.groups, [c.digits.length])) continue;
    if (!bsnValid(c.digits)) continue;
    const kw = hasKw(n, KW.BSN, c.start, c.end);
    out.push(
      det(
        n,
        "NL_BSN",
        c.start,
        c.end,
        c.digits.padStart(9, "0"),
        false,
        `elfproef-ok;len=${c.digits.length}${kw ? ";kw" : ""}`,
        "nl_bsn",
      ),
    );
  }
  return out;
}
export function detectBrCpf(n: Norm, cands = digitCandidates(n)): Detection[] {
  const out: Detection[] = [];
  for (const c of cands) {
    if (c.digits.length !== 11 || c.money || c.pct || c.attachedAlpha) continue;
    const formatted =
      eq(c.groups, [3, 3, 3, 2]) && (c.seps === ".-" || c.seps === "-." || c.seps === ".");
    if (!formatted && !eq(c.groups, [11])) continue;
    if (!cpfValid(c.digits)) continue;
    const kw = hasKw(n, KW.CPF, c.start, c.end);
    out.push(
      det(
        n,
        "BR_CPF",
        c.start,
        c.end,
        c.digits,
        false,
        `mod11x2-ok;fmt=${fmt(c.groups)}${kw ? ";kw" : ""}`,
        "br_cpf",
      ),
    );
  }
  return out;
}

export const ALL_RULES: { id: string; fn: (n: Norm, cands: Cand[]) => Detection[] }[] = [
  { id: "ssn", fn: detectSsn },
  { id: "itin", fn: detectItin },
  { id: "ein", fn: detectEin },
  { id: "mbi", fn: detectMbi },
  { id: "a_number", fn: detectANumber },
  { id: "uscis_receipt", fn: detectUscisReceipt },
  { id: "passport", fn: detectPassport },
  { id: "mrz", fn: detectMrz },
  { id: "drivers_license", fn: detectDriversLicense },
  { id: "routing", fn: detectRouting },
  { id: "bank_account", fn: detectBankAccount },
  { id: "card", fn: detectCard },
  { id: "iban", fn: detectIban },
  { id: "bic", fn: detectBic },
  { id: "phone", fn: detectPhone },
  { id: "e164", fn: detectE164 },
  { id: "email", fn: detectEmail },
  { id: "url", fn: detectUrl },
  { id: "ipv4", fn: detectIpv4 },
  { id: "ipv6", fn: detectIpv6 },
  { id: "mac", fn: detectMac },
  { id: "zip", fn: detectZip },
  { id: "date", fn: detectDate },
  { id: "vin", fn: detectVin },
  { id: "license_plate", fn: detectLicensePlate },
  { id: "coordinates", fn: detectCoordinates },
  { id: "es_dni", fn: detectEsDni },
  { id: "fr_nir", fn: detectFrNir },
  { id: "de_steuer_id", fn: detectDeSteuerId },
  { id: "it_cf", fn: detectItCodiceFiscale },
  { id: "pt_nif", fn: detectPtNif },
  { id: "nl_bsn", fn: detectNlBsn },
  { id: "br_cpf", fn: detectBrCpf },
];
