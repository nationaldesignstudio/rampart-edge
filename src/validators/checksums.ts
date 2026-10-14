// Each algorithm below follows the published standard named in its comment.

/** Luhn / ISO 7812 mod-10 (payment cards). */
export function luhn(digits: string): boolean {
  if (!/^\d{2,}$/.test(digits)) return false;
  let sum = 0,
    dbl = false;
  for (let i = digits.length - 1; i >= 0; i--) {
    let d = digits.charCodeAt(i) - 48;
    if (dbl) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
    dbl = !dbl;
  }
  return sum % 10 === 0;
}

/** ABA routing transit number: (3(d1+d4+d7) + 7(d2+d5+d8) + (d3+d6+d9)) mod 10 == 0. */
export function abaChecksum(d: string): boolean {
  if (!/^\d{9}$/.test(d)) return false;
  const n = (i: number) => d.charCodeAt(i) - 48;
  return (3 * (n(0) + n(3) + n(6)) + 7 * (n(1) + n(4) + n(7)) + (n(2) + n(5) + n(8))) % 10 === 0;
}
/** ABA first-two-digit ranges actually assigned: 00-12, 21-32, 61-72, 80. */
export function abaPrefixOk(d: string): boolean {
  const p = Number(d.slice(0, 2));
  return (p >= 0 && p <= 12) || (p >= 21 && p <= 32) || (p >= 61 && p <= 72) || p === 80;
}

/** SSA post-randomization validity: area != 000, 666, 900-999; group != 00; serial != 0000. */
export function ssnValid(d: string): boolean {
  if (!/^\d{9}$/.test(d)) return false;
  const area = d.slice(0, 3),
    group = d.slice(3, 5),
    serial = d.slice(5);
  if (area === "000" || area === "666" || area[0] === "9") return false;
  if (group === "00" || serial === "0000") return false;
  return true;
}
/** IRS ITIN: 9XX-XX-XXXX with 4th-5th digits in 50-65, 70-88, 90-92, 94-99. */
export function itinValid(d: string): boolean {
  if (!/^9\d{8}$/.test(d)) return false;
  const g = Number(d.slice(3, 5));
  return (
    (g >= 50 && g <= 65) || (g >= 70 && g <= 88) || (g >= 90 && g <= 92) || (g >= 94 && g <= 99)
  );
}
/** IRS valid EIN campus/online prefixes (irs.gov "How EINs are assigned and valid EIN prefixes"). */
export const EIN_PREFIXES = new Set([
  "10",
  "12",
  "60",
  "67",
  "50",
  "53",
  "01",
  "02",
  "03",
  "04",
  "05",
  "06",
  "11",
  "13",
  "14",
  "16",
  "21",
  "22",
  "23",
  "25",
  "34",
  "51",
  "52",
  "54",
  "55",
  "56",
  "57",
  "58",
  "59",
  "65",
  "30",
  "32",
  "35",
  "36",
  "37",
  "38",
  "61",
  "15",
  "24",
  "40",
  "44",
  "94",
  "95",
  "80",
  "90",
  "33",
  "39",
  "41",
  "42",
  "43",
  "46",
  "48",
  "62",
  "63",
  "64",
  "66",
  "68",
  "71",
  "72",
  "73",
  "74",
  "75",
  "76",
  "77",
  "85",
  "86",
  "87",
  "88",
  "91",
  "92",
  "93",
  "98",
  "99",
  "20",
  "26",
  "27",
  "45",
  "47",
  "81",
  "82",
  "83",
  "84",
]);
export function einValid(d: string): boolean {
  return /^\d{9}$/.test(d) && EIN_PREFIXES.has(d.slice(0, 2));
}

/** CMS Medicare Beneficiary Identifier: 11 chars, position classes; letters exclude S L O I B Z. */
const MBI_A = "[AC-HJKMNP-RT-Y]"; // A-Z minus S,L,O,I,B,Z
const MBI_AN = `(?:\\d|${MBI_A})`;
export const MBI_RE_SRC = `[1-9]${MBI_A}${MBI_AN}\\d${MBI_A}${MBI_AN}\\d${MBI_A}${MBI_A}\\d\\d`;
const MBI_RE = new RegExp(`^${MBI_RE_SRC}$`);
export function mbiValid(s: string): boolean {
  return MBI_RE.test(s);
}

/** ISO 13616 IBAN mod-97-10. Country lengths from the SWIFT IBAN registry. */
export const IBAN_LEN: Record<string, number> = {
  AL: 28,
  AD: 24,
  AT: 20,
  AZ: 28,
  BH: 22,
  BY: 28,
  BE: 16,
  BA: 20,
  BR: 29,
  BG: 22,
  CR: 22,
  HR: 21,
  CY: 28,
  CZ: 24,
  DK: 18,
  DO: 28,
  TL: 23,
  EG: 29,
  SV: 28,
  EE: 20,
  FO: 18,
  FI: 18,
  FR: 27,
  GE: 22,
  DE: 22,
  GI: 23,
  GR: 27,
  GL: 18,
  GT: 28,
  VA: 22,
  HU: 28,
  IS: 26,
  IQ: 23,
  IE: 22,
  IL: 23,
  IT: 27,
  JO: 30,
  KZ: 20,
  XK: 20,
  KW: 30,
  LV: 21,
  LB: 28,
  LY: 25,
  LI: 21,
  LT: 20,
  LU: 20,
  MT: 31,
  MR: 27,
  MU: 30,
  MD: 24,
  MC: 27,
  ME: 22,
  NL: 18,
  MK: 19,
  NO: 15,
  PK: 24,
  PS: 29,
  PL: 28,
  PT: 25,
  QA: 29,
  RO: 24,
  LC: 32,
  SM: 27,
  ST: 25,
  SA: 24,
  RS: 22,
  SC: 31,
  SK: 24,
  SI: 19,
  ES: 24,
  SD: 18,
  SE: 24,
  CH: 21,
  TN: 24,
  TR: 26,
  UA: 29,
  AE: 23,
  GB: 22,
  VG: 24,
};
export function mod97(s: string): number {
  let rem = 0;
  for (const ch of s) {
    const v = ch >= "A" && ch <= "Z" ? String(ch.charCodeAt(0) - 55) : ch;
    for (const c of v) rem = (rem * 10 + (c.charCodeAt(0) - 48)) % 97;
  }
  return rem;
}
export function ibanValid(s: string): boolean {
  if (!/^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$/.test(s)) return false;
  const len = IBAN_LEN[s.slice(0, 2)];
  if (!len || len !== s.length) return false;
  return mod97(s.slice(4) + s.slice(0, 4)) === 1;
}

/** 49 CFR 565.15(c): VIN check digit (position 9). */
const VIN_VAL: Record<string, number> = {
  A: 1,
  B: 2,
  C: 3,
  D: 4,
  E: 5,
  F: 6,
  G: 7,
  H: 8,
  J: 1,
  K: 2,
  L: 3,
  M: 4,
  N: 5,
  P: 7,
  R: 9,
  S: 2,
  T: 3,
  U: 4,
  V: 5,
  W: 6,
  X: 7,
  Y: 8,
  Z: 9,
};
const VIN_W = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2];
export function vinCheckDigit(v: string): boolean {
  if (!/^[A-HJ-NPR-Z0-9]{17}$/.test(v)) return false;
  let sum = 0;
  for (let i = 0; i < 17; i++) {
    const c = v[i];
    const val = c >= "0" && c <= "9" ? c.charCodeAt(0) - 48 : VIN_VAL[c];
    sum += val * VIN_W[i];
  }
  const r = sum % 11;
  return v[8] === (r === 10 ? "X" : String(r));
}

/** ICAO 9303 part 3 check digit: weights 7,3,1 repeating; '<' = 0; A=10..Z=35. */
export function mrzCheck(field: string, check: string): boolean {
  const w = [7, 3, 1];
  let sum = 0;
  for (let i = 0; i < field.length; i++) {
    const c = field[i];
    const v = c === "<" ? 0 : c >= "0" && c <= "9" ? c.charCodeAt(0) - 48 : c.charCodeAt(0) - 55;
    sum += v * w[i % 3];
  }
  return String(sum % 10) === check;
}

/** Spain DNI/NIE: letter = "TRWAGMYFPDXBNJZSQVHLCKE"[number mod 23]; NIE X/Y/Z -> 0/1/2. */
const DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE";
export function dniValid(s: string): boolean {
  const m = /^([XYZ]?)(\d{7,8})([A-Z])$/.exec(s);
  if (!m) return false;
  let num = m[2];
  if (m[1]) {
    if (num.length !== 7) return false;
    num = { X: "0", Y: "1", Z: "2" }[m[1]] + num;
  } else if (num.length !== 8) return false;
  return DNI_LETTERS[Number(num) % 23] === m[3];
}

/** France NIR (INSEE): 13 digits + 2-digit key, key = 97 - (n mod 97); Corsica 2A->19, 2B->18. */
export function nirValid(s: string): boolean {
  const m = /^([1-478])(\d{2})(\d{2})(\d[0-9AB])(\d{3})(\d{3})(\d{2})$/.exec(s);
  if (!m) return false;
  const month = Number(m[3]);
  if (
    !(
      (month >= 1 && month <= 12) ||
      month === 20 ||
      (month >= 30 && month <= 42) ||
      (month >= 50 && month <= 99)
    )
  )
    return false;
  let body = s.slice(0, 13);
  if (m[4] === "2A") body = body.replace("2A", "19");
  else if (m[4] === "2B") body = body.replace("2B", "18");
  if (!/^\d{13}$/.test(body)) return false;
  const key = 97 - Number(BigInt(body) % 97n);
  return key === Number(m[7]);
}

/** Germany Steuer-ID (IdNr): 11 digits, first != 0, one digit repeated 2x (or 3x non-consecutively) in first 10, ISO 7064 MOD 11,10 check. */
export function steuerIdValid(s: string): boolean {
  if (!/^[1-9]\d{10}$/.test(s)) return false;
  const first10 = s.slice(0, 10);
  const counts = new Map<string, number>();
  for (const c of first10) counts.set(c, (counts.get(c) ?? 0) + 1);
  const multi = [...counts.entries()].filter(([, n]) => n > 1);
  if (multi.length !== 1) return false;
  const [digit, n] = multi[0];
  if (n > 3) return false;
  if (n === 3) {
    // three occurrences must not be consecutive
    if (first10.includes(digit.repeat(3))) return false;
  }
  // ISO 7064 MOD 11,10
  let product = 10;
  for (const c of first10) {
    let sum = (Number(c) + product) % 10;
    if (sum === 0) sum = 10;
    product = (sum * 2) % 11;
  }
  let check = 11 - product;
  if (check === 10) check = 0;
  return check === Number(s[10]);
}

/** Netherlands BSN: 11-proof with weights 9..2 and -1 on the last digit (8-digit BSN padded with 0). */
export function bsnValid(s: string): boolean {
  if (!/^\d{8,9}$/.test(s)) return false;
  const d = s.padStart(9, "0");
  let sum = 0;
  for (let i = 0; i < 8; i++) sum += (9 - i) * Number(d[i]);
  sum -= Number(d[8]);
  return sum % 11 === 0 && d !== "000000000";
}

/** Italy codice fiscale (16 chars): odd/even tables, sum mod 26 -> letter. Omocodia letters L M N P Q R S T U V allowed in digit slots. */
const CF_ODD: Record<string, number> = {
  "0": 1,
  "1": 0,
  "2": 5,
  "3": 7,
  "4": 9,
  "5": 13,
  "6": 15,
  "7": 17,
  "8": 19,
  "9": 21,
  A: 1,
  B: 0,
  C: 5,
  D: 7,
  E: 9,
  F: 13,
  G: 15,
  H: 17,
  I: 19,
  J: 21,
  K: 2,
  L: 4,
  M: 18,
  N: 20,
  O: 11,
  P: 3,
  Q: 6,
  R: 8,
  S: 12,
  T: 14,
  U: 16,
  V: 10,
  W: 22,
  X: 25,
  Y: 24,
  Z: 23,
};
function cfEven(c: string): number {
  return c >= "0" && c <= "9" ? c.charCodeAt(0) - 48 : c.charCodeAt(0) - 65;
}
const CF_STRUCT =
  /^[A-Z]{6}[0-9LMNPQRSTUV]{2}[ABCDEHLMPRST][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z]$/;
export function codiceFiscaleValid(s: string): boolean {
  if (!CF_STRUCT.test(s)) return false;
  let sum = 0;
  for (let i = 0; i < 15; i++) sum += i % 2 === 0 ? CF_ODD[s[i]] : cfEven(s[i]); // i=0 is "1st" = odd position
  return String.fromCharCode(65 + (sum % 26)) === s[15];
}

/** Portugal NIF: 9 digits, weights 9..2, mod 11; remainder 0/1 -> check 0. First digit 1,2,3,5,6,7,8,9 or prefix 45. */
export function nifPtValid(s: string): boolean {
  if (!/^\d{9}$/.test(s)) return false;
  if (!/^(?:[1235689]|45)/.test(s)) return false;
  let sum = 0;
  for (let i = 0; i < 8; i++) sum += Number(s[i]) * (9 - i);
  const r = sum % 11;
  const check = r < 2 ? 0 : 11 - r;
  return check === Number(s[8]);
}

/** Brazil CPF: 11 digits, two mod-11 check digits (weights 10..2, then 11..2). */
export function cpfValid(s: string): boolean {
  if (!/^\d{11}$/.test(s) || /^(\d)\1{10}$/.test(s)) return false;
  const calc = (len: number) => {
    let sum = 0;
    for (let i = 0; i < len; i++) sum += Number(s[i]) * (len + 1 - i);
    const r = (sum * 10) % 11;
    return r === 10 ? 0 : r;
  };
  return calc(9) === Number(s[9]) && calc(10) === Number(s[10]);
}
