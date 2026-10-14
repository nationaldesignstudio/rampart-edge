import type { Norm } from "./types.js";

// Zero-width / joiner / BOM characters that v0.1.3 leaks on (audit: "zero-width/NBSP separators in SSNs").
const ZERO_WIDTH = new Set([
  0x200b, 0x200c, 0x200d, 0x200e, 0x200f, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0xfeff, 0x00ad,
  0x034f,
]);
// Space-like code points mapped to ASCII space (NBSP, NNBSP, figure space, thin space, ideographic space...).
const SPACES = new Set([
  0x00a0, 0x202f, 0x2007, 0x2008, 0x2009, 0x200a, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x3000,
  0x1680, 0x205f,
]); // tab is kept: it is a column separator, not a phone/id group separator
// Dash-like code points mapped to "-".
const DASHES = new Set([0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212, 0xfe63, 0xff0d]);

export function normalize(original: string): Norm {
  const out: string[] = [];
  const map: number[] = [];
  for (let i = 0; i < original.length; i++) {
    const cp = original.charCodeAt(i);
    if (ZERO_WIDTH.has(cp)) continue;
    let ch = original[i];
    if (SPACES.has(cp)) ch = " ";
    else if (DASHES.has(cp)) ch = "-";
    else if (cp >= 0xff10 && cp <= 0xff19) ch = String.fromCharCode(cp - 0xff10 + 0x30); // fullwidth digits
    else if (cp >= 0xff21 && cp <= 0xff3a) ch = String.fromCharCode(cp - 0xff21 + 0x41); // fullwidth A-Z
    else if (cp >= 0xff41 && cp <= 0xff5a) ch = String.fromCharCode(cp - 0xff41 + 0x61);
    out.push(ch);
    map.push(i);
  }
  return { text: out.join(""), map, original };
}

/** Convert [startNorm, endNorm) to original offsets. */
export function toOriginal(n: Norm, startNorm: number, endNorm: number): [number, number] {
  const s = n.map[startNorm];
  const e = n.map[endNorm - 1] + 1;
  return [s, e];
}
