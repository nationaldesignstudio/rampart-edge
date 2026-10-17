//! RMPW v1 weight container (frozen; see `docs/` of the exporter). Little-endian throughout.
//!
//! ```text
//! "RMPW" | u32 version=1 | u32 hidden, ffn, heads, layers, vocab, rank, max_pos, labels, n_tensors=42
//! | 32 B sha256(payload) | table: n_tensors x (u32 id, dtype{0=u8,1=f32}, nelems, offset, nbytes)
//! | pad to 16 | payload (every tensor 16-B aligned, offsets relative to payload start)
//! ```
//!
//! `parse` validates every header field against the compiled model shape and returns the tensor table
//! indexed by id. The payload digest is verified separately by the caller (`check_payload`).

use crate::sha256;

pub const MAGIC: u32 = 0x5750_4d52; // "RMPW"
pub const VERSION: u32 = 1;
pub const N_TENSORS: usize = 42;
pub const FIXED_BYTES: usize = 76;
pub const ENTRY_BYTES: usize = 20;
/// Header size including the table and its pad: (76 + 42 * 20 + 15) & !15.
pub const HEADER_BYTES: usize = (FIXED_BYTES + N_TENSORS * ENTRY_BYTES + 15) & !15;

pub const H: usize = 256;
pub const FF: usize = 1024;
pub const HEADS: usize = 8;
pub const LAYERS: usize = 4;
pub const VOCAB: usize = 23910;
pub const RANK: usize = 48;
pub const MAXP: usize = 384;
pub const LABELS: usize = 19;
pub const BLOCK: usize = 32;

/// Return codes of `init` (0 = ok). Stable; mirrored in `src/engine.ts`.
pub mod code {
    pub const OK: i32 = 0;
    /// `init` was already called successfully.
    pub const ALREADY_INIT: i32 = 1;
    /// The (ptr, len) range is not inside the arena (must come from `arena_alloc`) or is misaligned.
    pub const BAD_BUFFER: i32 = 2;
    pub const BAD_MAGIC: i32 = 3;
    pub const BAD_VERSION: i32 = 4;
    /// A shape field differs from the compiled constants, or n_tensors != 42.
    pub const BAD_SHAPE: i32 = 5;
    /// Tensor table malformed: duplicate/unknown id, wrong dtype/nelems/nbytes, misaligned, overlapping or out of range.
    pub const BAD_TABLE: i32 = 6;
    /// sha256(payload) does not match the header digest.
    pub const BAD_DIGEST: i32 = 7;
    /// Not enough arena left for the expanded weights and scratch.
    pub const ARENA_FULL: i32 = 8;
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Tensor {
    /// Byte offset relative to the payload start.
    pub off: usize,
    pub nbytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    U8 = 0,
    F32 = 1,
}

const fn blocks(k: usize) -> usize {
    k.div_ceil(BLOCK)
}

/// Expected (dtype, nelems) of tensor `id`, in exporter order.
pub const fn expected(id: usize) -> (Dtype, usize) {
    use Dtype::*;
    match id {
        0 => (U8, VOCAB * RANK / 2),
        1 => (F32, VOCAB),
        // MatMulNBits pairs: (Q4 bytes [N, K/32, 16], f32 scales [N, K/32])
        2 => (U8, H * blocks(RANK) * 16), // proj  48->256
        3 => (F32, H * blocks(RANK)),
        4 | 6 | 8 | 10 | 16 => (U8, H * blocks(H) * 16), // q, k, v, o, pool 256->256
        5 | 7 | 9 | 11 | 17 => (F32, H * blocks(H)),
        12 => (U8, FF * blocks(H) * 16), // w1 256->1024
        13 => (F32, FF * blocks(H)),
        14 => (U8, H * blocks(FF) * 16), // w2 1024->256
        15 => (F32, H * blocks(FF)),
        18 => (U8, blocks(H) * 16), // gate 256->1
        19 => (F32, blocks(H)),
        20 => (U8, LABELS * blocks(H) * 16), // token head 256->19
        21 => (F32, LABELS * blocks(H)),
        22 => (U8, MAXP * H), // position table u8 + [scale, zp]
        23 => (F32, 2),
        24 => (U8, 2 * H), // token-type table u8 + [scale, zp]
        25 => (F32, 2),
        26..=29 => (F32, H), // q, k, v, o bias
        30 => (F32, FF),     // ffn bias 1
        31 | 32 => (F32, H), // ffn bias 2, pooler bias
        33 => (F32, 1),      // gate bias
        34 => (F32, LABELS), // token head bias
        35..=40 => (F32, H), // LN0 w,b  LN1 w,b  LN2 w,b
        41 => (F32, 2),      // [q_scale, k_scale]
        _ => (U8, 0),
    }
}

#[inline]
fn rd_u32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]])
}

pub struct Header {
    pub tensors: [Tensor; N_TENSORS],
    pub digest: [u8; 32],
}

/// Validate the header of `buf` (the whole file). Never panics: every index is bounds-checked against
/// `buf.len()` before use and every size computation is overflow-checked.
pub fn parse(buf: &[u8]) -> Result<Header, i32> {
    if buf.len() < HEADER_BYTES {
        return Err(code::BAD_MAGIC);
    }
    if rd_u32(buf, 0) != MAGIC {
        return Err(code::BAD_MAGIC);
    }
    if rd_u32(buf, 4) != VERSION {
        return Err(code::BAD_VERSION);
    }
    let dims = [H, FF, HEADS, LAYERS, VOCAB, RANK, MAXP, LABELS, N_TENSORS];
    for (i, d) in dims.iter().enumerate() {
        if rd_u32(buf, 8 + 4 * i) as usize != *d {
            return Err(code::BAD_SHAPE);
        }
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&buf[44..76]);

    let payload_len = buf.len() - HEADER_BYTES;
    let mut tensors = [Tensor::default(); N_TENSORS];
    let mut seen = [false; N_TENSORS];
    for t in 0..N_TENSORS {
        let e = FIXED_BYTES + t * ENTRY_BYTES;
        let id = rd_u32(buf, e) as usize;
        let dtype = rd_u32(buf, e + 4);
        let nelems = rd_u32(buf, e + 8) as usize;
        let off = rd_u32(buf, e + 12) as usize;
        let nbytes = rd_u32(buf, e + 16) as usize;
        if id >= N_TENSORS || seen[id] {
            return Err(code::BAD_TABLE);
        }
        let (want_dtype, want_n) = expected(id);
        if dtype != want_dtype as u32 || nelems != want_n {
            return Err(code::BAD_TABLE);
        }
        let elem = match want_dtype {
            Dtype::U8 => 1,
            Dtype::F32 => 4,
        };
        let want_bytes = match nelems.checked_mul(elem) {
            Some(b) => b,
            None => return Err(code::BAD_TABLE),
        };
        if nbytes != want_bytes || !off.is_multiple_of(16) {
            return Err(code::BAD_TABLE);
        }
        let end = match off.checked_add(nbytes) {
            Some(e) => e,
            None => return Err(code::BAD_TABLE),
        };
        if end > payload_len {
            return Err(code::BAD_TABLE);
        }
        seen[id] = true;
        tensors[id] = Tensor { off, nbytes };
    }
    for i in 0..N_TENSORS {
        for j in 0..i {
            let (a, b) = (tensors[i], tensors[j]);
            if a.off < b.off + b.nbytes && b.off < a.off + a.nbytes {
                return Err(code::BAD_TABLE);
            }
        }
    }
    Ok(Header { tensors, digest })
}

/// Verify sha256(payload) against the header digest.
pub fn check_payload(buf: &[u8], header: &Header) -> Result<(), i32> {
    if buf.len() < HEADER_BYTES {
        return Err(code::BAD_MAGIC);
    }
    if sha256::digest(&buf[HEADER_BYTES..]) == header.digest {
        Ok(())
    } else {
        Err(code::BAD_DIGEST)
    }
}

/// Build a syntactically valid header + zero payload (host test helper).
#[cfg(test)]
pub fn synth_file() -> alloc_vec::Vec<u8> {
    let mut table = alloc_vec::Vec::new();
    let mut off = 0usize;
    for id in 0..N_TENSORS {
        let (dt, n) = expected(id);
        let nb = n * if matches!(dt, Dtype::F32) { 4 } else { 1 };
        table.push((id as u32, dt as u32, n as u32, off as u32, nb as u32));
        off = if id + 1 == N_TENSORS {
            off + nb
        } else {
            (off + nb + 15) & !15
        };
    }
    let payload = alloc_vec::vec![0u8; off];
    let mut buf = alloc_vec::Vec::with_capacity(HEADER_BYTES + off);
    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.extend_from_slice(&VERSION.to_le_bytes());
    for d in [H, FF, HEADS, LAYERS, VOCAB, RANK, MAXP, LABELS, N_TENSORS] {
        buf.extend_from_slice(&(d as u32).to_le_bytes());
    }
    buf.extend_from_slice(&sha256::digest(&payload));
    for (id, dt, n, o, nb) in table {
        for v in [id, dt, n, o, nb] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf.resize(HEADER_BYTES, 0);
    buf.extend_from_slice(&payload);
    buf
}

#[cfg(test)]
mod alloc_vec {
    extern crate alloc;
    pub use alloc::vec;
    pub use alloc::vec::Vec;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u32(buf: &mut [u8], at: usize, v: u32) {
        buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    #[test]
    fn header_constants() {
        assert_eq!(HEADER_BYTES, 928);
        // Exporter layout: every tensor 16-B aligned, no pad after the last one.
        assert_eq!(synth_file().len(), 1_331_672, "golden file size");
    }

    #[test]
    fn synth_parses() {
        let f = synth_file();
        let h = parse(&f).expect("valid");
        check_payload(&f, &h).expect("digest");
        assert_eq!(
            h.tensors[0],
            Tensor {
                off: 0,
                nbytes: VOCAB * RANK / 2
            }
        );
    }

    #[test]
    fn truncated_at_every_header_length() {
        let f = synth_file();
        for len in 0..HEADER_BYTES {
            assert_eq!(parse(&f[..len]).err(), Some(code::BAD_MAGIC), "len {len}");
        }
        // Full header but payload short by one byte -> the last tensor runs past the end.
        assert_eq!(parse(&f[..f.len() - 1]).err(), Some(code::BAD_TABLE));
    }

    #[test]
    fn every_fixed_field_is_checked() {
        let f = synth_file();
        let cases: [(usize, i32); 11] = [
            (0, code::BAD_MAGIC),
            (4, code::BAD_VERSION),
            (8, code::BAD_SHAPE),
            (12, code::BAD_SHAPE),
            (16, code::BAD_SHAPE),
            (20, code::BAD_SHAPE),
            (24, code::BAD_SHAPE),
            (28, code::BAD_SHAPE),
            (32, code::BAD_SHAPE),
            (36, code::BAD_SHAPE),
            (40, code::BAD_SHAPE),
        ];
        for (at, want) in cases {
            for delta in [1u32, 0x8000_0000, u32::MAX] {
                let mut g = f.clone();
                let v = rd_u32(&g, at).wrapping_add(delta);
                put_u32(&mut g, at, v);
                assert_eq!(parse(&g).err(), Some(want), "field @{at} +{delta:#x}");
            }
        }
        // Digest bytes: header parses, payload check fails.
        for i in 44..76 {
            let mut g = f.clone();
            g[i] ^= 0x01;
            let h = parse(&g).expect("header still valid");
            assert_eq!(
                check_payload(&g, &h).err(),
                Some(code::BAD_DIGEST),
                "digest byte {i}"
            );
        }
        // Payload bytes: a single flipped bit is caught.
        for i in [HEADER_BYTES, HEADER_BYTES + 12345, f.len() - 1] {
            let mut g = f.clone();
            g[i] ^= 0x80;
            let h = parse(&g).expect("header still valid");
            assert_eq!(
                check_payload(&g, &h).err(),
                Some(code::BAD_DIGEST),
                "payload byte {i}"
            );
        }
    }

    #[test]
    fn every_table_field_is_checked() {
        let f = synth_file();
        for t in 0..N_TENSORS {
            let e = FIXED_BYTES + t * ENTRY_BYTES;
            for field in 0..5 {
                for delta in [1u32, 16, 0x8000_0000, u32::MAX] {
                    let mut g = f.clone();
                    let at = e + 4 * field;
                    let v = rd_u32(&g, at).wrapping_add(delta);
                    put_u32(&mut g, at, v);
                    let r = parse(&g).err();
                    assert_eq!(
                        r,
                        Some(code::BAD_TABLE),
                        "tensor {t} field {field} +{delta:#x}"
                    );
                }
            }
        }
        // Duplicate id (second entry copies the first) -> BAD_TABLE even though each is well-formed.
        let mut g = f.clone();
        let (a, b) = (FIXED_BYTES, FIXED_BYTES + ENTRY_BYTES);
        let first: Vec<u8> = g[a..a + ENTRY_BYTES].to_vec();
        g[b..b + ENTRY_BYTES].copy_from_slice(&first);
        assert_eq!(parse(&g).err(), Some(code::BAD_TABLE));
        // Swapping two entries is fine: the table is keyed by id.
        let mut g = f.clone();
        let second: Vec<u8> = g[b..b + ENTRY_BYTES].to_vec();
        g[a..a + ENTRY_BYTES].copy_from_slice(&second);
        g[b..b + ENTRY_BYTES].copy_from_slice(&first);
        assert!(parse(&g).is_ok());
    }

    #[test]
    fn random_mutations_never_panic() {
        // Deterministic xorshift fuzz over the header region; the only acceptable outcomes are Ok or an error code.
        let f = synth_file();
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..20_000 {
            let mut g = f.clone();
            let flips = 1 + (s % 4) as usize;
            for _ in 0..flips {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let at = (s % HEADER_BYTES as u64) as usize;
                g[at] ^= (s >> 20) as u8 | 1;
            }
            let _ = parse(&g).map(|h| check_payload(&g, &h));
            // Random truncation too.
            let cut = (s % (g.len() as u64 + 1)) as usize;
            let _ = parse(&g[..cut]).map(|h| check_payload(&g[..cut], &h));
        }
    }
}
