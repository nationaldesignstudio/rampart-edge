//! Weight expansion (4-bit -> packed fp32) and the forward pass of the gate model:
//! int4 embedding table (23910 x 48) -> 48->256 projection + position + token-type -> LayerNorm ->
//! 4 x [shared layer: QKV, 8-head attention, output proj + residual + LN, FFN(1024, erf-GELU) + residual + LN]
//! -> 19-label token head on every position, pooler(tanh) + gate head on [CLS].

use crate::format::{
    code, Header, Tensor, FF, H, HEADER_BYTES, HEADS, LABELS, LAYERS, MAXP, RANK, VOCAB,
};
use crate::kernels::{self, gemm, layernorm, softmax_rows, Packed, GROUP};
use crate::simd::V4;

pub const DH: usize = H / HEADS;
pub const MAXN: usize = MAXP;
/// Token head padded to a multiple of the 8-column group.
pub const LABP: usize = LABELS.div_ceil(GROUP) * GROUP;

/// Bump allocator over the static arena. Every allocation is 64-byte aligned and bounds-checked.
pub struct Arena {
    pub base: *mut u8,
    pub cap: usize,
    pub top: usize,
}

impl Arena {
    pub fn alloc(&mut self, bytes: usize) -> Result<*mut u8, i32> {
        let start = self.top.checked_add(63).ok_or(code::ARENA_FULL)? & !63;
        let end = start.checked_add(bytes).ok_or(code::ARENA_FULL)?;
        if end > self.cap {
            return Err(code::ARENA_FULL);
        }
        self.top = end;
        // SAFETY: start < cap, so the pointer stays inside the arena allocation.
        Ok(unsafe { self.base.add(start) })
    }
    fn f32s(&mut self, n: usize) -> Result<*mut f32, i32> {
        let bytes = n.checked_mul(4).ok_or(code::ARENA_FULL)?;
        let p = self.alloc(bytes)?.cast::<f32>();
        // SAFETY: freshly allocated range of `n` floats.
        unsafe { core::ptr::write_bytes(p, 0, n) };
        Ok(p)
    }
    fn i32s(&mut self, n: usize) -> Result<*mut i32, i32> {
        Ok(self.f32s(n)?.cast::<i32>())
    }
}

pub struct Model {
    emb_q4: *const u8,
    emb_s: *const f32,
    proj: Packed,
    qkv: Packed,
    o: Packed,
    w1: Packed,
    w2: Packed,
    pool: Packed,
    tok: Packed,
    gate_w: *const f32,
    gate_b: f32,
    /// position[p] + token_type[0], precomputed so the embedding sum fuses into the projection GEMM.
    postyp: *const f32,
    ln0: (*const f32, *const f32),
    ln1: (*const f32, *const f32),
    ln2: (*const f32, *const f32),
    qk_scale: f32,
}

pub struct Scratch {
    pub ids: *mut i32,
    pub mask: *mut i32,
    pub tok_out: *mut f32,
    maskf: *mut f32,
    embf: *mut f32,
    x: *mut f32,
    y: *mut f32,
    qkv: *mut f32,
    ctx: *mut f32,
    ff: *mut f32,
    scores: *mut f32,
    kpack: *mut f32,
    vpack: *mut f32,
    tokp: *mut f32,
    pooled: *mut f32,
    zeros: *const f32,
}

/// Dequantize a MatMulNBits tensor (`[n_real, K/32, 16]` nibbles, block 32, unsigned zero point 8) straight into
/// the packed group layout (`out` must be zeroed; padded columns past `n_real` stay zero).
///
/// # Safety
/// `q`/`s` are the validated tensors; `out` has `n_pad * k` zeroed floats.
unsafe fn dequant_pack(q: *const u8, s: *const f32, n_real: usize, k: usize, out: *mut f32) {
    let blocks = k.div_ceil(32);
    for j in 0..n_real {
        let (g, c) = (j / GROUP, j % GROUP);
        for b in 0..blocks {
            let scale = *s.add(j * blocks + b);
            let row = q.add((j * blocks + b) * 16);
            for i in 0..16 {
                let byte = *row.add(i);
                let k0 = b * 32 + 2 * i;
                if k0 < k {
                    *out.add((g * k + k0) * GROUP + c) = ((byte & 15) as i32 - 8) as f32 * scale;
                }
                if k0 + 1 < k {
                    *out.add((g * k + k0 + 1) * GROUP + c) =
                        ((byte >> 4) as i32 - 8) as f32 * scale;
                }
            }
        }
    }
}

/// Expand the validated file into `arena`. The file must stay mapped for the lifetime of the model
/// (the int4 embedding table is read in place).
///
/// # Safety
/// `file` is the buffer that `header` was parsed from.
pub unsafe fn expand(
    file: &[u8],
    header: &Header,
    arena: &mut Arena,
) -> Result<(Model, Scratch), i32> {
    let payload = file.as_ptr().add(HEADER_BYTES);
    let t = |id: usize| -> Tensor { header.tensors[id] };
    let u8p = |id: usize| payload.add(t(id).off);
    let f32p = |id: usize| payload.add(t(id).off).cast::<f32>();

    let mut linear = |qid: usize,
                      bias: Option<usize>,
                      n_real: usize,
                      n_pad: usize,
                      k: usize|
     -> Result<Packed, i32> {
        let p = arena.f32s(n_pad.checked_mul(k).ok_or(code::ARENA_FULL)?)?;
        dequant_pack(u8p(qid), f32p(qid + 1), n_real, k, p);
        let b = arena.f32s(n_pad)?;
        if let Some(bid) = bias {
            core::ptr::copy_nonoverlapping(f32p(bid), b, n_real);
        }
        Ok(Packed {
            p,
            bias: b,
            n: n_pad,
            k,
        })
    };

    let proj = linear(2, None, H, H, RANK)?;
    let o = linear(10, Some(29), H, H, H)?;
    let w1 = linear(12, Some(30), FF, FF, H)?;
    let w2 = linear(14, Some(31), H, H, FF)?;
    let pool = linear(16, Some(32), H, H, H)?;
    let tok = linear(20, Some(34), LABELS, LABP, H)?;

    // Fused QKV: three 256x256 projections stacked into one 768-column packed matrix.
    let qkv_p = arena.f32s(3 * H * H)?;
    let qkv_b = arena.f32s(3 * H)?;
    for (i, qid) in [4usize, 6, 8].iter().enumerate() {
        dequant_pack(u8p(*qid), f32p(qid + 1), H, H, qkv_p.add(i * H * H));
        core::ptr::copy_nonoverlapping(f32p(26 + i), qkv_b.add(i * H), H);
    }
    let qkv = Packed {
        p: qkv_p,
        bias: qkv_b,
        n: 3 * H,
        k: H,
    };

    // Gate head 256 -> 1: dequantize to a plain row.
    let gate_w = arena.f32s(H)?;
    {
        let q = u8p(18);
        let s = f32p(19);
        for b in 0..H / 32 {
            let scale = *s.add(b);
            for i in 0..16 {
                let byte = *q.add(b * 16 + i);
                *gate_w.add(b * 32 + 2 * i) = ((byte & 15) as i32 - 8) as f32 * scale;
                *gate_w.add(b * 32 + 2 * i + 1) = ((byte >> 4) as i32 - 8) as f32 * scale;
            }
        }
    }

    // position + token-type[0] tables (uint8, scalar scale / zero point each).
    let postyp = arena.f32s(MAXP * H)?;
    {
        let (ps, pz) = (*f32p(23), *f32p(23).add(1));
        let (ts, tz) = (*f32p(25), *f32p(25).add(1));
        let pq = u8p(22);
        let tq = u8p(24);
        for p in 0..MAXP {
            for c in 0..H {
                let typ = (*tq.add(c) as f32 - tz) * ts;
                let pos = (*pq.add(p * H + c) as f32 - pz) * ps;
                *postyp.add(p * H + c) = typ + pos;
            }
        }
    }

    let model = Model {
        emb_q4: u8p(0),
        emb_s: f32p(1),
        proj,
        qkv,
        o,
        w1,
        w2,
        pool,
        tok,
        gate_w,
        gate_b: *f32p(33),
        postyp,
        ln0: (f32p(35), f32p(36)),
        ln1: (f32p(37), f32p(38)),
        ln2: (f32p(39), f32p(40)),
        qk_scale: *f32p(41) * *f32p(41).add(1),
    };

    let scratch = Scratch {
        ids: arena.i32s(MAXN)?,
        mask: arena.i32s(MAXN)?,
        tok_out: arena.f32s(MAXN * LABELS)?,
        maskf: arena.f32s(MAXN)?,
        embf: arena.f32s(MAXN * RANK)?,
        x: arena.f32s(MAXN * H)?,
        y: arena.f32s(MAXN * H)?,
        qkv: arena.f32s(MAXN * 3 * H)?,
        ctx: arena.f32s(MAXN * H)?,
        ff: arena.f32s(MAXN * FF)?,
        scores: arena.f32s(MAXN * MAXN)?,
        kpack: arena.f32s(MAXN * DH)?,
        vpack: arena.f32s(MAXN * DH)?,
        tokp: arena.f32s(MAXN * LABP)?,
        pooled: arena.f32s(H)?,
        zeros: arena.f32s(MAXN)?,
    };
    Ok((model, scratch))
}

/// Pack head `h` of K (scaled by `scale`) as `ld/8` groups of `DH x 8`; key columns `>= n` are zero.
///
/// # Safety
/// `qkv` is `n x 3H`, `out` has `ld * DH` floats.
unsafe fn pack_k(qkv: *const f32, n: usize, ld: usize, h: usize, scale: f32, out: *mut f32) {
    for g in 0..ld / GROUP {
        for d in 0..DH {
            for c in 0..GROUP {
                let j = g * GROUP + c;
                let v = if j < n {
                    *qkv.add(j * 3 * H + H + h * DH + d) * scale
                } else {
                    0.0
                };
                *out.add((g * DH + d) * GROUP + c) = v;
            }
        }
    }
}

/// Pack head `h` of V as `DH/8` groups of `n x 8` (the "weight" of the probs x V GEMM, k = n).
///
/// # Safety
/// `qkv` is `n x 3H`, `out` has `n * DH` floats.
unsafe fn pack_v(qkv: *const f32, n: usize, h: usize, out: *mut f32) {
    for g in 0..DH / GROUP {
        for j in 0..n {
            let src = qkv.add(j * 3 * H + 2 * H + h * DH + g * GROUP);
            let dst = out.add((g * n + j) * GROUP);
            V4::load(src).store(dst);
            V4::load(src.add(4)).store(dst.add(4));
        }
    }
}

/// Full forward for `n` tokens from `s.ids` / `s.mask`. Returns the gate logit, or NaN if a token id is out
/// of range or the result is not finite. Token logits (`n x 19`) are left in `s.tok_out`.
///
/// # Safety
/// `m` and `s` come from `expand`; `1 <= n <= MAXN`.
pub unsafe fn forward(m: &Model, s: &mut Scratch, n: usize) -> f32 {
    if n == 0 || n > MAXN {
        return f32::NAN;
    }
    let ld = n.div_ceil(GROUP) * GROUP;

    for t in 0..n {
        let id = *s.ids.add(t);
        if id < 0 || id as usize >= VOCAB {
            return f32::NAN;
        }
        let id = id as usize;
        let scale = *m.emb_s.add(id);
        let row = m.emb_q4.add(id * RANK / 2);
        for i in 0..RANK / 2 {
            let byte = *row.add(i);
            let lo = (byte & 15) as i32;
            let hi = (byte >> 4) as i32;
            let lo = if lo >= 8 { lo - 16 } else { lo };
            let hi = if hi >= 8 { hi - 16 } else { hi };
            *s.embf.add(t * RANK + 2 * i) = lo as f32 * scale;
            *s.embf.add(t * RANK + 2 * i + 1) = hi as f32 * scale;
        }
        *s.maskf.add(t) = if *s.mask.add(t) != 0 {
            0.0
        } else {
            f32::NEG_INFINITY
        };
    }
    for j in n..ld {
        *s.maskf.add(j) = f32::NEG_INFINITY;
    }

    // Embeddings: projection with the (position + token-type) sum fused as the residual, then LN.
    let (mut x, mut y) = (s.x, s.y);
    gemm(s.embf, RANK, &m.proj, m.postyp, H, x, H, n);
    layernorm(x, n, H, m.ln0.0, m.ln0.1);

    for _ in 0..LAYERS {
        gemm(x, H, &m.qkv, core::ptr::null(), 0, s.qkv, 3 * H, n);
        for h in 0..HEADS {
            pack_k(s.qkv, n, ld, h, m.qk_scale, s.kpack);
            let kw = Packed {
                p: s.kpack,
                bias: s.zeros,
                n: ld,
                k: DH,
            };
            gemm(
                s.qkv.add(h * DH),
                3 * H,
                &kw,
                core::ptr::null(),
                0,
                s.scores,
                ld,
                n,
            );
            softmax_rows(s.scores, n, ld, s.maskf);
            pack_v(s.qkv, n, h, s.vpack);
            let vw = Packed {
                p: s.vpack,
                bias: s.zeros,
                n: DH,
                k: n,
            };
            gemm(
                s.scores,
                ld,
                &vw,
                core::ptr::null(),
                0,
                s.ctx.add(h * DH),
                H,
                n,
            );
        }
        gemm(s.ctx, H, &m.o, x, H, y, H, n);
        layernorm(y, n, H, m.ln1.0, m.ln1.1);
        core::mem::swap(&mut x, &mut y);

        gemm(x, H, &m.w1, core::ptr::null(), 0, s.ff, FF, n);
        kernels::gelu(s.ff, n * FF);
        gemm(s.ff, FF, &m.w2, x, H, y, H, n);
        layernorm(y, n, H, m.ln2.0, m.ln2.1);
        core::mem::swap(&mut x, &mut y);
    }

    gemm(x, H, &m.tok, core::ptr::null(), 0, s.tokp, LABP, n);
    for t in 0..n {
        core::ptr::copy_nonoverlapping(s.tokp.add(t * LABP), s.tok_out.add(t * LABELS), LABELS);
    }

    gemm(x, H, &m.pool, core::ptr::null(), 0, s.pooled, H, 1);
    let mut gate = m.gate_b;
    for c in 0..H {
        gate += kernels::tanh_f(*s.pooled.add(c)) * *m.gate_w.add(c);
    }
    if gate.is_finite() {
        gate
    } else {
        f32::NAN
    }
}
