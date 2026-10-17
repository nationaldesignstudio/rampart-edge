//! fp32 kernels over `V4`. All arithmetic is plain IEEE add/sub/mul/div in a fixed order (no FMA, no
//! reassociation), so results are identical on every wasm engine and on the host test build.
//!
//! Weight layout for `gemm`: `Packed` holds `n/8` column groups, each `k x 8` floats
//! (`p[(g*k + kk)*8 + c] = W[8g + c][kk]`), and a bias of `n` floats. `n % 8 == 0` is required.

use crate::simd::V4;

pub const GROUP: usize = 8;

#[derive(Clone, Copy)]
pub struct Packed {
    pub p: *const f32,
    pub bias: *const f32,
    pub n: usize,
    pub k: usize,
}

/// `RB` rows x 8 columns, K unrolled by 4, bias folded into the accumulator init, optional residual add.
///
/// # Safety
/// `a` must hold `RB` rows of `k` floats (stride `lda`), `p` a `k x 8` panel, `bias` 8 floats, `c` `RB` rows
/// of 8 floats (stride `ldc`), and `res` (if non-null) `RB` rows of 8 floats (stride `ldr`).
#[inline(always)]
unsafe fn micro<const RB: usize>(
    a: *const f32,
    lda: usize,
    p: *const f32,
    bias: *const f32,
    res: *const f32,
    ldr: usize,
    c: *mut f32,
    ldc: usize,
    k: usize,
) {
    let mut acc = [[V4::load(bias), V4::load(bias.add(4))]; RB];
    let mut pk = p;
    let mut kk = 0usize;
    while kk + 4 <= k {
        let mut u = 0;
        while u < 4 {
            let w0 = V4::load(pk.add(u * 8));
            let w1 = V4::load(pk.add(u * 8 + 4));
            let mut r = 0;
            while r < RB {
                let x = V4::load_splat(a.add(r * lda + kk + u));
                acc[r][0] = acc[r][0].add(x.mul(w0));
                acc[r][1] = acc[r][1].add(x.mul(w1));
                r += 1;
            }
            u += 1;
        }
        pk = pk.add(32);
        kk += 4;
    }
    while kk < k {
        let w0 = V4::load(pk);
        let w1 = V4::load(pk.add(4));
        let mut r = 0;
        while r < RB {
            let x = V4::load_splat(a.add(r * lda + kk));
            acc[r][0] = acc[r][0].add(x.mul(w0));
            acc[r][1] = acc[r][1].add(x.mul(w1));
            r += 1;
        }
        pk = pk.add(8);
        kk += 1;
    }
    let mut r = 0;
    while r < RB {
        let (mut v0, mut v1) = (acc[r][0], acc[r][1]);
        if !res.is_null() {
            v0 = v0.add(V4::load(res.add(r * ldr)));
            v1 = v1.add(V4::load(res.add(r * ldr + 4)));
        }
        v0.store(c.add(r * ldc));
        v1.store(c.add(r * ldc + 4));
        r += 1;
    }
}

#[inline(always)]
fn row_res(res: *const f32, off: usize) -> *const f32 {
    if res.is_null() {
        res
    } else {
        res.wrapping_add(off)
    }
}

/// `C[m][n] = A[m][k] . W^T + bias (+ res)`.
///
/// # Safety
/// `a` is `m x k` (stride `lda`), `c` is `m x n` (stride `ldc`), `res` (nullable) is `m x n` (stride `ldr`),
/// `w` is packed for exactly `k`.
pub unsafe fn gemm(
    a: *const f32,
    lda: usize,
    w: &Packed,
    res: *const f32,
    ldr: usize,
    c: *mut f32,
    ldc: usize,
    m: usize,
) {
    let k = w.k;
    let groups = w.n / GROUP;
    for g in 0..groups {
        let p = w.p.add(g * k * GROUP);
        let bias = w.bias.add(g * GROUP);
        let cg = c.add(g * GROUP);
        let rg = if res.is_null() {
            res
        } else {
            res.add(g * GROUP)
        };
        let mut i = 0usize;
        while m - i >= 6 {
            micro::<6>(
                a.add(i * lda),
                lda,
                p,
                bias,
                row_res(rg, i * ldr),
                ldr,
                cg.add(i * ldc),
                ldc,
                k,
            );
            i += 6;
        }
        if m - i >= 4 {
            micro::<4>(
                a.add(i * lda),
                lda,
                p,
                bias,
                row_res(rg, i * ldr),
                ldr,
                cg.add(i * ldc),
                ldc,
                k,
            );
            i += 4;
        }
        if m - i >= 2 {
            micro::<2>(
                a.add(i * lda),
                lda,
                p,
                bias,
                row_res(rg, i * ldr),
                ldr,
                cg.add(i * ldc),
                ldc,
                k,
            );
            i += 2;
        }
        if m - i >= 1 {
            micro::<1>(
                a.add(i * lda),
                lda,
                p,
                bias,
                row_res(rg, i * ldr),
                ldr,
                cg.add(i * ldc),
                ldc,
                k,
            );
        }
    }
}

/// Pack a row-major `W[n][k]` (fp32) into the group layout. `n % 8 == 0`.
///
/// # Safety
/// `w` is `n x k`, `out` has `n * k` floats.
pub unsafe fn pack_rows(w: *const f32, n: usize, k: usize, out: *mut f32) {
    for g in 0..n / GROUP {
        for kk in 0..k {
            for c in 0..GROUP {
                *out.add((g * k + kk) * GROUP + c) = *w.add((g * GROUP + c) * k + kk);
            }
        }
    }
}

/// `exp(x)` lane-wise: `2^n * p(r)`, degree-6 Taylor on `|r| <= ln2/2`, exact 0 below -87.
#[inline(always)]
pub fn exp4(x: V4) -> V4 {
    let lo = V4::splat(-87.0);
    let keep = x.gt(lo);
    let xc = x.min(V4::splat(88.0)).max(lo);
    let n = xc.mul(V4::splat(core::f32::consts::LOG2_E)).nearest();
    let r = xc.sub(n.mul(V4::splat(0.693_145_75)));
    let r = r.sub(n.mul(V4::splat(1.428_606_8e-6)));
    let mut p = V4::splat(1.0 / 720.0);
    for c in [1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0] {
        p = p.mul(r).add(V4::splat(c));
    }
    p.mul(n.pow2_from_int()).and(keep)
}

/// `erf(x)` lane-wise, Abramowitz-Stegun 7.1.26 (max abs err ~1.5e-7).
#[inline(always)]
pub fn erf4(x: V4) -> V4 {
    let sign = x.sign_bits();
    let ax = x.abs();
    let one = V4::splat(1.0);
    let t = one.div(one.add(V4::splat(0.327_591_1).mul(ax)));
    let mut p = V4::splat(1.061_405_4);
    for c in [-1.453_152, 1.421_413_7, -0.284_496_74, 0.254_829_6] {
        p = p.mul(t).add(V4::splat(c));
    }
    let y = one.sub(p.mul(t).mul(exp4(ax.mul(ax).neg())));
    y.or(sign)
}

/// In-place erf-GELU over `n` floats (`n % 4 == 0`).
///
/// # Safety
/// `x` has `n` floats.
pub unsafe fn gelu(x: *mut f32, n: usize) {
    let h = V4::splat(0.5);
    let one = V4::splat(1.0);
    let rs2 = V4::splat(core::f32::consts::FRAC_1_SQRT_2);
    let mut i = 0;
    while i < n {
        let v = V4::load(x.add(i));
        v.mul(h).mul(one.add(erf4(v.mul(rs2)))).store(x.add(i));
        i += 4;
    }
}

/// Two-pass LayerNorm over rows of `h` floats (`h % 4 == 0`), eps 1e-12, in place.
///
/// # Safety
/// `x` is `m x h` contiguous; `w`, `b` have `h` floats.
pub unsafe fn layernorm(x: *mut f32, m: usize, h: usize, w: *const f32, b: *const f32) {
    let inv_h = 1.0 / h as f32;
    for i in 0..m {
        let row = x.add(i * h);
        let mut s = V4::zero();
        let mut j = 0;
        while j < h {
            s = s.add(V4::load(row.add(j)));
            j += 4;
        }
        let mean = V4::splat(s.hsum() * inv_h);
        let mut v = V4::zero();
        j = 0;
        while j < h {
            let d = V4::load(row.add(j)).sub(mean);
            v = v.add(d.mul(d));
            j += 4;
        }
        let var = V4::splat(v.hsum() * inv_h + 1e-12);
        let inv = V4::splat(1.0).div(var.sqrt());
        j = 0;
        while j < h {
            let d = V4::load(row.add(j)).sub(mean);
            d.mul(inv)
                .mul(V4::load(w.add(j)))
                .add(V4::load(b.add(j)))
                .store(row.add(j));
            j += 4;
        }
    }
}

/// Masked softmax over rows of `ld` floats (`ld % 4 == 0`), in place. `maskf[j]` is 0 or -inf and must be
/// -inf for every padding column `j >= n`, which therefore end up exactly 0.
///
/// # Safety
/// `x` is `m x ld` contiguous; `maskf` has `ld` floats.
pub unsafe fn softmax_rows(x: *mut f32, m: usize, ld: usize, maskf: *const f32) {
    for i in 0..m {
        let row = x.add(i * ld);
        let mut mx = V4::splat(f32::NEG_INFINITY);
        let mut j = 0;
        while j < ld {
            let v = V4::load(row.add(j)).add(V4::load(maskf.add(j)));
            v.store(row.add(j));
            mx = mx.max(v);
            j += 4;
        }
        let m4 = V4::splat(mx.hmax());
        let mut sum = V4::zero();
        j = 0;
        while j < ld {
            let e = exp4(V4::load(row.add(j)).sub(m4));
            e.store(row.add(j));
            sum = sum.add(e);
            j += 4;
        }
        let inv = V4::splat(1.0 / sum.hsum());
        j = 0;
        while j < ld {
            V4::load(row.add(j)).mul(inv).store(row.add(j));
            j += 4;
        }
    }
}

/// `tanh(x) = 1 - 2 / (exp(2x) + 1)` through `exp4` (scalar, used on the 256 pooled values).
#[inline(always)]
pub fn tanh_f(x: f32) -> f32 {
    let e = exp4(V4::splat(2.0 * x)).lane0();
    1.0 - 2.0 / (e + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }

    fn v4(x: [f32; 4]) -> V4 {
        unsafe { V4::load(x.as_ptr()) }
    }
    fn arr(v: V4) -> [f32; 4] {
        let mut o = [0f32; 4];
        unsafe { v.store(o.as_mut_ptr()) };
        o
    }

    #[test]
    fn gemm_matches_scalar_reference_every_row_tail() {
        let mut seed = 7u64;
        for &(m, n, k) in &[
            (1usize, 8usize, 5usize),
            (2, 16, 48),
            (3, 8, 33),
            (5, 24, 64),
            (6, 8, 4),
            (7, 32, 7),
            (13, 16, 256),
            (384, 8, 32),
        ] {
            let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
            let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
            let bias: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
            let res: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed)).collect();
            let mut packed = vec![0f32; n * k];
            unsafe { pack_rows(w.as_ptr(), n, k, packed.as_mut_ptr()) };
            let pk = Packed {
                p: packed.as_ptr(),
                bias: bias.as_ptr(),
                n,
                k,
            };
            for use_res in [false, true] {
                let mut c = vec![0f32; m * n];
                let rp = if use_res {
                    res.as_ptr()
                } else {
                    core::ptr::null()
                };
                unsafe { gemm(a.as_ptr(), k, &pk, rp, n, c.as_mut_ptr(), n, m) };
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = bias[j];
                        for kk in 0..k {
                            acc += a[i * k + kk] * w[j * k + kk];
                        }
                        if use_res {
                            acc += res[i * n + j];
                        }
                        let got = c[i * n + j];
                        assert!(
                            (got - acc).abs() <= 1e-5 * (1.0 + acc.abs()),
                            "m{m} n{n} k{k} [{i},{j}] {got} vs {acc}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gemm_is_bit_exact_vs_sequential_scalar() {
        // Same operation order as the kernel (bias first, then k ascending) -> bit-identical.
        let (m, n, k) = (9usize, 16usize, 11usize);
        let mut seed = 3u64;
        let a: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed)).collect();
        let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
        let bias: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
        let mut packed = vec![0f32; n * k];
        unsafe { pack_rows(w.as_ptr(), n, k, packed.as_mut_ptr()) };
        let pk = Packed {
            p: packed.as_ptr(),
            bias: bias.as_ptr(),
            n,
            k,
        };
        let mut c = vec![0f32; m * n];
        unsafe {
            gemm(
                a.as_ptr(),
                k,
                &pk,
                core::ptr::null(),
                0,
                c.as_mut_ptr(),
                n,
                m,
            )
        };
        for i in 0..m {
            for j in 0..n {
                let mut acc = bias[j];
                for kk in 0..k {
                    acc += a[i * k + kk] * w[j * k + kk];
                }
                assert_eq!(c[i * n + j].to_bits(), acc.to_bits());
            }
        }
    }

    #[test]
    fn exp_erf_tanh_vs_libm() {
        let mut x = -90.0f32;
        while x < 90.0 {
            let v = arr(exp4(V4::splat(x)))[0];
            let want = x.min(88.0).exp();
            if x < -87.0 {
                assert_eq!(v, 0.0);
            } else {
                assert!(
                    (v - want).abs() <= 3e-7 * want.max(1e-30),
                    "exp {x}: {v} vs {want}"
                );
            }
            x += 0.137;
        }
        assert_eq!(arr(exp4(V4::splat(f32::NEG_INFINITY)))[0], 0.0);
        let mut x = -6.0f32;
        while x < 6.0 {
            let e = arr(erf4(V4::splat(x)))[0];
            let want = libm_erf(x as f64) as f32;
            assert!((e - want).abs() <= 5e-7, "erf {x}: {e} vs {want}");
            let t = tanh_f(x);
            assert!(
                (t - x.tanh()).abs() <= 4e-7,
                "tanh {x}: {t} vs {}",
                x.tanh()
            );
            x += 0.0173;
        }
        assert_eq!(tanh_f(60.0), 1.0);
        assert_eq!(tanh_f(-60.0), -1.0);
    }

    fn libm_erf(x: f64) -> f64 {
        // High-precision reference via series / continued fraction (good to ~1e-12 on |x|<6).
        let ax = x.abs();
        let r = if ax < 2.5 {
            let mut sum = ax;
            let mut term = ax;
            let mut n = 0.0;
            while term.abs() > 1e-17 {
                n += 1.0;
                term *= -ax * ax / n;
                sum += term / (2.0 * n + 1.0);
            }
            sum * 2.0 / core::f64::consts::PI.sqrt()
        } else {
            // erfc continued fraction
            let mut f = 0.0;
            for k in (1..60).rev() {
                f = (k as f64 / 2.0) / (ax + f);
            }
            1.0 - (-ax * ax).exp() / core::f64::consts::PI.sqrt() / (ax + f)
        };
        if x < 0.0 {
            -r
        } else {
            r
        }
    }

    #[test]
    fn gelu_vs_scalar() {
        let xs: Vec<f32> = (0..64).map(|i| i as f32 * 0.25 - 8.0).collect();
        let mut got = xs.clone();
        unsafe { gelu(got.as_mut_ptr(), got.len()) };
        for (x, g) in xs.iter().zip(&got) {
            let want = 0.5 * x * (1.0 + libm_erf(*x as f64 / core::f64::consts::SQRT_2) as f32);
            assert!((g - want).abs() <= 1e-6, "gelu {x}: {g} vs {want}");
        }
    }

    #[test]
    fn layernorm_vs_scalar() {
        let (m, h) = (5usize, 256usize);
        let mut seed = 11u64;
        let x: Vec<f32> = (0..m * h).map(|_| lcg(&mut seed) * 3.0).collect();
        let w: Vec<f32> = (0..h).map(|_| lcg(&mut seed)).collect();
        let b: Vec<f32> = (0..h).map(|_| lcg(&mut seed)).collect();
        let mut got = x.clone();
        unsafe { layernorm(got.as_mut_ptr(), m, h, w.as_ptr(), b.as_ptr()) };
        for i in 0..m {
            let row = &x[i * h..(i + 1) * h];
            let mean: f64 = row.iter().map(|v| *v as f64).sum::<f64>() / h as f64;
            let var: f64 = row.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / h as f64;
            for j in 0..h {
                let want = ((row[j] as f64 - mean) / (var + 1e-12).sqrt() * w[j] as f64
                    + b[j] as f64) as f32;
                assert!((got[i * h + j] - want).abs() <= 2e-5, "ln [{i},{j}]");
            }
        }
    }

    #[test]
    fn softmax_vs_scalar_with_mask_and_padding() {
        for &(n, ld) in &[
            (1usize, 8usize),
            (3, 8),
            (8, 8),
            (13, 16),
            (384, 384),
            (383, 384),
        ] {
            let mut seed = 5u64;
            let m = 3;
            let x: Vec<f32> = (0..m * ld).map(|_| lcg(&mut seed) * 5.0).collect();
            let mut maskf = vec![f32::NEG_INFINITY; ld];
            for (j, mf) in maskf.iter_mut().enumerate().take(n) {
                *mf = if j % 7 == 3 && n > 4 {
                    f32::NEG_INFINITY
                } else {
                    0.0
                };
            }
            let mut got = x.clone();
            unsafe { softmax_rows(got.as_mut_ptr(), m, ld, maskf.as_ptr()) };
            for i in 0..m {
                let row = &x[i * ld..(i + 1) * ld];
                let live: Vec<usize> = (0..ld).filter(|j| maskf[*j] == 0.0).collect();
                let mx = live
                    .iter()
                    .map(|j| row[*j] as f64)
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = live.iter().map(|j| (row[*j] as f64 - mx).exp()).sum();
                for j in 0..ld {
                    let want = if maskf[j] == 0.0 {
                        ((row[j] as f64 - mx).exp() / sum) as f32
                    } else {
                        0.0
                    };
                    let g = got[i * ld + j];
                    if want == 0.0 {
                        assert_eq!(g, 0.0, "n{n} ld{ld} [{i},{j}] should be exactly 0");
                    } else {
                        assert!(
                            (g - want).abs() <= 2e-6,
                            "n{n} ld{ld} [{i},{j}] {g} vs {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn v4_helpers() {
        assert_eq!(v4([1.0, 2.0, 3.0, 4.0]).hsum(), 10.0);
        assert_eq!(v4([1.0, 9.0, 3.0, 4.0]).hmax(), 9.0);
        assert_eq!(
            arr(v4([1.5, -2.5, 0.5, 2.5]).nearest()),
            [2.0, -2.0, 0.0, 2.0]
        );
        assert_eq!(arr(V4::splat(3.0).pow2_from_int()), [8.0; 4]);
    }
}
