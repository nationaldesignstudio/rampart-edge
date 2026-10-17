//! Four-lane f32 vector used by every kernel. On wasm32 it is one `v128` and every method is a single
//! simd128 instruction; on other targets it is four scalars doing the same IEEE operation per lane, so the
//! host-side `cargo test` exercises the identical arithmetic (same operations in the same order, no FMA,
//! no reassociation) and produces bit-identical results.

#[cfg(target_arch = "wasm32")]
mod imp {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct V4(pub v128);

    #[allow(clippy::should_implement_trait)]
    impl V4 {
        #[inline(always)]
        pub fn splat(x: f32) -> Self {
            V4(f32x4_splat(x))
        }
        #[inline(always)]
        pub fn zero() -> Self {
            V4(f32x4_splat(0.0))
        }
        /// # Safety
        /// `p` must point at four readable `f32`s (alignment is not required).
        #[inline(always)]
        pub unsafe fn load(p: *const f32) -> Self {
            V4(v128_load(p as *const v128))
        }
        /// # Safety
        /// `p` must point at one readable `f32`.
        #[inline(always)]
        pub unsafe fn load_splat(p: *const f32) -> Self {
            V4(v128_load32_splat(p as *const u32))
        }
        /// # Safety
        /// `p` must point at four writable `f32`s.
        #[inline(always)]
        pub unsafe fn store(self, p: *mut f32) {
            v128_store(p as *mut v128, self.0)
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            V4(f32x4_add(self.0, o.0))
        }
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            V4(f32x4_sub(self.0, o.0))
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            V4(f32x4_mul(self.0, o.0))
        }
        #[inline(always)]
        pub fn div(self, o: Self) -> Self {
            V4(f32x4_div(self.0, o.0))
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            V4(f32x4_max(self.0, o.0))
        }
        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            V4(f32x4_min(self.0, o.0))
        }
        #[inline(always)]
        pub fn abs(self) -> Self {
            V4(f32x4_abs(self.0))
        }
        #[inline(always)]
        pub fn neg(self) -> Self {
            V4(f32x4_neg(self.0))
        }
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            V4(f32x4_sqrt(self.0))
        }
        /// Round to nearest, ties to even (`f32x4.nearest`).
        #[inline(always)]
        pub fn nearest(self) -> Self {
            V4(f32x4_nearest(self.0))
        }
        /// Lane-wise `self > o` as an all-ones / all-zeros mask.
        #[inline(always)]
        pub fn gt(self, o: Self) -> Self {
            V4(f32x4_gt(self.0, o.0))
        }
        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            V4(v128_and(self.0, o.0))
        }
        #[inline(always)]
        pub fn or(self, o: Self) -> Self {
            V4(v128_or(self.0, o.0))
        }
        /// Lane-wise `2^n` for integer-valued `n` already clamped to [-126, 127]: builds the exponent bits.
        #[inline(always)]
        pub fn pow2_from_int(self) -> Self {
            let e = i32x4_shl(
                i32x4_add(i32x4_trunc_sat_f32x4(self.0), i32x4_splat(127)),
                23,
            );
            V4(e)
        }
        /// Sign bits only (lane-wise `x & 0x8000_0000`).
        #[inline(always)]
        pub fn sign_bits(self) -> Self {
            V4(v128_and(self.0, i32x4_splat(i32::MIN)))
        }
        #[inline(always)]
        pub fn lane0(self) -> f32 {
            f32x4_extract_lane::<0>(self.0)
        }
        /// Horizontal sum in the fixed order ((l0+l2)+(l1+l3)).
        #[inline(always)]
        pub fn hsum(self) -> f32 {
            let s = f32x4_add(self.0, i32x4_shuffle::<2, 3, 0, 1>(self.0, self.0));
            let s = f32x4_add(s, i32x4_shuffle::<1, 0, 3, 2>(s, s));
            f32x4_extract_lane::<0>(s)
        }
        /// Horizontal max in the same order as `hsum`.
        #[inline(always)]
        pub fn hmax(self) -> f32 {
            let s = f32x4_max(self.0, i32x4_shuffle::<2, 3, 0, 1>(self.0, self.0));
            let s = f32x4_max(s, i32x4_shuffle::<1, 0, 3, 2>(s, s));
            f32x4_extract_lane::<0>(s)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct V4(pub [f32; 4]);

    #[inline(always)]
    fn map2(a: [f32; 4], b: [f32; 4], f: impl Fn(f32, f32) -> f32) -> [f32; 4] {
        [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
    }
    #[inline(always)]
    fn map1(a: [f32; 4], f: impl Fn(f32) -> f32) -> [f32; 4] {
        [f(a[0]), f(a[1]), f(a[2]), f(a[3])]
    }
    #[inline(always)]
    fn bits2(a: [f32; 4], b: [f32; 4], f: impl Fn(u32, u32) -> u32) -> [f32; 4] {
        map2(a, b, |x, y| f32::from_bits(f(x.to_bits(), y.to_bits())))
    }
    /// wasm `f32x4.max`/`min`: NaN if either lane is NaN, `-0 < +0`.
    #[inline(always)]
    fn wmax(x: f32, y: f32) -> f32 {
        if x.is_nan() || y.is_nan() {
            f32::NAN
        } else if x == y {
            if x.is_sign_negative() {
                y
            } else {
                x
            }
        } else if x > y {
            x
        } else {
            y
        }
    }
    #[inline(always)]
    fn wmin(x: f32, y: f32) -> f32 {
        if x.is_nan() || y.is_nan() {
            f32::NAN
        } else if x == y {
            if x.is_sign_negative() {
                x
            } else {
                y
            }
        } else if x < y {
            x
        } else {
            y
        }
    }

    #[allow(clippy::should_implement_trait)]
    impl V4 {
        #[inline(always)]
        pub fn splat(x: f32) -> Self {
            V4([x; 4])
        }
        #[inline(always)]
        pub fn zero() -> Self {
            V4([0.0; 4])
        }
        /// # Safety
        /// `p` must point at four readable `f32`s.
        #[inline(always)]
        pub unsafe fn load(p: *const f32) -> Self {
            V4(core::ptr::read_unaligned(p as *const [f32; 4]))
        }
        /// # Safety
        /// `p` must point at one readable `f32`.
        #[inline(always)]
        pub unsafe fn load_splat(p: *const f32) -> Self {
            V4([*p; 4])
        }
        /// # Safety
        /// `p` must point at four writable `f32`s.
        #[inline(always)]
        pub unsafe fn store(self, p: *mut f32) {
            core::ptr::write_unaligned(p as *mut [f32; 4], self.0)
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            V4(map2(self.0, o.0, |x, y| x + y))
        }
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            V4(map2(self.0, o.0, |x, y| x - y))
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            V4(map2(self.0, o.0, |x, y| x * y))
        }
        #[inline(always)]
        pub fn div(self, o: Self) -> Self {
            V4(map2(self.0, o.0, |x, y| x / y))
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            V4(map2(self.0, o.0, wmax))
        }
        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            V4(map2(self.0, o.0, wmin))
        }
        #[inline(always)]
        pub fn abs(self) -> Self {
            V4(map1(self.0, f32::abs))
        }
        #[inline(always)]
        pub fn neg(self) -> Self {
            V4(map1(self.0, |x| -x))
        }
        #[inline(always)]
        pub fn sqrt(self) -> Self {
            V4(map1(self.0, f32::sqrt))
        }
        #[inline(always)]
        pub fn nearest(self) -> Self {
            V4(map1(self.0, f32::round_ties_even))
        }
        #[inline(always)]
        pub fn gt(self, o: Self) -> Self {
            V4(map2(self.0, o.0, |x, y| {
                f32::from_bits(if x > y { u32::MAX } else { 0 })
            }))
        }
        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            V4(bits2(self.0, o.0, |x, y| x & y))
        }
        #[inline(always)]
        pub fn or(self, o: Self) -> Self {
            V4(bits2(self.0, o.0, |x, y| x | y))
        }
        #[inline(always)]
        pub fn pow2_from_int(self) -> Self {
            V4(map1(self.0, |n| {
                let e = (n as i32).wrapping_add(127).wrapping_shl(23);
                f32::from_bits(e as u32)
            }))
        }
        #[inline(always)]
        pub fn sign_bits(self) -> Self {
            V4(map1(self.0, |x| f32::from_bits(x.to_bits() & 0x8000_0000)))
        }
        #[inline(always)]
        pub fn lane0(self) -> f32 {
            self.0[0]
        }
        #[inline(always)]
        pub fn hsum(self) -> f32 {
            let [a, b, c, d] = self.0;
            (a + c) + (b + d)
        }
        #[inline(always)]
        pub fn hmax(self) -> f32 {
            let [a, b, c, d] = self.0;
            wmax(wmax(a, c), wmax(b, d))
        }
    }
}

pub use imp::V4;
