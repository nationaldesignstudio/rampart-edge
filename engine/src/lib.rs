//! Rampart Edge inference engine: fp32 forward of the 4-bit PII gate model (`model_q4.onnx` semantics) as a
//! dependency-free `wasm32-unknown-unknown` module with simd128 only. Imports nothing; exports
//! `memory`, `arena_alloc`, `init`, `ids_ptr`, `mask_ptr`, `tok_ptr`, `forward`, `version`.
//!
//! Host protocol (see `src/engine.ts`):
//! 1. `p = arena_alloc(len)`; copy the `.rmpw` file to `memory[p..p+len]`.
//! 2. `init(p, len)` -> 0, or an error code from [`format::code`].
//! 3. Per call: write `n` i32 ids to `ids_ptr()` and `n` i32 mask values to `mask_ptr()`; `forward(n)` returns the
//!    gate logit (NaN on bad `n`, bad id or non-finite result); `n x 19` token logits are at `tok_ptr()`.
//!
//! The module is single-threaded by construction (no shared memory, no atomics); the globals below are only
//! ever touched from the one wasm thread that owns the instance.
#![cfg_attr(all(not(test), target_arch = "wasm32"), no_std)]
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

pub mod format;
pub mod kernels;
pub mod model;
pub mod sha256;
pub mod simd;

use core::cell::UnsafeCell;
use model::{Arena, Model, Scratch};

/// Engine version reported by `version()`: major << 16 | minor << 8 | patch.
pub const VERSION: u32 = 3 << 16;

/// Arena size: 1.33 MB weight file + ~3.9 MB expanded weights + ~4.9 MB scratch at n = 384, with headroom.
pub const ARENA_BYTES: usize = 12 * 1024 * 1024;

#[repr(C, align(64))]
struct ArenaBuf(UnsafeCell<[u8; ARENA_BYTES]>);
// SAFETY: wasm32-unknown-unknown has no threads; the host test suite serialises access to the globals.
unsafe impl Sync for ArenaBuf {}
static ARENA_BUF: ArenaBuf = ArenaBuf(UnsafeCell::new([0; ARENA_BYTES]));

struct State {
    top: usize,
    model: Option<(Model, Scratch)>,
}
struct Globals(UnsafeCell<State>);
// SAFETY: as above.
unsafe impl Sync for Globals {}
static STATE: Globals = Globals(UnsafeCell::new(State {
    top: 0,
    model: None,
}));

#[inline]
fn arena_base() -> *mut u8 {
    ARENA_BUF.0.get().cast::<u8>()
}

/// Reserve `bytes` of arena (64-byte aligned). Returns 0 when the arena is exhausted.
#[no_mangle]
pub extern "C" fn arena_alloc(bytes: usize) -> *mut u8 {
    // SAFETY: single-threaded access to the global state.
    let st = unsafe { &mut *STATE.0.get() };
    let mut arena = Arena {
        base: arena_base(),
        cap: ARENA_BYTES,
        top: st.top,
    };
    match arena.alloc(bytes) {
        Ok(p) => {
            st.top = arena.top;
            p
        }
        Err(_) => core::ptr::null_mut(),
    }
}

/// Validate and expand the `.rmpw` file at `memory[ptr..ptr+len]` (which must have come from `arena_alloc`).
/// Returns 0 on success or a [`format::code`] value.
#[no_mangle]
pub extern "C" fn init(ptr: usize, len: usize) -> i32 {
    // SAFETY: single-threaded access to the global state.
    let st = unsafe { &mut *STATE.0.get() };
    if st.model.is_some() {
        return format::code::ALREADY_INIT;
    }
    let base = arena_base() as usize;
    let start = ptr;
    let Some(end) = start.checked_add(len) else {
        return format::code::BAD_BUFFER;
    };
    if start < base || end > base + st.top || !start.is_multiple_of(4) {
        return format::code::BAD_BUFFER;
    }
    // SAFETY: the range lies inside the arena region handed out by `arena_alloc`.
    let file: &[u8] = unsafe { core::slice::from_raw_parts(ptr as *const u8, len) };
    let header = match format::parse(file) {
        Ok(h) => h,
        Err(c) => return c,
    };
    if let Err(c) = format::check_payload(file, &header) {
        return c;
    }
    let mut arena = Arena {
        base: arena_base(),
        cap: ARENA_BYTES,
        top: st.top,
    };
    // SAFETY: `file` is the buffer `header` was parsed from and stays mapped in the arena.
    match unsafe { model::expand(file, &header, &mut arena) } {
        Ok(ms) => {
            st.top = arena.top;
            st.model = Some(ms);
            format::code::OK
        }
        Err(c) => c,
    }
}

fn scratch() -> Option<&'static mut Scratch> {
    // SAFETY: single-threaded access to the global state.
    unsafe { (*STATE.0.get()).model.as_mut().map(|(_, s)| s) }
}

/// Input ids (`i32`, up to 384). Null before `init`.
#[no_mangle]
pub extern "C" fn ids_ptr() -> *mut i32 {
    scratch().map_or(core::ptr::null_mut(), |s| s.ids)
}

/// Attention mask (`i32`, 0/1, up to 384). Null before `init`.
#[no_mangle]
pub extern "C" fn mask_ptr() -> *mut i32 {
    scratch().map_or(core::ptr::null_mut(), |s| s.mask)
}

/// Token logits of the last `forward`, `n x 19` row-major `f32`. Null before `init`.
#[no_mangle]
pub extern "C" fn tok_ptr() -> *mut f32 {
    scratch().map_or(core::ptr::null_mut(), |s| s.tok_out)
}

/// Run the model on the first `n` ids/mask entries. Returns the gate logit; NaN if not initialised,
/// `n` is outside 1..=384, an id is outside the vocabulary, or the result is not finite.
#[no_mangle]
pub extern "C" fn forward(n: usize) -> f32 {
    // SAFETY: single-threaded access to the global state.
    let st = unsafe { &mut *STATE.0.get() };
    match st.model.as_mut() {
        // SAFETY: model and scratch were produced by `expand`; `forward` checks `n` itself.
        Some((m, s)) => unsafe { model::forward(m, s, n) },
        None => f32::NAN,
    }
}

/// Select the GEMM microkernel: 0 = unrolled (default), 1 = compact. Returns 0 on success, 1 for an
/// unknown value (the setting is left unchanged). Output is bit-identical for both kernels.
#[no_mangle]
pub extern "C" fn set_kernel(kernel: u32) -> u32 {
    if kernel > kernels::KERNEL_COMPACT {
        return 1;
    }
    kernels::KERNEL.store(kernel, core::sync::atomic::Ordering::Relaxed);
    0
}

#[no_mangle]
pub extern "C" fn version() -> u32 {
    VERSION
}

/// Bytes of arena in use (weights file + expanded model + scratch).
#[cfg(feature = "test-exports")]
#[no_mangle]
pub extern "C" fn arena_used() -> usize {
    // SAFETY: single-threaded access to the global state.
    unsafe { (*STATE.0.get()).top }
}

#[cfg(all(not(test), target_arch = "wasm32"))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    // Unreachable by construction: every untrusted value is checked before use. Kept only to satisfy no_std.
    core::arch::wasm32::unreachable()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static LOCK: Mutex<()> = Mutex::new(());

    /// Copy `file` into one shared arena buffer (a failed `init` leaves it reusable) and call `init`.
    fn load(file: &[u8]) -> i32 {
        static BUF: Mutex<usize> = Mutex::new(0);
        let mut slot = BUF.lock().unwrap();
        if *slot == 0 {
            let p = arena_alloc(2_000_000);
            assert!(!p.is_null());
            *slot = p as usize;
        }
        let p = *slot as *mut u8;
        unsafe { core::ptr::copy_nonoverlapping(file.as_ptr(), p, file.len()) };
        init(p as usize, file.len())
    }

    #[test]
    fn lifecycle_on_synthetic_weights() {
        let _g = LOCK.lock().unwrap();
        assert!(ids_ptr().is_null() && mask_ptr().is_null() && tok_ptr().is_null());
        assert!(forward(1).is_nan());
        assert_eq!(version(), VERSION);

        let file = format::synth_file();
        // Buffer outside the arena / past the allocated top -> BAD_BUFFER.
        assert_eq!(
            init(file.as_ptr() as usize, file.len()),
            format::code::BAD_BUFFER
        );
        assert_eq!(
            init(arena_base() as usize, ARENA_BYTES + 1),
            format::code::BAD_BUFFER
        );
        assert_eq!(init(usize::MAX, 16), format::code::BAD_BUFFER);
        // Malformed copies inside the arena report their codes and leave the engine uninitialised.
        let mut bad = file.clone();
        bad[0] ^= 1;
        assert_eq!(load(&bad), format::code::BAD_MAGIC);
        let mut bad = file.clone();
        bad[format::HEADER_BYTES + 100] ^= 1;
        assert_eq!(load(&bad), format::code::BAD_DIGEST);
        assert!(ids_ptr().is_null());
        // Arena exhaustion is reported, not trapped.
        assert!(arena_alloc(usize::MAX).is_null());
        assert!(arena_alloc(ARENA_BYTES).is_null());

        assert_eq!(load(&file), format::code::OK);
        assert_eq!(load(&file), format::code::ALREADY_INIT);
        assert!(!ids_ptr().is_null());
        assert!(forward(0).is_nan());
        assert!(forward(385).is_nan());
        // Out-of-range id -> NaN, never a trap.
        unsafe { *ids_ptr() = format::VOCAB as i32 };
        assert!(forward(1).is_nan());
        unsafe { *ids_ptr() = -1 };
        assert!(forward(1).is_nan());
        // All-zero weights: a finite, deterministic result for every n.
        for n in [1usize, 7, 8, 9, 384] {
            for t in 0..n {
                unsafe {
                    *ids_ptr().add(t) = (t % 100) as i32;
                    *mask_ptr().add(t) = 1;
                }
            }
            let a = forward(n);
            let b = forward(n);
            assert!(a.is_finite(), "n {n}");
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
