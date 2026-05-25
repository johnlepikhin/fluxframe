//! Blur backend trait + CPU implementation.
//!
//! [`BlurBackend`] is the seam through which `BackgroundBlurEffect`
//! delegates the box-blur stage of its compositing pipeline.  A future
//! GPU-side implementation slots in by implementing the same trait —
//! the effect itself stays unchanged.
//!
//! Scratch ownership rule: each backend owns whatever scratch buffers
//! it needs (allocated in [`BlurBackend::prepare`] and reused across
//! [`BlurBackend::blur`] calls).  Callers MUST NOT pass scratch in;
//! a GPU backend would otherwise have to maintain a CPU scratch slice
//! it never touches.
//!
//! Failure surface: both `prepare` and `blur` return [`EffectError`]
//! — `PrepareFailed` for setup errors (allocation, device init) and
//! `ProcessFailed` for per-frame errors (shape mismatch, runtime
//! driver error).  The CPU implementation cannot fail at runtime in
//! practice, but the `Result` is kept so the trait stays uniform
//! across future backends.

use fluxframe_core::error::EffectError;

use crate::processing::blur::box_blur_rgb;

/// Pluggable separable box-blur backend for packed RGB buffers.
///
/// Lifecycle:
/// 1. Construct (no allocation yet — `new`-style constructors).
/// 2. Call [`Self::prepare`] once with the frame dimensions.  The
///    backend allocates whatever scratch / device buffers it needs.
/// 3. Call [`Self::blur`] per frame.  `src` and `dst` MUST be exactly
///    `width * height * 3` bytes and the dimensions MUST match what
///    was passed to `prepare`.
/// 4. To resize at runtime: call `prepare` again with the new
///    dimensions.  The backend is free to reallocate.
///
/// `Send` only (not `Sync`): GPU contexts (Vulkan queues, wgpu
/// devices) are typically not `Sync`.  The effect chain runs on a
/// single thread, so this matches both the consumer and the future
/// implementor.
pub trait BlurBackend: Send {
    /// Stable identifier used in logs and metrics — `"cpu"` for the
    /// reference implementation, e.g. `"wgpu"` / `"gst-gl"` for future
    /// GPU backends.  Lowercase, no whitespace.
    fn name(&self) -> &'static str;

    /// Allocate (or reallocate) scratch for the given frame size.
    /// MUST be called before [`Self::blur`].  Calling again with new
    /// dimensions is allowed and resizes the scratch.
    ///
    /// # Errors
    ///
    /// [`EffectError::PrepareFailed`] for allocation / device-init
    /// failures.  The CPU implementation cannot fail in practice; the
    /// fallible return keeps the API uniform with future GPU backends.
    fn prepare(&mut self, width: u32, height: u32) -> Result<(), EffectError>;

    /// Apply a separable box blur to a packed RGB frame.
    ///
    /// `src` and `dst` MUST both be exactly `width * height * 3`
    /// bytes; `width` / `height` MUST equal the values passed to the
    /// last successful [`Self::prepare`].  `passes` ≥ 1 approximates a
    /// Gaussian by chaining box passes; `passes == 0` returns `src`
    /// verbatim (matches the underlying primitive).
    ///
    /// # Errors
    ///
    /// [`EffectError::ProcessFailed`] for shape mismatches or runtime
    /// backend errors.
    fn blur(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        height: u32,
        radius: u32,
        passes: u32,
    ) -> Result<(), EffectError>;
}

/// CPU reference implementation of [`BlurBackend`].
///
/// Thin wrapper around [`crate::processing::blur::box_blur_rgb`]: holds
/// the intermediate scratch buffer that the separable box-blur
/// primitive needs and forwards every call.  Allocation happens once
/// in [`Self::prepare`]; the per-frame [`Self::blur`] path is
/// allocation-free.
///
/// This is the only `BlurBackend` that ships in Stage 6.  It exists
/// so callers can already program against the trait, and so the
/// future GPU backend(s) have a stable secondary to fall back to.
#[derive(Debug, Default)]
pub struct CpuBlurBackend {
    scratch: Vec<u8>,
    prepared_width: u32,
    prepared_height: u32,
}

impl CpuBlurBackend {
    /// Construct an empty backend.  Buffers are allocated on the first
    /// [`BlurBackend::prepare`] call; the constructor itself does no
    /// allocation, matching the deferred-init style of the rest of
    /// the effects layer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            scratch: Vec::new(),
            prepared_width: 0,
            prepared_height: 0,
        }
    }
}

/// Build the canonical "backend name" component for error messages so
/// the message reads consistently with the rest of `background_blur`
/// (which reports the effect name as the carrier).  The trait does
/// not know which effect it's running inside, so we tag the source
/// generically.
const COMPONENT: &str = "blur backend (cpu)";

impl BlurBackend for CpuBlurBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn prepare(&mut self, width: u32, height: u32) -> Result<(), EffectError> {
        // Box-blur primitive panics on a 1×N (or N×1) input because the
        // sliding-window initialisation reads `w-1` / `h-1`.  Reject
        // here so the failure mode is a structured error rather than a
        // panic from inside the primitive on the first frame.
        if width < 2 || height < 2 {
            return Err(EffectError::PrepareFailed {
                name: COMPONENT.into(),
                reason: format!("frame {width}x{height} too small; box blur needs at least 2x2"),
            });
        }
        let bytes = (width as usize)
            .checked_mul(height as usize)
            .and_then(|p| p.checked_mul(3))
            .ok_or_else(|| EffectError::PrepareFailed {
                name: COMPONENT.into(),
                reason: format!("frame {width}x{height} overflows usize"),
            })?;
        // `resize` shrinks or grows in place: reuses existing capacity
        // when the backend is re-prepared at the same or smaller size,
        // and only allocates when growing past current capacity.
        self.scratch.resize(bytes, 0);
        self.prepared_width = width;
        self.prepared_height = height;
        Ok(())
    }

    fn blur(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        height: u32,
        radius: u32,
        passes: u32,
    ) -> Result<(), EffectError> {
        if width != self.prepared_width || height != self.prepared_height {
            return Err(EffectError::ProcessFailed {
                name: COMPONENT.into(),
                reason: format!(
                    "frame {width}x{height} differs from prepared {}x{}",
                    self.prepared_width, self.prepared_height
                ),
            });
        }
        let expected = (width as usize) * (height as usize) * 3;
        if src.len() != expected || dst.len() != expected {
            return Err(EffectError::ProcessFailed {
                name: COMPONENT.into(),
                reason: format!(
                    "buffer length mismatch: src={}, dst={}, expected={expected}",
                    src.len(),
                    dst.len()
                ),
            });
        }
        box_blur_rgb(src, dst, &mut self.scratch, width, height, radius, passes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_stable() {
        let b = CpuBlurBackend::new();
        assert_eq!(b.name(), "cpu");
    }

    #[test]
    fn prepare_rejects_too_small_frame() {
        let mut b = CpuBlurBackend::new();
        assert!(b.prepare(1, 4).is_err());
        assert!(b.prepare(4, 1).is_err());
        assert!(b.prepare(0, 0).is_err());
    }

    #[test]
    fn prepare_then_blur_matches_direct_call() {
        // Regression pin: CpuBlurBackend MUST produce byte-for-byte the
        // same result as a direct `box_blur_rgb` invocation, otherwise
        // the Stage 6 refactor would silently change pixel output.
        let w = 8u32;
        let h = 4u32;
        let mut src = vec![0u8; (w * h * 3) as usize];
        // Non-uniform pattern so the blur actually changes pixels.
        for (i, b) in src.iter_mut().enumerate() {
            *b = ((i * 7) % 251) as u8;
        }
        let mut dst_backend = vec![0u8; src.len()];
        let mut dst_direct = vec![0u8; src.len()];
        let mut scratch_direct = vec![0u8; src.len()];

        let mut backend = CpuBlurBackend::new();
        backend.prepare(w, h).expect("prepare");
        backend
            .blur(&src, &mut dst_backend, w, h, 2, 1)
            .expect("blur");
        box_blur_rgb(&src, &mut dst_direct, &mut scratch_direct, w, h, 2, 1);

        assert_eq!(
            dst_backend, dst_direct,
            "backend output must match direct primitive output bit-for-bit"
        );
    }

    #[test]
    fn blur_rejects_dimension_mismatch() {
        let mut b = CpuBlurBackend::new();
        b.prepare(8, 8).expect("prepare");
        let src = vec![0u8; 8 * 8 * 3];
        let mut dst = vec![0u8; 8 * 8 * 3];
        // Lie about dimensions — must be caught.
        let err = b.blur(&src, &mut dst, 4, 8, 1, 1).expect_err("must fail");
        assert!(format!("{err}").contains("differs from prepared"));
    }

    #[test]
    fn blur_rejects_buffer_length_mismatch() {
        let mut b = CpuBlurBackend::new();
        b.prepare(4, 4).expect("prepare");
        let src = vec![0u8; 4 * 4 * 3];
        let mut dst_short = vec![0u8; 4 * 4 * 3 - 1];
        let err = b
            .blur(&src, &mut dst_short, 4, 4, 1, 1)
            .expect_err("must fail");
        assert!(format!("{err}").contains("buffer length"));
    }

    #[test]
    fn reprepare_resizes_scratch() {
        let mut b = CpuBlurBackend::new();
        b.prepare(4, 4).expect("prepare 4x4");
        b.prepare(8, 6).expect("prepare 8x6");
        // After re-prepare with larger size the new blur must succeed.
        let src = vec![0u8; 8 * 6 * 3];
        let mut dst = vec![0u8; 8 * 6 * 3];
        b.blur(&src, &mut dst, 8, 6, 1, 1).expect("blur 8x6");
    }

    #[test]
    fn cpu_backend_is_send() {
        // Trait bound is `Send`; verify at compile time via the static
        // assertion that `Box<dyn BlurBackend>` carries `Send`.
        fn assert_send<T: Send>(_: &T) {}
        let b: Box<dyn BlurBackend> = Box::new(CpuBlurBackend::new());
        assert_send(&b);
    }
}
