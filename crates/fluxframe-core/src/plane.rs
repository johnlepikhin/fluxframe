//! Plane views and `*Effect` traits for the segmented-composite pipeline.
//!
//! Stage 9 splits the monolithic `background_blur` effect into three
//! sub-pipelines that run inside one top-level [`crate::VideoEffect`]:
//!
//! 1. **Mask pipeline**: inference produces a confidence mask, then a
//!    chain of [`MaskEffect`]s post-processes it (threshold, dilate,
//!    feather, temporal smoothing, …).
//! 2. **Background pipeline**: a chain of [`PlaneEffect`]s transforms a
//!    copy of the original frame into the "background" plane (blur,
//!    solid colour fill, image substitute, …).
//! 3. **Foreground pipeline**: another [`PlaneEffect`] chain produces
//!    the "foreground" plane (default = identity over the original).
//!
//! Mask and plane effects intentionally use different traits because
//! their value types differ (`f32` confidence vs `u8` RGB), and a
//! shared generic trait would push every implementation through
//! associated types or a wrapper enum without buying anything.

use crate::context::{FrameContext, ProcessingContext};
use crate::error::EffectError;
use crate::traits::RawEffectParams;

/// Mutable view on a mask plane: `f32` confidence values in `[0, 1]`
/// laid out as `height * width` in row-major order.
///
/// The struct is a thin borrow over an existing buffer; effects mutate
/// `data` in place. Each effect either rewrites every value or leaves
/// the plane intact — they do not resize.
///
/// `#[non_exhaustive]` blocks external struct-literal construction so
/// the `width * height == data.len()` invariant can only be established
/// through [`MaskPlane::new`]. In-crate code may still construct the
/// struct directly when convenient.
#[derive(Debug)]
#[non_exhaustive]
pub struct MaskPlane<'a> {
    /// Confidence values. Length is `(width as usize) * (height as usize)`.
    pub data: &'a mut [f32],
    /// Plane width in pixels.
    pub width: u32,
    /// Plane height in pixels.
    pub height: u32,
}

impl<'a> MaskPlane<'a> {
    /// Borrow `data` as a mask plane with the given dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height`.
    #[must_use]
    pub fn new(data: &'a mut [f32], width: u32, height: u32) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize),
            "mask plane length must equal width * height",
        );
        Self {
            data,
            width,
            height,
        }
    }
}

/// Mutable view on a packed RGB frame plane: three `u8` channels per
/// pixel laid out as `height * width * 3` in row-major order.
///
/// Like [`MaskPlane`] this is a thin borrow over an existing buffer.
/// Effects mutate `data` in place; the dimensions never change inside
/// the pipeline (resize is a separate effect that the design does not
/// yet expose).
///
/// `#[non_exhaustive]` blocks external struct-literal construction so
/// the `width * height * 3 == data.len()` invariant can only be
/// established through [`FramePlane::new`]. In-crate code may still
/// construct the struct directly when convenient.
#[derive(Debug)]
#[non_exhaustive]
pub struct FramePlane<'a> {
    /// Packed RGB pixels. Length is `(width as usize) * (height as usize) * 3`.
    pub data: &'a mut [u8],
    /// Plane width in pixels.
    pub width: u32,
    /// Plane height in pixels.
    pub height: u32,
}

impl<'a> FramePlane<'a> {
    /// Borrow `data` as an RGB frame plane with the given dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height * 3`.
    #[must_use]
    pub fn new(data: &'a mut [u8], width: u32, height: u32) -> Self {
        assert_eq!(
            data.len(),
            (width as usize) * (height as usize) * 3,
            "frame plane length must equal width * height * 3 (RGB)",
        );
        Self {
            data,
            width,
            height,
        }
    }
}

/// Effect operating on a [`MaskPlane`].
///
/// Used inside the mask sub-pipeline of the composite effect: each
/// instance reads + mutates the mask in place. Implementations
/// typically wrap a primitive in `fluxframe-effects::processing`
/// (threshold, dilate, feather, EMA smoothing, …).
///
/// Lifecycle mirrors [`crate::VideoEffect`]: `configure` → `prepare` →
/// repeated `process` → `shutdown`.
///
/// # Threading
///
/// `Send` but not `Sync` — the composite owns the chain on the same
/// worker thread as the rest of the effect.
pub trait MaskEffect: Send {
    /// Stable identifier used by the registry and config (`snake_case`).
    fn name(&self) -> &'static str;

    /// Apply user-supplied configuration. May be a no-op for stateless
    /// effects (`invert`).
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::InvalidConfig`] if parsing fails or the
    /// values are out of range.
    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError>;

    /// Allocate scratch and load resources once the pipeline dimensions
    /// are known. The supplied [`ProcessingContext`] carries the
    /// *frame*-side dimensions; the mask resolution is whatever the
    /// preceding stage produces and is communicated via the
    /// [`MaskPlane`] passed to [`Self::process`].
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::PrepareFailed`] on resource-acquisition
    /// failure.
    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError>;

    /// Apply the effect to `mask` in place. Must not block on I/O.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::ProcessFailed`] if the input plane is
    /// inconsistent with the prepared state (e.g. a stateful smoother
    /// allocated for a different resolution).
    fn process(
        &mut self,
        mask: &mut MaskPlane<'_>,
        context: &mut FrameContext,
    ) -> Result<(), EffectError>;

    /// Release runtime resources. Default is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError`] if teardown fails; the runtime logs but
    /// continues.
    fn shutdown(&mut self) -> Result<(), EffectError> {
        Ok(())
    }
}

/// Effect operating on a [`FramePlane`].
///
/// Used inside the background and foreground sub-pipelines of the
/// composite effect. Implementations typically wrap an existing
/// image-processing primitive (box blur, solid fill, image overlay,
/// …) and operate on the plane in place — either truly in place
/// (colour fill) or through scratch + ping-pong (blur).
///
/// Lifecycle and threading mirror [`MaskEffect`].
pub trait PlaneEffect: Send {
    /// Stable identifier used by the registry and config (`snake_case`).
    fn name(&self) -> &'static str;

    /// Apply user-supplied configuration.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::InvalidConfig`] on a malformed payload.
    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError>;

    /// Allocate scratch and load resources at the negotiated
    /// resolution. Plane effects always run at frame resolution — the
    /// composite hands the original frame's `(width, height)` through
    /// `context`.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::PrepareFailed`] on resource acquisition
    /// failure.
    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError>;

    /// Transform `plane` in place. Must not block on I/O.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::ProcessFailed`] on a transient backend
    /// failure (e.g. a GPU blur backend that lost its device).
    fn process(
        &mut self,
        plane: &mut FramePlane<'_>,
        context: &mut FrameContext,
    ) -> Result<(), EffectError>;

    /// Release runtime resources. Default is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError`] if teardown fails.
    fn shutdown(&mut self) -> Result<(), EffectError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_plane_new_round_trips_dimensions() {
        let mut buf = vec![0.0_f32; 6];
        let plane = MaskPlane::new(&mut buf, 3, 2);
        assert_eq!(plane.width, 3);
        assert_eq!(plane.height, 2);
        assert_eq!(plane.data.len(), 6);
    }

    #[test]
    fn frame_plane_new_round_trips_dimensions() {
        let mut buf = vec![0_u8; 12];
        let plane = FramePlane::new(&mut buf, 2, 2);
        assert_eq!(plane.width, 2);
        assert_eq!(plane.height, 2);
        assert_eq!(plane.data.len(), 12);
    }

    #[test]
    #[should_panic(expected = "mask plane length")]
    fn mask_plane_asserts_length() {
        let mut buf = vec![0.0_f32; 5];
        let _ = MaskPlane::new(&mut buf, 3, 2);
    }

    #[test]
    #[should_panic(expected = "frame plane length")]
    fn frame_plane_asserts_length() {
        let mut buf = vec![0_u8; 11];
        let _ = FramePlane::new(&mut buf, 2, 2);
    }
}
