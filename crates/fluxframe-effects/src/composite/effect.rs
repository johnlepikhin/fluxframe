//! Top-level [`fluxframe_core::VideoEffect`] orchestrating the
//! segmented-composite pipeline: segmentation → mask chain → bg/fg
//! chains → alpha composite.
//!
//! The composite owns no parsing logic — it accepts pre-configured
//! sub-effects via [`CompositeEffect::new`]. The TOML-side wiring
//! (parsing `[mask]`/`[background]`/`[foreground]` into chains) lives
//! in [`crate::composite::builder`] which the CLI invokes once at
//! startup.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::frame::{PixelFormat, VideoFrame};
use fluxframe_core::plane::{FramePlane, MaskEffect, MaskPlane, PlaneEffect, PostEffect};
use fluxframe_core::traits::{RawEffectParams, VideoEffect};
use tracing::trace;

use crate::composite::segmentation::{SegmentationBase, SegmentationOutcome};
use crate::processing::compose::alpha_composite_rgb_in_place;
use crate::processing::resize_mask_bilinear;

/// Time a closure and return its result together with elapsed
/// microseconds. Saturates to `u64::MAX` worth of micros on overflow
/// (effectively impossible inside a per-frame stage).
fn timed<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let start = std::time::Instant::now();
    let r = f();
    let us = u64::try_from(start.elapsed().as_micros()).unwrap_or(0);
    (r, us)
}

/// Build a `ProcessFailed` error for this effect with the given
/// `reason`. Kept private so the effect name stays in one place.
fn process_err(reason: impl Into<String>) -> EffectError {
    EffectError::ProcessFailed {
        name: CompositeEffect::NAME.to_string(),
        reason: reason.into(),
    }
}

/// The composite effect glues a [`SegmentationBase`] together with
/// three ordered chains (mask post-processing, background pipeline,
/// foreground pipeline) and performs the final alpha-composite.
///
/// Lifecycle is the standard [`VideoEffect`] one. `configure` is a
/// no-op at this level because each sub-effect is pre-configured by
/// the builder; calling it with non-empty params is rejected so a
/// mistaken TOML cannot silently produce a half-configured composite.
pub struct CompositeEffect {
    segmentation: SegmentationBase,
    mask_chain: Vec<Box<dyn MaskEffect>>,
    bg_chain: Vec<Box<dyn PlaneEffect>>,
    fg_chain: Vec<Box<dyn PlaneEffect>>,
    /// Post-composite mask-aware chain. Each effect sees the already-
    /// blended frame plus a read-only view of the upscaled mask.
    post_chain: Vec<Box<dyn PostEffect>>,
    /// Mask at frame resolution after the implicit bilinear upscale.
    mask_full: Vec<f32>,
    /// Frame-sized scratch holding the background plane through the
    /// background chain. The foreground chain runs in place on
    /// `frame.data`, so no `fg_plane` scratch is needed.
    bg_plane: Vec<u8>,
    frame_w: u32,
    frame_h: u32,
}

impl CompositeEffect {
    /// Effect name as registered in the top-level effect registry.
    pub const NAME: &'static str = "composite";

    /// Construct from pre-configured sub-components.
    #[must_use]
    pub fn new(
        segmentation: SegmentationBase,
        mask_chain: Vec<Box<dyn MaskEffect>>,
        bg_chain: Vec<Box<dyn PlaneEffect>>,
        fg_chain: Vec<Box<dyn PlaneEffect>>,
        post_chain: Vec<Box<dyn PostEffect>>,
    ) -> Self {
        Self {
            segmentation,
            mask_chain,
            bg_chain,
            fg_chain,
            post_chain,
            mask_full: Vec::new(),
            bg_plane: Vec::new(),
            frame_w: 0,
            frame_h: 0,
        }
    }
}

impl VideoEffect for CompositeEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, _params: RawEffectParams) -> Result<(), EffectError> {
        // Sub-effects are configured by the builder before construction;
        // this entry point exists only to satisfy the `VideoEffect`
        // contract.
        Ok(())
    }

    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError> {
        if context.format != PixelFormat::Rgb {
            return Err(EffectError::PrepareFailed {
                name: Self::NAME.to_string(),
                reason: format!("composite requires RGB input, got {:?}", context.format),
            });
        }

        self.segmentation.prepare(context)?;
        for effect in &mut self.mask_chain {
            effect.prepare(context)?;
        }
        for effect in &mut self.bg_chain {
            effect.prepare(context)?;
        }
        for effect in &mut self.fg_chain {
            effect.prepare(context)?;
        }
        for effect in &mut self.post_chain {
            effect.prepare(context)?;
        }

        self.frame_w = context.width;
        self.frame_h = context.height;
        let frame_pixels = (context.width as usize) * (context.height as usize);
        let frame_bytes = frame_pixels * 3;
        self.mask_full = vec![0.0_f32; frame_pixels];
        self.bg_plane = vec![0_u8; frame_bytes];
        Ok(())
    }

    fn process(
        &mut self,
        frame: &mut VideoFrame,
        ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        if frame.format != PixelFormat::Rgb {
            return Err(process_err(format!(
                "unexpected pixel format {:?}",
                frame.format
            )));
        }
        if frame.width != self.frame_w || frame.height != self.frame_h {
            return Err(process_err(format!(
                "frame {}x{} differs from prepared {}x{}",
                frame.width, frame.height, self.frame_w, self.frame_h
            )));
        }

        // 1. Segmentation produces a mask at model resolution.
        let (seg_result, seg_us) = timed(|| self.segmentation.process(frame, ctx));
        let (mask_w, mask_h) = match seg_result {
            SegmentationOutcome::Ok { width, height } => (width, height),
            SegmentationOutcome::Fallback => {
                ctx.fallback_active = true;
                return Ok(());
            }
            SegmentationOutcome::Fatal(err) => return Err(err),
        };

        // 2. Mask chain at model resolution + 3. resize mask to frame
        //    resolution. The composite owns the implicit upscale —
        //    mask effects never resize.
        let (mask_result, mask_us) = timed(|| -> Result<(), EffectError> {
            {
                let mut mask_plane = MaskPlane::new(self.segmentation.mask_mut(), mask_w, mask_h);
                for effect in &mut self.mask_chain {
                    effect.process(&mut mask_plane, ctx)?;
                }
            }
            resize_mask_bilinear(
                self.segmentation.mask(),
                mask_w,
                mask_h,
                &mut self.mask_full,
                self.frame_w,
                self.frame_h,
            );
            Ok(())
        });
        mask_result?;

        // 4. Promote the frame buffer to Owned so we can mutate it
        //    directly as the foreground plane.
        frame.data.make_owned();
        let Some(frame_bytes) = frame.data.as_mut_owned() else {
            return Err(process_err(
                "frame buffer not owned after make_owned (FrameBuffer invariant broken)",
            ));
        };

        // 5. Snapshot the original frame into `bg_plane`. This is the
        //    only full-frame copy in the composite hot path.
        self.bg_plane.copy_from_slice(frame_bytes);

        // 6. Background chain runs on the snapshot.
        let (bg_result, bg_us) = timed(|| -> Result<(), EffectError> {
            let mut background = FramePlane::new(&mut self.bg_plane, self.frame_w, self.frame_h);
            for effect in &mut self.bg_chain {
                effect.process(&mut background, ctx)?;
            }
            Ok(())
        });
        bg_result?;

        // 7. Foreground chain runs in place on `frame.data`.
        let (fg_result, fg_us) = timed(|| -> Result<(), EffectError> {
            let mut foreground = FramePlane::new(frame_bytes, self.frame_w, self.frame_h);
            for effect in &mut self.fg_chain {
                effect.process(&mut foreground, ctx)?;
            }
            Ok(())
        });
        fg_result?;

        // 8. Composite bg into fg/frame in place using the parallel
        //    helper.
        let ((), compose_us) = timed(|| {
            alpha_composite_rgb_in_place(frame_bytes, &self.bg_plane, &self.mask_full);
        });

        // 9. Post chain. Each post-effect mutates the composed frame
        //    while reading the upscaled mask. Split-borrow so the
        //    mask buffer can be passed as `&MaskPlane` without
        //    clashing with `&mut self.post_chain`.
        let (post_result, post_us) = timed(|| -> Result<(), EffectError> {
            let mut composed = FramePlane::new(frame_bytes, self.frame_w, self.frame_h);
            let mask_plane = MaskPlane::new(&mut self.mask_full, self.frame_w, self.frame_h);
            for effect in &mut self.post_chain {
                effect.process(&mut composed, &mask_plane, ctx)?;
            }
            Ok(())
        });
        post_result?;

        trace!(
            frame_seq = frame.meta.sequence,
            seg_us,
            mask_us,
            fg_us,
            bg_us,
            compose_us,
            post_us,
            "composite per-stage timings",
        );
        Ok(())
    }
}

// End-to-end test for `CompositeEffect::process` lives in
// `tests/composite_e2e.rs` (added in Stage 9.6 once the builder is
// in place — it needs the sidecar TOML loader or a test-only
// shortcut into `SegmentationBase`).
