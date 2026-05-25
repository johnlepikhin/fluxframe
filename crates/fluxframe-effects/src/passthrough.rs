//! Passthrough effect — required by §25 of the spec.
//!
//! The implementation is intentionally trivial.  It serves as:
//!
//! * baseline for `benchmark`,
//! * fallback target for the rest of the pipeline,
//! * smoke test that the chain plumbing is wired correctly.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::frame::VideoFrame;
use fluxframe_core::traits::{RawEffectParams, VideoEffect};

/// No-op effect: leaves every frame untouched.
///
/// Used as the §25 baseline for benchmarking and as a default fallback
/// when an effect chain is otherwise empty.
#[derive(Debug, Default)]
pub struct PassthroughEffect;

impl PassthroughEffect {
    /// Canonical (snake_case) name used by the registry and CLI.
    pub const NAME: &'static str = "passthrough";

    /// Construct a fresh passthrough effect.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl VideoEffect for PassthroughEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, _params: RawEffectParams) -> Result<(), EffectError> {
        Ok(())
    }

    fn prepare(&mut self, _context: &ProcessingContext) -> Result<(), EffectError> {
        Ok(())
    }

    fn process(
        &mut self,
        _frame: &mut VideoFrame,
        _context: &mut FrameContext,
    ) -> Result<(), EffectError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxframe_core::frame::{FrameBuffer, FrameMeta, PixelFormat};

    #[test]
    fn passthrough_does_not_modify_frame() {
        let pixels: Vec<u8> = (0..(4 * 4 * 3)).map(|i| i as u8).collect();
        let original = pixels.clone();
        let mut frame = VideoFrame::new_packed(
            FrameBuffer::Owned(pixels),
            4,
            4,
            PixelFormat::Rgb,
            FrameMeta::default(),
        )
        .expect("packed RGB frame builds");

        let mut effect = PassthroughEffect::new();
        let ctx = ProcessingContext {
            width: 4,
            height: 4,
            format: PixelFormat::Rgb,
            fps: 30,
            counters: None,
        };
        effect.prepare(&ctx).unwrap();

        let mut frame_context = FrameContext::default();
        effect.process(&mut frame, &mut frame_context).unwrap();

        assert_eq!(frame.data.as_slice(), original.as_slice());
        assert_eq!(frame.width, 4);
        assert_eq!(frame.height, 4);
        assert_eq!(frame.format, PixelFormat::Rgb);
    }
}
