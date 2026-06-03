//! Solid-colour background fill.
//!
//! New in Stage 9: replaces the background with a single RGB colour
//! supplied by the operator. Pairs naturally with the segmentation
//! mask to produce a "green-screen" or branded-backdrop effect.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::{CommitStrategy, EffectMetadata, ParamDescriptor, ParamKind};
use fluxframe_core::plane::{FramePlane, PlaneEffect};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

/// TOML schema:
///
/// ```toml
/// [background.color_fill]
/// rgb = [0, 120, 215]   # required, 3-byte array in 0..=255
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColorFillConfig {
    /// Fill colour in packed RGB order. Three `u8` channels.
    pub rgb: [u8; 3],
}

/// `PlaneEffect` that overwrites every pixel of the plane with a
/// fixed RGB colour.
pub struct ColorFillEffect {
    rgb: [u8; 3],
}

impl ColorFillEffect {
    /// Effect name as registered in the plane registry.
    pub const NAME: &'static str = "color_fill";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Replace the plane with a single RGB fill colour.",
        params: &[ParamDescriptor {
            name: "rgb",
            kind: ParamKind::Color {
                default: [128, 128, 128],
            },
            help: "Fill colour as a 3-byte RGB tuple.",
            commit: CommitStrategy::OnCommit,
        }],
    };

    /// Construct with an explicit colour (mostly for tests; the
    /// production path goes through [`PlaneEffect::configure`]).
    #[must_use]
    pub fn new(rgb: [u8; 3]) -> Self {
        Self { rgb }
    }
}

impl Default for ColorFillEffect {
    /// Default colour is a neutral mid-grey so a misconfigured effect
    /// is visually obvious without crashing.
    fn default() -> Self {
        Self {
            rgb: [128, 128, 128],
        }
    }
}

impl PlaneEffect for ColorFillEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: ColorFillConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        self.rgb = cfg.rgb;
        Ok(())
    }

    fn prepare(&mut self, _context: &ProcessingContext) -> Result<(), EffectError> {
        Ok(())
    }

    fn process(
        &mut self,
        plane: &mut FramePlane<'_>,
        _ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        for chunk in plane.data.chunks_exact_mut(3) {
            chunk[0] = self.rgb[0];
            chunk[1] = self.rgb[1];
            chunk[2] = self.rgb[2];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configure_sets_rgb() {
        let mut effect = ColorFillEffect::default();
        let params: RawEffectParams = toml::from_str("rgb = [10, 20, 30]").unwrap();
        effect.configure(params).expect("ok");
        assert_eq!(effect.rgb, [10, 20, 30]);
    }

    #[test]
    fn process_overwrites_every_pixel() {
        let mut effect = ColorFillEffect::new([255, 0, 0]);
        // 2x2 RGB frame initially zero.
        let mut data = vec![0u8; 12];
        let mut plane = FramePlane::new(&mut data, 2, 2);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("ok");
        for chunk in data.chunks_exact(3) {
            assert_eq!(chunk, [255, 0, 0]);
        }
    }

    #[test]
    fn rejects_unknown_field() {
        let mut effect = ColorFillEffect::default();
        let params: RawEffectParams = toml::from_str("rgb = [1, 2, 3]\nfoo = 1").unwrap();
        assert!(effect.configure(params).is_err());
    }
}
