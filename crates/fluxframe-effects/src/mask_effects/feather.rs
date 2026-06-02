//! Separable box-blur of the mask producing a soft edge transition.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::plane::{MaskEffect, MaskPlane};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

use crate::processing::feather;

const MAX_RADIUS: u32 = 64;

/// TOML schema: `radius = 7` (default).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatherConfig {
    /// Half-kernel radius in pixels at *mask* resolution. Values
    /// `< 1` disable the effect.
    #[serde(default = "default_radius")]
    pub radius: u32,
}

fn default_radius() -> u32 {
    7
}

impl Default for FeatherConfig {
    fn default() -> Self {
        Self {
            radius: default_radius(),
        }
    }
}

/// `MaskEffect` wrapper around [`crate::processing::feather`].
#[derive(Default)]
pub struct FeatherMaskEffect {
    radius: u32,
    scratch: Vec<f32>,
}

impl FeatherMaskEffect {
    /// Effect name as registered in the mask registry.
    pub const NAME: &'static str = "feather";

    /// Construct with the default radius.
    #[must_use]
    pub fn new() -> Self {
        Self {
            radius: default_radius(),
            scratch: Vec::new(),
        }
    }
}

impl MaskEffect for FeatherMaskEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: FeatherConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        if cfg.radius > MAX_RADIUS {
            return Err(super::invalid_config(
                Self::NAME,
                format!("radius must be <= {MAX_RADIUS}, got {}", cfg.radius),
            ));
        }
        self.radius = cfg.radius;
        Ok(())
    }

    fn prepare(&mut self, _context: &ProcessingContext) -> Result<(), EffectError> {
        Ok(())
    }

    fn process(
        &mut self,
        mask: &mut MaskPlane<'_>,
        _ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        if self.radius == 0 {
            return Ok(());
        }
        if self.scratch.len() != mask.data.len() {
            self.scratch.resize(mask.data.len(), 0.0);
        }
        feather(
            mask.data,
            &mut self.scratch,
            mask.width,
            mask.height,
            self.radius,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radius_zero_is_no_op() {
        let mut effect = FeatherMaskEffect::new();
        effect.radius = 0;
        let mut data = vec![0.0, 1.0, 0.0, 1.0];
        let snapshot = data.clone();
        let mut plane = MaskPlane::new(&mut data, 2, 2);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("ok");
        assert_eq!(data, snapshot);
    }

    #[test]
    fn softens_edges() {
        let mut effect = FeatherMaskEffect::new();
        let params: RawEffectParams = toml::from_str("radius = 1").unwrap();
        effect.configure(params).expect("ok");

        // 3x3 mask: hard step between left two columns (fg) and right column (bg).
        let mut data = vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let mut plane = MaskPlane::new(&mut data, 3, 3);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("ok");
        // After feathering, the rightmost column must have non-zero
        // values (bled in from the left), and the centre column must be
        // strictly less than 1.
        assert!(data[2] > 0.0 && data[5] > 0.0 && data[8] > 0.0);
        assert!(data[1] < 1.0 && data[4] < 1.0 && data[7] < 1.0);
    }

    #[test]
    fn configure_after_prepare_is_safe() {
        // Live-reconfig contract (Stage 13): configure() may be re-called
        // after prepare(). Verify the next process() reflects the new
        // `radius` (visible as a stronger blur of the same input) and
        // does not panic.
        let mut effect = FeatherMaskEffect::new();
        effect
            .configure(toml::from_str("radius = 1").unwrap())
            .expect("configure 1");
        let context = ProcessingContext {
            width: 8,
            height: 8,
            format: fluxframe_core::PixelFormat::Rgb,
            fps: 30,
            counters: None,
        };
        effect.prepare(&context).expect("prepare");

        // Seed mask: half-on / half-off step in the middle column.
        // 5×1 row simplifies the verification — feather(radius=1) leaks
        // exactly one pixel either side, radius=3 leaks further.
        let seed = vec![1.0, 1.0, 0.0, 0.0, 0.0];
        let mut data = seed.clone();
        {
            let mut plane = MaskPlane::new(&mut data, 5, 1);
            let mut fctx = FrameContext::default();
            effect.process(&mut plane, &mut fctx).expect("process 1");
        }
        let small_radius_far_edge = data[4];

        // Re-configure with much larger radius.
        effect
            .configure(toml::from_str("radius = 3").unwrap())
            .expect("re-configure ok");
        // Re-seed and re-process: large radius must leak further into
        // the originally-zero side of the mask.
        let mut data = seed.clone();
        let mut plane = MaskPlane::new(&mut data, 5, 1);
        let mut fctx = FrameContext::default();
        effect.process(&mut plane, &mut fctx).expect("process 2");
        let big_radius_far_edge = data[4];
        assert!(
            big_radius_far_edge > small_radius_far_edge,
            "radius=3 must bleed further than radius=1: r1={small_radius_far_edge}, r3={big_radius_far_edge}",
        );
        assert_eq!(effect.radius, 3);
    }
}
