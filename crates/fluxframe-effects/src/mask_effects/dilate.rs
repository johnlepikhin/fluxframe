//! 3×3 max-kernel dilation with configurable iteration count.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::plane::{MaskEffect, MaskPlane};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

use crate::processing::dilate;

const MAX_ITERATIONS: u32 = 32;

/// TOML schema: `iterations = 1` (default).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DilateConfig {
    /// Number of 3×3 dilation passes; one pass expands the foreground
    /// by ~1 pixel of the *current* mask resolution.
    #[serde(default = "default_iterations")]
    pub iterations: u32,
}

fn default_iterations() -> u32 {
    1
}

impl Default for DilateConfig {
    fn default() -> Self {
        Self {
            iterations: default_iterations(),
        }
    }
}

/// `MaskEffect` wrapper around [`crate::processing::dilate`].
#[derive(Default)]
pub struct DilateMaskEffect {
    iterations: u32,
    /// Same-length scratch buffer required by the primitive. Grown
    /// lazily on first `process` so we do not need the mask
    /// dimensions at `prepare` time (mask resolution is determined by
    /// the preceding stage, not by `ProcessingContext.width/height`).
    scratch: Vec<f32>,
}

impl DilateMaskEffect {
    /// Effect name as registered in the mask registry.
    pub const NAME: &'static str = "dilate";

    /// Construct with the default iteration count.
    #[must_use]
    pub fn new() -> Self {
        Self {
            iterations: default_iterations(),
            scratch: Vec::new(),
        }
    }
}

impl MaskEffect for DilateMaskEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: DilateConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        if cfg.iterations > MAX_ITERATIONS {
            return Err(super::invalid_config(
                Self::NAME,
                format!(
                    "iterations must be <= {MAX_ITERATIONS}, got {}",
                    cfg.iterations
                ),
            ));
        }
        self.iterations = cfg.iterations;
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
        if self.iterations == 0 {
            return Ok(());
        }
        if self.scratch.len() != mask.data.len() {
            self.scratch.resize(mask.data.len(), 0.0);
        }
        dilate(
            mask.data,
            &mut self.scratch,
            mask.width,
            mask.height,
            self.iterations,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_when_iterations_zero() {
        let mut effect = DilateMaskEffect::new();
        effect.iterations = 0;
        let mut data = vec![0.0, 1.0, 0.0, 0.0];
        let snapshot = data.clone();
        let mut plane = MaskPlane::new(&mut data, 2, 2);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("ok");
        assert_eq!(data, snapshot);
    }

    #[test]
    fn expands_foreground() {
        let mut effect = DilateMaskEffect::new();
        let params: RawEffectParams = toml::from_str("iterations = 1").unwrap();
        effect.configure(params).expect("ok");

        // 3x3 mask with single foreground pixel in the centre.
        let mut data = vec![0.0; 9];
        data[4] = 1.0;
        let mut plane = MaskPlane::new(&mut data, 3, 3);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("ok");
        // All 9 pixels must now be foreground (3x3 kernel max).
        assert!(data.iter().all(|&v| v > 0.99), "got {data:?}");
    }

    #[test]
    fn rejects_excessive_iterations() {
        let mut effect = DilateMaskEffect::new();
        let raw = format!("iterations = {}", MAX_ITERATIONS + 1);
        let params: RawEffectParams = toml::from_str(&raw).unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn configure_after_prepare_is_safe() {
        // Live-reconfig contract (Stage 13): configure() may be re-called
        // after prepare(). Verify the next process() reflects the new
        // `iterations` count without panicking.
        let mut effect = DilateMaskEffect::new();
        effect
            .configure(toml::Value::Table(toml::map::Map::new()))
            .expect("configure 1 (defaults)");

        let context = ProcessingContext {
            width: 8,
            height: 8,
            format: fluxframe_core::PixelFormat::Rgb,
            fps: 30,
            counters: None,
        };
        effect.prepare(&context).expect("prepare");

        // 5×5 mask, single hot pixel at the centre.
        let mut data = vec![0.0_f32; 25];
        data[12] = 1.0;
        {
            let mut plane = MaskPlane::new(&mut data, 5, 5);
            let mut fctx = FrameContext::default();
            effect.process(&mut plane, &mut fctx).expect("process 1");
        }
        // 1 iteration of 3×3 max → expands to a 3×3 block (corners at
        // distance 1 from the centre). Corners of the 5×5 mask stay 0.
        assert!(data[0] < 0.5, "iter=1 corner must stay 0: {data:?}");

        // Re-configure with higher iterations (post-prepare).
        effect
            .configure(toml::from_str("iterations = 3").unwrap())
            .expect("re-configure ok");

        // Re-seed: single centre pixel again so we can observe the
        // bigger spread.
        let mut data = vec![0.0_f32; 25];
        data[12] = 1.0;
        let mut plane = MaskPlane::new(&mut data, 5, 5);
        let mut fctx = FrameContext::default();
        effect.process(&mut plane, &mut fctx).expect("process 2");
        // 3 iterations of 3×3 max from the centre saturate the full 5×5
        // (the diagonal corner is at Chebyshev distance 2; 3 iterations
        // reach distance 3).
        assert!(
            data.iter().all(|&v| v > 0.99),
            "iter=3 must fill the 5×5 mask: {data:?}"
        );
        assert_eq!(effect.iterations, 3);
    }
}
