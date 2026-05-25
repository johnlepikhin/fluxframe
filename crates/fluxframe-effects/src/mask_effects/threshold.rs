//! Binarise the mask at a configurable level.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::plane::{MaskEffect, MaskPlane};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

use crate::processing::threshold;

/// TOML schema: `level = 0.5` (default).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdConfig {
    /// Values `>= level` become 1.0, others 0.0. Range `[0, 1]`.
    #[serde(default = "default_level")]
    pub level: f32,
}

fn default_level() -> f32 {
    0.5
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            level: default_level(),
        }
    }
}

/// `MaskEffect` implementation of [`crate::processing::threshold`].
#[derive(Default)]
pub struct ThresholdMaskEffect {
    level: f32,
}

impl ThresholdMaskEffect {
    /// Effect name as registered in the mask registry.
    pub const NAME: &'static str = "threshold";

    /// Construct with default `level = 0.5`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            level: default_level(),
        }
    }
}

impl MaskEffect for ThresholdMaskEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: ThresholdConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        if !(0.0..=1.0).contains(&cfg.level) {
            return Err(super::invalid_config(
                Self::NAME,
                format!("level must be in [0,1], got {}", cfg.level),
            ));
        }
        self.level = cfg.level;
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
        threshold(mask.data, self.level);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_when_no_params() {
        let mut effect = ThresholdMaskEffect::new();
        let params: RawEffectParams = toml::Value::Table(toml::map::Map::new());
        effect.configure(params).expect("ok");
        assert!((effect.level - 0.5).abs() < 1e-6);
    }

    #[test]
    fn applies_level() {
        let mut effect = ThresholdMaskEffect::new();
        let params: RawEffectParams = toml::from_str("level = 0.3").unwrap();
        effect.configure(params).expect("ok");

        let mut data = vec![0.1, 0.3, 0.5, 0.9];
        let mut plane = MaskPlane::new(&mut data, 2, 2);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("ok");
        assert_eq!(data, vec![0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn rejects_out_of_range_level() {
        let mut effect = ThresholdMaskEffect::new();
        let params: RawEffectParams = toml::from_str("level = 1.5").unwrap();
        assert!(effect.configure(params).is_err());
    }
}
