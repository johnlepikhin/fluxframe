//! Exponential moving average smoothing across consecutive frames.
//!
//! Stateful: keeps the previous frame's raw mask between calls. When
//! the mask resolution changes (or on first run), the state buffer is
//! re-initialised from the current mask so there is no transient
//! "ghost" from a stale plane size.
//!
//! The blending formula matches the legacy `background_blur` effect's
//! behaviour:
//!     `smoothed[i] = factor * prev_raw[i] + (1 - factor) * curr_raw[i]`
//! After each call `prev_raw` is updated to the current frame's raw
//! mask so the next call sees the correct two-tap interpolation
//! input.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::plane::{MaskEffect, MaskPlane};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

use crate::processing::smooth_temporal;

/// TOML schema: `factor = 0.65` (default). Closer to 1 → smoother
/// (more inertia, more visible lag); 0 → no smoothing.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothTemporalConfig {
    /// EMA factor in `[0, 1]`. `1.0` keeps the previous mask
    /// indefinitely; `0.0` disables the effect.
    #[serde(default = "default_factor")]
    pub factor: f32,
}

fn default_factor() -> f32 {
    0.65
}

impl Default for SmoothTemporalConfig {
    fn default() -> Self {
        Self {
            factor: default_factor(),
        }
    }
}

/// Two-buffer state machine implementing the temporal EMA over a
/// `MaskPlane`.
pub struct SmoothTemporalMaskEffect {
    factor: f32,
    /// Previous frame's raw mask. Length tracks the mask plane seen
    /// last; re-allocated if the plane resolution changes.
    prev: Vec<f32>,
    /// Scratch for the swap dance in [`Self::process`]. Same length
    /// as `prev` once seeded.
    scratch: Vec<f32>,
}

impl SmoothTemporalMaskEffect {
    /// Effect name as registered in the mask registry.
    pub const NAME: &'static str = "smooth_temporal";

    /// Construct with the default factor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            factor: default_factor(),
            prev: Vec::new(),
            scratch: Vec::new(),
        }
    }
}

impl Default for SmoothTemporalMaskEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl MaskEffect for SmoothTemporalMaskEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: SmoothTemporalConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        if !(0.0..=1.0).contains(&cfg.factor) {
            return Err(super::invalid_config(
                Self::NAME,
                format!("factor must be in [0,1], got {}", cfg.factor),
            ));
        }
        self.factor = cfg.factor;
        Ok(())
    }

    fn prepare(&mut self, _context: &ProcessingContext) -> Result<(), EffectError> {
        self.prev.clear();
        self.scratch.clear();
        Ok(())
    }

    fn process(
        &mut self,
        mask: &mut MaskPlane<'_>,
        _ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        if self.factor == 0.0 {
            return Ok(());
        }
        let n = mask.data.len();
        if self.prev.len() != n {
            // First call or resolution change: seed state with the
            // current mask and emit it unchanged. Smoothing kicks in
            // from the next frame.
            self.prev.clear();
            self.prev.extend_from_slice(mask.data);
            return Ok(());
        }
        if self.scratch.len() != n {
            self.scratch.resize(n, 0.0);
        }
        // 1. Save raw current into scratch (will become next call's
        //    `prev`).
        self.scratch.copy_from_slice(mask.data);
        // 2. EMA: prev = factor * prev + (1 - factor) * mask.
        smooth_temporal(&mut self.prev, mask.data, self.factor);
        // 3. Emit smoothed: mask <- prev.
        mask.data.copy_from_slice(&self.prev);
        // 4. prev <- raw current (saved in scratch).
        std::mem::swap(&mut self.prev, &mut self.scratch);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_call_seeds_state_and_emits_input() {
        let mut effect = SmoothTemporalMaskEffect::new();
        let mut data = vec![0.5, 0.5, 0.5, 0.5];
        let mut plane = MaskPlane::new(&mut data, 2, 2);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("ok");
        assert_eq!(data, vec![0.5, 0.5, 0.5, 0.5]);
        assert_eq!(effect.prev, vec![0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn factor_zero_is_no_op() {
        let mut effect = SmoothTemporalMaskEffect::new();
        effect.factor = 0.0;
        let mut data = vec![0.1, 0.9];
        let mut plane = MaskPlane::new(&mut data, 2, 1);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("ok");
        assert_eq!(data, vec![0.1, 0.9]);
        assert!(effect.prev.is_empty());
    }

    #[test]
    fn second_call_blends() {
        let mut effect = SmoothTemporalMaskEffect::new();
        effect.factor = 0.5;
        let mut ctx = FrameContext::default();

        // Frame 1: prev = [0.0,0.0], mask unchanged, prev seeded.
        let mut frame1 = vec![0.0, 0.0];
        {
            let mut plane = MaskPlane::new(&mut frame1, 2, 1);
            effect.process(&mut plane, &mut ctx).expect("ok");
        }
        assert_eq!(frame1, vec![0.0, 0.0]);

        // Frame 2: prev = [0.0,0.0], curr = [1.0,1.0], factor=0.5
        //   smoothed = 0.5*[0,0] + 0.5*[1,1] = [0.5,0.5].
        let mut frame2 = vec![1.0, 1.0];
        {
            let mut plane = MaskPlane::new(&mut frame2, 2, 1);
            effect.process(&mut plane, &mut ctx).expect("ok");
        }
        assert!((frame2[0] - 0.5).abs() < 1e-6, "got {}", frame2[0]);
        assert!((frame2[1] - 0.5).abs() < 1e-6, "got {}", frame2[1]);
        // prev now holds raw frame2 = [1.0, 1.0] (for next iteration).
        assert_eq!(effect.prev, vec![1.0, 1.0]);
    }
}
