//! Radial darkening (vignette). Multiplies each pixel's RGB channels by
//! a per-pixel attenuation factor that depends on the normalised
//! distance from the frame centre. The factor is `1.0` inside
//! `inner_radius` and falls off smoothly to `1.0 - strength` at the
//! corners.
//!
//! Use cases:
//! * `[background]` — gentle "cinematographic" attention focus toward
//!   the speaker; hides messy edges of the room.
//! * `[foreground]` — uncommon; would darken the speaker's outline.
//!
//! Cost: `O(width × height)` per frame — one multiplicative pass against
//! a precomputed lookup table allocated once in `prepare()`.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::{
    CommitStrategy, DEBOUNCE_FAST_MS, EffectMetadata, ParamDescriptor, ParamKind, Scale,
};
use fluxframe_core::plane::{FramePlane, PlaneEffect};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

use super::helpers::reject_out_of_range;

/// Default maximum darkening at the frame corners for the `strength` field.
pub const DEFAULT_STRENGTH: f32 = 0.4;
/// Default normalised radius at which the vignette begins to take effect.
pub const DEFAULT_INNER_RADIUS: f32 = 0.5;

/// TOML schema:
///
/// ```toml
/// [background.vignette]
/// strength = 0.4
/// inner_radius = 0.5
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VignetteConfig {
    /// Maximum darkening at the frame corners. `0.0` is a no-op,
    /// `1.0` drives the corner pixels fully black. Validated to be a
    /// finite number in `[0.0, 1.0]`.
    #[serde(default = "default_strength")]
    pub strength: f32,
    /// Normalised radius (`0.0 = centre`, `1.0 = corner`) at which the
    /// vignette begins to take effect. Pixels closer to the centre
    /// than `inner_radius` are unchanged. Validated to be a finite
    /// number in `[0.0, 1.0)` — exactly `1.0` would push all falloff
    /// past the frame corners and make the effect a no-op, so reject
    /// it loudly.
    #[serde(default = "default_inner_radius")]
    pub inner_radius: f32,
}

fn default_strength() -> f32 {
    DEFAULT_STRENGTH
}

fn default_inner_radius() -> f32 {
    DEFAULT_INNER_RADIUS
}

impl Default for VignetteConfig {
    fn default() -> Self {
        Self {
            strength: default_strength(),
            inner_radius: default_inner_radius(),
        }
    }
}

/// `PlaneEffect` applying a smooth radial vignette.
pub struct VignetteEffect {
    config: VignetteConfig,
    /// Per-pixel multiplicative attenuation factors in `0..=255`, laid
    /// out as `width * height` in row-major order. Built once in
    /// `prepare` so the hot loop avoids per-pixel `sqrt`.
    lut: Vec<u8>,
    frame_w: u32,
    frame_h: u32,
}

impl VignetteEffect {
    /// Effect name as registered in the plane registry.
    pub const NAME: &'static str = "vignette";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Smooth radial darkening toward the frame corners.",
        params: &[
            ParamDescriptor {
                name: "strength",
                kind: ParamKind::Float {
                    default: DEFAULT_STRENGTH,
                    min: 0.0,
                    max: 1.0,
                    step: 0.05,
                    scale: Scale::Linear,
                },
                help: "Max darkening at the corners; 0 disables.",
                commit: CommitStrategy::Live {
                    debounce_ms: DEBOUNCE_FAST_MS,
                },
            },
            ParamDescriptor {
                name: "inner_radius",
                kind: ParamKind::Float {
                    default: DEFAULT_INNER_RADIUS,
                    min: 0.0,
                    max: 0.99,
                    step: 0.05,
                    scale: Scale::Linear,
                },
                help: "Normalised radius where falloff starts (0=centre).",
                commit: CommitStrategy::Live {
                    debounce_ms: DEBOUNCE_FAST_MS,
                },
            },
        ],
    };

    /// Construct with the default strength and inner radius — must be
    /// `configure`d and `prepare`d before use.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: VignetteConfig::default(),
            lut: Vec::new(),
            frame_w: 0,
            frame_h: 0,
        }
    }
}

impl Default for VignetteEffect {
    fn default() -> Self {
        Self::new()
    }
}

/// Standard smoothstep: returns a smooth interpolation `0..=1` for
/// `x` advancing across `[edge0, edge1]`. Outside the interval the
/// result is clamped to `0` or `1`.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl PlaneEffect for VignetteEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: VignetteConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        reject_out_of_range(Self::NAME, "strength", cfg.strength, 0.0, 1.0)?;
        // `inner_radius` has an exclusive upper bound — the shared
        // helper assumes inclusive ranges, so validate it inline.
        if !cfg.inner_radius.is_finite() {
            return Err(super::invalid_config(
                Self::NAME,
                format!("inner_radius must be finite, got {}", cfg.inner_radius),
            ));
        }
        if cfg.inner_radius < 0.0 || cfg.inner_radius >= 1.0 {
            return Err(super::invalid_config(
                Self::NAME,
                format!(
                    "inner_radius must be in [0.0, 1.0), got {}",
                    cfg.inner_radius
                ),
            ));
        }
        self.config = cfg;
        Ok(())
    }

    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError> {
        self.frame_w = context.width;
        self.frame_h = context.height;
        let width = context.width as usize;
        let height = context.height as usize;
        self.lut = vec![255u8; width * height];
        if width == 0 || height == 0 {
            return Ok(());
        }
        // Use the frame centre as the reference point and the corner
        // distance as the normalising scale. `+ 0.5` puts coordinates
        // at the pixel centre.
        let cx = (context.width as f32) * 0.5;
        let cy = (context.height as f32) * 0.5;
        // Distance to the corner — used to normalise `d` into `[0, 1]`.
        // For non-square frames this puts the falloff at the corner,
        // not at the long-axis edge.
        let r_max = (cx * cx + cy * cy).sqrt();
        // `r_max > 0` because width and height are both ≥ 1 here.
        let strength = self.config.strength;
        let inner = self.config.inner_radius;
        for y in 0..height {
            let py = (y as f32) + 0.5;
            for x in 0..width {
                let px = (x as f32) + 0.5;
                let dx = px - cx;
                let dy = py - cy;
                let d = (dx * dx + dy * dy).sqrt() / r_max;
                let falloff = smoothstep(inner, 1.0, d);
                let m = (1.0 - strength * falloff).clamp(0.0, 1.0);
                self.lut[y * width + x] = (m * 255.0 + 0.5) as u8;
            }
        }
        Ok(())
    }

    fn process(
        &mut self,
        plane: &mut FramePlane<'_>,
        _ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        if plane.width != self.frame_w || plane.height != self.frame_h {
            return Err(EffectError::ProcessFailed {
                name: Self::NAME.to_string(),
                reason: format!(
                    "plane dimensions {}x{} differ from prepared {}x{}",
                    plane.width, plane.height, self.frame_w, self.frame_h
                ),
            });
        }
        if self.config.strength == 0.0 {
            return Ok(());
        }
        let width = plane.width as usize;
        let height = plane.height as usize;
        if width == 0 || height == 0 {
            return Ok(());
        }
        debug_assert_eq!(
            plane.data.len(),
            width * height * 3,
            "FramePlane invariant: data.len() must equal width * height * 3",
        );
        debug_assert_eq!(
            self.lut.len(),
            width * height,
            "LUT size diverged from prepared dimensions",
        );
        // Walk the plane and the LUT in lockstep. The 3-byte chunks
        // make the per-pixel multiply auto-vectorisable and remove the
        // manual row/column indexing.
        for (pixel, &m) in plane.data.chunks_exact_mut(3).zip(self.lut.iter()) {
            let m = u16::from(m);
            // Integer multiply-divide: `(c * m + 127) / 255` rounds to
            // the nearest integer; max value is 255 * 255 = 65 025
            // which fits in u16.
            for c in pixel.iter_mut() {
                let v = u16::from(*c);
                *c = ((v * m + 127) / 255) as u8;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(effect: &mut VignetteEffect, data: &mut [u8], w: u32, h: u32) {
        let context = ProcessingContext {
            width: w,
            height: h,
            format: fluxframe_core::PixelFormat::Rgb,
            fps: 30,
            counters: None,
        };
        effect.prepare(&context).expect("prepare");
        let mut plane = FramePlane::new(data, w, h);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("process");
    }

    #[test]
    fn defaults_when_no_params() {
        let mut effect = VignetteEffect::new();
        let params: RawEffectParams = toml::Value::Table(toml::map::Map::new());
        effect.configure(params).expect("ok");
        assert!((effect.config.strength - default_strength()).abs() < 1e-6);
        assert!((effect.config.inner_radius - default_inner_radius()).abs() < 1e-6);
    }

    #[test]
    fn rejects_negative_strength() {
        let mut effect = VignetteEffect::new();
        let params: RawEffectParams = toml::from_str("strength = -0.1").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn rejects_strength_above_one() {
        let mut effect = VignetteEffect::new();
        let params: RawEffectParams = toml::from_str("strength = 1.5").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn rejects_inner_radius_at_one() {
        // Exactly 1.0 would push the falloff out of the visible
        // window (the effect becomes a no-op). Reject loudly so the
        // operator notices.
        let mut effect = VignetteEffect::new();
        let params: RawEffectParams = toml::from_str("inner_radius = 1.0").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn rejects_inner_radius_negative() {
        let mut effect = VignetteEffect::new();
        let params: RawEffectParams = toml::from_str("inner_radius = -0.1").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn strength_zero_is_identity() {
        let mut effect = VignetteEffect::new();
        let params: RawEffectParams = toml::from_str("strength = 0.0").unwrap();
        effect.configure(params).expect("configure");
        let mut data: Vec<u8> = (0..(4 * 4 * 3)).map(|i| i as u8).collect();
        let original = data.clone();
        run(&mut effect, &mut data, 4, 4);
        assert_eq!(data, original, "strength=0 must be a no-op");
    }

    #[test]
    fn corner_pixels_darker_than_center() {
        // Constant grey frame. After vignette the corner pixels must be
        // strictly darker than the centre pixels.
        let mut effect = VignetteEffect::new();
        let params: RawEffectParams = toml::from_str("strength = 0.7\ninner_radius = 0.0").unwrap();
        effect.configure(params).expect("configure");
        let w: u32 = 8;
        let h: u32 = 8;
        let mut data = vec![200u8; (w * h * 3) as usize];
        run(&mut effect, &mut data, w, h);
        // Centre pixel — closest to (3.5, 3.5), pick (3, 3).
        let centre_offset = ((3 * w + 3) * 3) as usize;
        let centre_r = data[centre_offset];
        // Corner pixel — (0, 0).
        let corner_offset = 0usize;
        let corner_r = data[corner_offset];
        assert!(
            corner_r < centre_r,
            "corner ({corner_r}) should be darker than centre ({centre_r})"
        );
        // And the centre should be close to the original value (only
        // mild attenuation, if any).
        assert!(
            centre_r >= 180,
            "centre too dark: {centre_r} (expected ≥ 180)"
        );
    }

    #[test]
    fn rejects_process_on_dimension_mismatch() {
        // Prepare for 8x8 then submit a 4x4 plane — must fail loudly
        // because the supervisor contract is to re-prepare on resize.
        let mut effect = VignetteEffect::new();
        effect
            .configure(toml::Value::Table(toml::map::Map::new()))
            .expect("configure");
        let context = ProcessingContext {
            width: 8,
            height: 8,
            format: fluxframe_core::PixelFormat::Rgb,
            fps: 30,
            counters: None,
        };
        effect.prepare(&context).expect("prepare");
        let mut data = vec![128u8; 4 * 4 * 3];
        let mut plane = FramePlane::new(&mut data, 4, 4);
        let mut ctx = FrameContext::default();
        let err = effect.process(&mut plane, &mut ctx).expect_err("must fail");
        match err {
            EffectError::ProcessFailed { reason, .. } => {
                assert!(
                    reason.contains("dimensions"),
                    "expected dimension-mismatch reason, got: {reason}"
                );
            }
            other => panic!("expected ProcessFailed, got {other:?}"),
        }
    }

    #[test]
    fn corner_attenuation_matches_strength() {
        // For strength = 0.5 and inner_radius = 0.0 the smoothstep at
        // the corner gives `falloff = 1.0`, so the corner multiplier
        // becomes `1 - 0.5 = 0.5`. A grey 200 pixel should map to
        // approximately 100. Use a 32×32 frame so the pixel-centre
        // half-offset does not visibly shift the corner away from the
        // analytical `d = 1.0`.
        let mut effect = VignetteEffect::new();
        let params: RawEffectParams = toml::from_str("strength = 0.5\ninner_radius = 0.0").unwrap();
        effect.configure(params).expect("configure");
        let w: u32 = 32;
        let h: u32 = 32;
        let mut data = vec![200u8; (w * h * 3) as usize];
        run(&mut effect, &mut data, w, h);
        let corner = data[0];
        // Allow ±2 for rounding.
        assert!(
            corner.abs_diff(100) <= 2,
            "corner attenuation should yield ~100, got {corner}"
        );
    }
}
