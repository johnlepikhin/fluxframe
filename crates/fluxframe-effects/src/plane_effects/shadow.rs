//! Soft drop shadow of the silhouette on the room behind it.
//!
//! The mask is blurred and shifted by an offset; the plane is darkened
//! where the shifted blur covers it. Used on `[background]`, the shadow
//! peeks out beside the person, as if a lamp in front of them threw it
//! on the wall. The person's own pixels come from the foreground plane,
//! so darkening under them is hidden by the composite.
//!
//! Cost per frame: one pass over the mask (block average at a quarter
//! size, blur there) and one multiplicative pass over the plane.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::{DEBOUNCE_FAST_MS, DEBOUNCE_STANDARD_MS, EffectMetadata};
use fluxframe_core::plane::{FramePlane, MaskPlane, PlaneEffect};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

use super::helpers::{
    check_mask_size, float_param, integer_param, reject_int_out_of_range, reject_out_of_range,
};
use crate::processing::soft_mask::{SoftMask, shade};

/// Largest accepted shadow offset per axis, in pixels.
const MAX_OFFSET: i64 = 128;
/// Largest accepted blur radius, in pixels.
const MAX_BLUR: i64 = 64;

/// Default horizontal offset, in pixels (positive = right).
pub const DEFAULT_OFFSET_X: i32 = 14;
/// Default vertical offset, in pixels (positive = down).
pub const DEFAULT_OFFSET_Y: i32 = 8;
/// Default blur radius, in pixels.
pub const DEFAULT_BLUR: u32 = 12;
/// Default darkening at the shadow's core.
pub const DEFAULT_STRENGTH: f32 = 0.35;

/// TOML schema:
///
/// ```toml
/// [background.shadow]
/// offset_x = 14
/// offset_y = 8
/// blur = 12
/// strength = 0.35
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowConfig {
    /// Horizontal shift of the shadow in pixels; positive moves it right.
    #[serde(default = "default_offset_x")]
    pub offset_x: i32,
    /// Vertical shift of the shadow in pixels; positive moves it down.
    #[serde(default = "default_offset_y")]
    pub offset_y: i32,
    /// Softness of the shadow's edge, in pixels.
    #[serde(default = "default_blur")]
    pub blur: u32,
    /// Darkening where the shadow is full: `0.0` none, `1.0` black.
    #[serde(default = "default_strength")]
    pub strength: f32,
}

fn default_offset_x() -> i32 {
    DEFAULT_OFFSET_X
}
fn default_offset_y() -> i32 {
    DEFAULT_OFFSET_Y
}
fn default_blur() -> u32 {
    DEFAULT_BLUR
}
fn default_strength() -> f32 {
    DEFAULT_STRENGTH
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            offset_x: default_offset_x(),
            offset_y: default_offset_y(),
            blur: default_blur(),
            strength: default_strength(),
        }
    }
}

/// `PlaneEffect` casting a soft shadow of the silhouette.
pub struct ShadowEffect {
    config: ShadowConfig,
    soft: SoftMask,
}

impl ShadowEffect {
    /// Effect name as registered in the plane registry.
    pub const NAME: &'static str = "shadow";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Soft shadow of the person cast on the plane (uses the mask).",
        params: &[
            integer_param(
                "offset_x",
                DEFAULT_OFFSET_X as i64,
                -MAX_OFFSET,
                MAX_OFFSET,
                DEBOUNCE_FAST_MS,
                "Horizontal shift of the shadow in pixels; positive moves it right.",
            ),
            integer_param(
                "offset_y",
                DEFAULT_OFFSET_Y as i64,
                -MAX_OFFSET,
                MAX_OFFSET,
                DEBOUNCE_FAST_MS,
                "Vertical shift of the shadow in pixels; positive moves it down.",
            ),
            integer_param(
                "blur",
                DEFAULT_BLUR as i64,
                0,
                MAX_BLUR,
                DEBOUNCE_STANDARD_MS,
                "Softness of the shadow's edge, in pixels.",
            ),
            float_param(
                "strength",
                DEFAULT_STRENGTH,
                0.0,
                1.0,
                DEBOUNCE_FAST_MS,
                "Darkening where the shadow is full: 0 none, 1 black.",
            ),
        ],
    };

    /// Construct with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ShadowConfig::default(),
            soft: SoftMask::new(),
        }
    }
}

impl Default for ShadowEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl PlaneEffect for ShadowEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: ShadowConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        for (field, value) in [("offset_x", cfg.offset_x), ("offset_y", cfg.offset_y)] {
            reject_int_out_of_range(Self::NAME, field, value.into(), -MAX_OFFSET, MAX_OFFSET)?;
        }
        reject_int_out_of_range(Self::NAME, "blur", cfg.blur.into(), 0, MAX_BLUR)?;
        reject_out_of_range(Self::NAME, "strength", cfg.strength, 0.0, 1.0)?;
        self.config = cfg;
        Ok(())
    }

    fn prepare(&mut self, _context: &ProcessingContext) -> Result<(), EffectError> {
        Ok(())
    }

    fn process(
        &mut self,
        plane: &mut FramePlane<'_>,
        mask: &MaskPlane<'_>,
        _ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        check_mask_size(Self::NAME, plane, mask)?;
        let width = plane.width as usize;
        if width == 0 || plane.height == 0 || self.config.strength == 0.0 {
            return Ok(());
        }
        self.soft
            .update(mask.data, plane.width, plane.height, self.config.blur);
        let strength = self.config.strength * 256.0;
        let shift = (self.config.offset_x, self.config.offset_y);
        shade(plane.data, width, &self.soft, shift, |s| {
            256 - ((s * strength + 0.5) as u32).min(256)
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::filled_mask;

    const W: u32 = 96;
    const H: u32 = 64;

    fn configured(cfg: &str) -> ShadowEffect {
        let mut e = ShadowEffect::new();
        e.configure(toml::from_str(cfg).unwrap())
            .expect("configure");
        e
    }

    /// Mask with a person rectangle `[x0, x1) × [y0, y1)`.
    fn person(x0: u32, x1: u32, y0: u32, y1: u32) -> Vec<f32> {
        (0..W * H)
            .map(|i| {
                let (x, y) = (i % W, i / W);
                if (x0..x1).contains(&x) && (y0..y1).contains(&y) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn run(e: &mut ShadowEffect, mask: &mut [f32]) -> Vec<u8> {
        let mut data = vec![200_u8; (W * H * 3) as usize];
        let mut plane = FramePlane::new(&mut data, W, H);
        let mask = MaskPlane::new(mask, W, H);
        e.process(&mut plane, &mask, &mut FrameContext::default())
            .expect("process");
        data
    }

    fn at(data: &[u8], x: u32, y: u32) -> u8 {
        data[((y * W + x) * 3) as usize]
    }

    #[test]
    fn no_person_leaves_the_plane_untouched() {
        let mut e = configured("");
        let mut buf = Vec::new();
        let mask = filled_mask(&mut buf, W, H, 0.0);
        let mut data = vec![123_u8; (W * H * 3) as usize];
        let mut plane = FramePlane::new(&mut data, W, H);
        e.process(&mut plane, &mask, &mut FrameContext::default())
            .unwrap();
        assert!(data.iter().all(|&v| v == 123));
    }

    #[test]
    fn the_shadow_falls_beside_the_person_along_the_offset() {
        let mut e = configured("offset_x = 16\noffset_y = 0\nblur = 4\nstrength = 0.5");
        let mut mask = person(32, 48, 16, 48);
        let out = run(&mut e, &mut mask);
        // Right of the person, inside the offset: dark.
        assert!(at(&out, 56, 32) < 120, "{}", at(&out, 56, 32));
        // Left of the person, opposite to the offset: untouched.
        assert_eq!(at(&out, 20, 32), 200);
        // Far right, beyond the shadow: untouched.
        assert_eq!(at(&out, 90, 32), 200);
    }

    #[test]
    fn zero_strength_is_a_no_op() {
        let mut e = configured("strength = 0.0");
        let mut mask = person(32, 48, 16, 48);
        assert!(run(&mut e, &mut mask).iter().all(|&v| v == 200));
    }

    #[test]
    fn full_strength_reaches_black_inside_a_large_shadow() {
        let mut e = configured("offset_x = 0\noffset_y = 0\nblur = 0\nstrength = 1.0");
        let mut mask = vec![1.0_f32; (W * H) as usize];
        assert!(run(&mut e, &mut mask).iter().all(|&v| v == 0));
    }

    #[test]
    fn mismatched_mask_is_an_error() {
        let mut e = configured("");
        let mut data = vec![0_u8; (W * H * 3) as usize];
        let mut plane = FramePlane::new(&mut data, W, H);
        let mut buf = Vec::new();
        let mask = filled_mask(&mut buf, W / 2, H, 0.0);
        let err = e
            .process(&mut plane, &mask, &mut FrameContext::default())
            .unwrap_err();
        assert!(matches!(err, EffectError::ProcessFailed { .. }));
    }

    #[test]
    fn rejects_out_of_range_values_and_keeps_the_config() {
        let mut e = configured("blur = 20");
        for bad in [
            "offset_x = 129",
            "offset_y = -129",
            "blur = 65",
            "strength = 1.5",
            "strength = -0.1",
            "unknown = 1",
        ] {
            assert!(e.configure(toml::from_str(bad).unwrap()).is_err(), "{bad}");
            assert_eq!(e.config.blur, 20, "{bad} changed the config");
        }
        for good in ["offset_x = -128", "offset_y = 128", "blur = 0", "blur = 64"] {
            e.configure(toml::from_str(good).unwrap())
                .unwrap_or_else(|err| panic!("{good}: {err}"));
        }
    }
}
