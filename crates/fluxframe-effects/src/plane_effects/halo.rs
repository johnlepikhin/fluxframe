//! Halo: the plane is lit around the person and dimmed away from them.
//!
//! A vignette centred on the silhouette instead of the frame. The mask
//! is blurred by `radius` pixels; next to the person the plane is
//! brightened by up to `glow`, far from them it is darkened by `dim`.
//! The halo moves only when the person moves and is soft by
//! construction, so it does not draw attention.
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
use crate::processing::soft_mask::{SoftMask, closeness, shade};

const MIN_RADIUS: i64 = 8;
const MAX_RADIUS: i64 = 256;

/// Default reach of the halo, in pixels.
pub const DEFAULT_RADIUS: u32 = 60;
/// Default brightening next to the person.
pub const DEFAULT_GLOW: f32 = 0.15;
/// Default darkening far from the person.
pub const DEFAULT_DIM: f32 = 0.3;

/// TOML schema:
///
/// ```toml
/// [background.halo]
/// radius = 60
/// glow = 0.15
/// dim = 0.3
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HaloConfig {
    /// How far the halo reaches from the person, in pixels.
    #[serde(default = "default_radius")]
    pub radius: u32,
    /// Brightening right next to the person: `0.0` none, `1.0` double.
    #[serde(default = "default_glow")]
    pub glow: f32,
    /// Darkening far from the person: `0.0` none, `1.0` black.
    #[serde(default = "default_dim")]
    pub dim: f32,
}

fn default_radius() -> u32 {
    DEFAULT_RADIUS
}
fn default_glow() -> f32 {
    DEFAULT_GLOW
}
fn default_dim() -> f32 {
    DEFAULT_DIM
}

impl Default for HaloConfig {
    fn default() -> Self {
        Self {
            radius: default_radius(),
            glow: default_glow(),
            dim: default_dim(),
        }
    }
}

/// `PlaneEffect` lighting the plane around the person.
pub struct HaloEffect {
    config: HaloConfig,
    soft: SoftMask,
}

impl HaloEffect {
    /// Effect name as registered in the plane registry.
    pub const NAME: &'static str = "halo";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Light the plane around the person and dim it far away (uses the mask).",
        params: &[
            integer_param(
                "radius",
                DEFAULT_RADIUS as i64,
                MIN_RADIUS,
                MAX_RADIUS,
                DEBOUNCE_STANDARD_MS,
                "How far the halo reaches from the person, in pixels.",
            ),
            float_param(
                "glow",
                DEFAULT_GLOW,
                0.0,
                1.0,
                DEBOUNCE_FAST_MS,
                "Brightening next to the person: 0 none, 1 double.",
            ),
            float_param(
                "dim",
                DEFAULT_DIM,
                0.0,
                1.0,
                DEBOUNCE_FAST_MS,
                "Darkening far from the person: 0 none, 1 black.",
            ),
        ],
    };

    /// Construct with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: HaloConfig::default(),
            soft: SoftMask::new(),
        }
    }
}

impl Default for HaloEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl PlaneEffect for HaloEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: HaloConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        reject_int_out_of_range(
            Self::NAME,
            "radius",
            cfg.radius.into(),
            MIN_RADIUS,
            MAX_RADIUS,
        )?;
        reject_out_of_range(Self::NAME, "glow", cfg.glow, 0.0, 1.0)?;
        reject_out_of_range(Self::NAME, "dim", cfg.dim, 0.0, 1.0)?;
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
        let HaloConfig { radius, glow, dim } = self.config;
        if width == 0 || plane.height == 0 || (glow == 0.0 && dim == 0.0) {
            return Ok(());
        }
        self.soft
            .update(mask.data, plane.width, plane.height, radius);
        shade(plane.data, width, &self.soft, (0, 0), |s| {
            let near = closeness(s);
            let gain = 1.0 + glow * near - dim * (1.0 - near);
            (gain * 256.0 + 0.5) as u32
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::filled_mask;

    const W: u32 = 160;
    const H: u32 = 64;

    fn configured(cfg: &str) -> HaloEffect {
        let mut e = HaloEffect::new();
        e.configure(toml::from_str(cfg).unwrap())
            .expect("configure");
        e
    }

    /// Person in columns `[x0, x1)`, full height.
    fn person(x0: u32, x1: u32) -> Vec<f32> {
        (0..W * H)
            .map(|i| {
                if (x0..x1).contains(&(i % W)) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn run(e: &mut HaloEffect, mask: &mut [f32], value: u8) -> Vec<u8> {
        let mut data = vec![value; (W * H * 3) as usize];
        let mut plane = FramePlane::new(&mut data, W, H);
        let mask = MaskPlane::new(mask, W, H);
        e.process(&mut plane, &mask, &mut FrameContext::default())
            .expect("process");
        data
    }

    fn at(data: &[u8], x: u32) -> u8 {
        data[((H / 2 * W + x) * 3) as usize]
    }

    #[test]
    fn lighter_next_to_the_person_darker_far_away() {
        let mut e = configured("radius = 24\nglow = 0.2\ndim = 0.4");
        let mut mask = person(0, 40);
        let out = run(&mut e, &mut mask, 100);
        assert!(at(&out, 42) > 100, "next to: {}", at(&out, 42));
        assert!(at(&out, 150) == 60, "far: {}", at(&out, 150));
        // Monotonic fall-off away from the person.
        let profile: Vec<u8> = (40..W).map(|x| at(&out, x)).collect();
        assert!(profile.windows(2).all(|p| p[1] <= p[0]), "{profile:?}");
    }

    #[test]
    fn no_person_dims_everything_evenly() {
        let mut e = configured("dim = 0.5");
        let mut buf = Vec::new();
        let mask = filled_mask(&mut buf, W, H, 0.0);
        let mut data = vec![200_u8; (W * H * 3) as usize];
        let mut plane = FramePlane::new(&mut data, W, H);
        e.process(&mut plane, &mask, &mut FrameContext::default())
            .unwrap();
        assert!(data.iter().all(|&v| v == 100));
    }

    #[test]
    fn glow_saturates_instead_of_wrapping() {
        let mut e = configured("glow = 1.0\ndim = 0.0");
        let mut mask = vec![1.0_f32; (W * H) as usize];
        assert!(run(&mut e, &mut mask, 250).iter().all(|&v| v == 255));
    }

    #[test]
    fn zero_glow_and_dim_is_a_no_op() {
        let mut e = configured("glow = 0.0\ndim = 0.0");
        let mut mask = person(0, 40);
        assert!(run(&mut e, &mut mask, 77).iter().all(|&v| v == 77));
    }

    #[test]
    fn rejects_out_of_range_values_and_keeps_the_config() {
        let mut e = configured("radius = 30");
        for bad in [
            "radius = 7",
            "radius = 257",
            "glow = 1.1",
            "dim = -0.1",
            "x = 1",
        ] {
            assert!(e.configure(toml::from_str(bad).unwrap()).is_err(), "{bad}");
            assert_eq!(e.config.radius, 30, "{bad} changed the config");
        }
        for good in ["radius = 8", "radius = 256", "glow = 1.0", "dim = 1.0"] {
            e.configure(toml::from_str(good).unwrap())
                .unwrap_or_else(|err| panic!("{good}: {err}"));
        }
    }
}
