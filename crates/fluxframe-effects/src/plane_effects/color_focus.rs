//! Colour focus: the room keeps its colour next to the person and fades
//! towards grey away from them.
//!
//! Like `halo`, a vignette centred on the silhouette, but it takes the
//! colour away instead of the light: attention goes to the person while
//! the room keeps its brightness. The mask is blurred by `radius`
//! pixels; each pixel keeps `1 - fade × (1 - closeness)` of its
//! saturation around its Rec.601 luma.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::{DEBOUNCE_FAST_MS, DEBOUNCE_STANDARD_MS, EffectMetadata};
use fluxframe_core::plane::{FramePlane, MaskPlane, PlaneEffect};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

use super::helpers::{
    check_mask_size, float_param, integer_param, luma_rec601, reject_int_out_of_range,
    reject_out_of_range,
};
use crate::processing::soft_mask::{SoftMask, closeness, map_rows};

const MIN_RADIUS: i64 = 8;
const MAX_RADIUS: i64 = 256;

/// Default reach of the colour, in pixels.
pub const DEFAULT_RADIUS: u32 = 80;
/// Default loss of colour far from the person.
pub const DEFAULT_FADE: f32 = 0.8;

/// TOML schema:
///
/// ```toml
/// [background.color_focus]
/// radius = 80
/// fade = 0.8
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColorFocusConfig {
    /// How far the colour reaches from the person, in pixels.
    #[serde(default = "default_radius")]
    pub radius: u32,
    /// Colour lost far from the person: `0.0` none, `1.0` grey.
    #[serde(default = "default_fade")]
    pub fade: f32,
}

fn default_radius() -> u32 {
    DEFAULT_RADIUS
}
fn default_fade() -> f32 {
    DEFAULT_FADE
}

impl Default for ColorFocusConfig {
    fn default() -> Self {
        Self {
            radius: default_radius(),
            fade: default_fade(),
        }
    }
}

/// Scale the saturation of `px` around its luma by `keep / 256`.
fn desaturate(px: &mut [u8], keep: u32) {
    let y = i32::from(luma_rec601([px[0], px[1], px[2]]));
    // `keep <= 256` by construction.
    let keep = i32::try_from(keep).unwrap_or(256);
    for c in px {
        let v = y + ((i32::from(*c) - y) * keep + 128).div_euclid(256);
        *c = v.clamp(0, 255) as u8;
    }
}

/// `PlaneEffect` draining colour from the plane away from the person.
pub struct ColorFocusEffect {
    config: ColorFocusConfig,
    soft: SoftMask,
}

impl ColorFocusEffect {
    /// Effect name as registered in the plane registry.
    pub const NAME: &'static str = "color_focus";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Keep colour next to the person and fade it to grey far away (uses the mask).",
        params: &[
            integer_param(
                "radius",
                DEFAULT_RADIUS as i64,
                MIN_RADIUS,
                MAX_RADIUS,
                DEBOUNCE_STANDARD_MS,
                "How far the colour reaches from the person, in pixels.",
            ),
            float_param(
                "fade",
                DEFAULT_FADE,
                0.0,
                1.0,
                DEBOUNCE_FAST_MS,
                "Colour lost far from the person: 0 none, 1 grey.",
            ),
        ],
    };

    /// Construct with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ColorFocusConfig::default(),
            soft: SoftMask::new(),
        }
    }
}

impl Default for ColorFocusEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl PlaneEffect for ColorFocusEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: ColorFocusConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        reject_int_out_of_range(
            Self::NAME,
            "radius",
            cfg.radius.into(),
            MIN_RADIUS,
            MAX_RADIUS,
        )?;
        reject_out_of_range(Self::NAME, "fade", cfg.fade, 0.0, 1.0)?;
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
        let ColorFocusConfig { radius, fade } = self.config;
        if width == 0 || plane.height == 0 || fade == 0.0 {
            return Ok(());
        }
        self.soft
            .update(mask.data, plane.width, plane.height, radius);
        map_rows(
            plane.data,
            width,
            &self.soft,
            (0, 0),
            |s| ((1.0 - fade * (1.0 - closeness(s))) * 256.0 + 0.5) as u32,
            |px, keep| {
                if keep < 256 {
                    desaturate(px, keep);
                }
            },
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{mask_from, run_masked};

    const W: u32 = 160;
    const H: u32 = 64;
    const ROOM: [u8; 3] = [200, 120, 40];

    fn configured(cfg: &str) -> ColorFocusEffect {
        let mut e = ColorFocusEffect::new();
        e.configure(toml::from_str(cfg).unwrap())
            .expect("configure");
        e
    }

    /// Person in columns `0..40`, full height; returns the row profile.
    fn profile(e: &mut ColorFocusEffect, fill: [u8; 3]) -> Vec<[u8; 3]> {
        let mut mask = mask_from(W, H, |x, _| if x < 40 { 1.0 } else { 0.0 });
        let out = run_masked(e, &mut mask, W, H, fill);
        let row = (H / 2 * W * 3) as usize;
        out[row..row + (W * 3) as usize]
            .chunks_exact(3)
            .map(|p| [p[0], p[1], p[2]])
            .collect()
    }

    fn spread(px: [u8; 3]) -> u8 {
        px.iter().max().unwrap() - px.iter().min().unwrap()
    }

    #[test]
    fn colour_next_to_the_person_grey_far_away() {
        let mut e = configured("radius = 24\nfade = 1.0");
        let row = profile(&mut e, ROOM);
        // Right next to the person almost all the colour is kept (the
        // blurred mask is a little under one half just outside the edge).
        assert!(
            u32::from(spread(row[41])) * 10 >= u32::from(spread(ROOM)) * 9,
            "{:?}",
            row[41]
        );
        let grey = luma_rec601(ROOM);
        assert_eq!(row[150], [grey; 3]);
        // Saturation falls monotonically away from the person.
        let s: Vec<u8> = (40..W as usize).map(|x| spread(row[x])).collect();
        assert!(s.windows(2).all(|p| p[1] <= p[0]), "{s:?}");
    }

    #[test]
    fn partial_fade_keeps_some_colour() {
        let mut e = configured("radius = 24\nfade = 0.5");
        let far = profile(&mut e, ROOM)[150];
        let s = spread(far);
        assert!((70..=90).contains(&s), "{far:?}");
    }

    #[test]
    fn grey_stays_grey() {
        let mut e = configured("fade = 1.0");
        assert!(profile(&mut e, [90; 3]).iter().all(|p| *p == [90; 3]));
    }

    #[test]
    fn zero_fade_is_a_no_op() {
        let mut e = configured("fade = 0.0");
        assert!(profile(&mut e, ROOM).iter().all(|p| *p == ROOM));
    }

    #[test]
    fn rejects_out_of_range_values_and_keeps_the_config() {
        let mut e = configured("radius = 30");
        for bad in ["radius = 7", "radius = 257", "fade = 1.1", "x = 1"] {
            assert!(e.configure(toml::from_str(bad).unwrap()).is_err(), "{bad}");
            assert_eq!(e.config.radius, 30, "{bad} changed the config");
        }
        for good in ["radius = 8", "radius = 256", "fade = 0.0", "fade = 1.0"] {
            e.configure(toml::from_str(good).unwrap())
                .unwrap_or_else(|err| panic!("{good}: {err}"));
        }
    }
}
