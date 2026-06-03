//! Auto exposure correction. Builds a 256-bin luminance histogram of
//! the plane, finds the low/high percentile cutoffs, and applies a
//! lookup-table mapping that combines a linear histogram stretch with
//! an optional gamma correction toward a target mean brightness.
//!
//! The effect is intentionally conservative: when the input already has
//! a wide dynamic range (`p_high - p_low >= GATE_RANGE`) the whole pass
//! short-circuits to a no-op. This keeps well-set-up webcams from being
//! re-graded.
//!
//! Use cases:
//! * `[foreground]` — fix backlit faces / dim office lighting on cheap
//!   webcams.
//! * `[background]` — uncommon; tends to wash out the backdrop.
//!
//! Cost: `O(width × height)` per frame — one histogram pass plus one
//! LUT-application pass. Both are trivially row-parallel.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::{
    CommitStrategy, DEBOUNCE_STANDARD_MS, EffectMetadata, ParamDescriptor, ParamKind, Scale,
};
use fluxframe_core::plane::{FramePlane, PlaneEffect};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

use super::helpers::{luma_rec601, reject_out_of_range};

/// Dynamic-range gate. If `p_high - p_low` is at least this many code
/// values the frame is already well-exposed and the effect short-circuits.
/// `200` corresponds to the histogram occupying ~78 % of the full
/// `[0, 255]` range — anything wider than that is almost certainly a
/// camera with proper auto-exposure.
const GATE_RANGE: u8 = 200;

/// Minimum dynamic range below which the linear stretch is suppressed
/// in favour of gamma-only correction toward `target_brightness`.
/// Stretching a near-flat histogram amplifies camera noise into
/// visible banding; `10` keeps the noise floor untouched while still
/// fixing the average brightness of low-contrast frames.
const MIN_STRETCH_RANGE: u8 = 10;

/// Lower clamp on `target_brightness`. Below `0.2` (~ 51 / 255) the
/// effect drags everything into shadows and produces a noticeably
/// muddy picture.
const MIN_TARGET_BRIGHTNESS: f32 = 0.2;
/// Upper clamp on `target_brightness`. Above `0.9` highlights pin to
/// 255 and the picture goes flat.
const MAX_TARGET_BRIGHTNESS: f32 = 0.9;
/// Maximum permitted `percentile_low`. Above `0.2` (20 %) the stretch
/// starts clipping mid-tones into pure black.
const MAX_PERCENTILE_LOW: f32 = 0.2;
/// Minimum permitted `percentile_high`. Below `0.8` (80 %) the stretch
/// blows highlights to pure white.
const MIN_PERCENTILE_HIGH: f32 = 0.8;

/// Default target mean luminance for the `target_brightness` field.
pub const DEFAULT_TARGET_BRIGHTNESS: f32 = 0.55;
/// Default lower percentile cutoff for the `percentile_low` field.
pub const DEFAULT_PERCENTILE_LOW: f32 = 0.05;
/// Default upper percentile cutoff for the `percentile_high` field.
pub const DEFAULT_PERCENTILE_HIGH: f32 = 0.95;

/// Symmetric guard the normalised post-stretch mean is clamped into
/// before feeding the logarithm. Keeps `mean = 0` or `mean = 255`
/// (fully black / fully white frames) from producing `log(0) = -inf`
/// and a NaN gamma. Picked tight enough that the gamma stays close to
/// the analytical value but loose enough that the logarithm output
/// has no representable overflow.
const MEAN_LOG_GUARD: f32 = 0.001;

/// TOML schema:
///
/// ```toml
/// [foreground.exposure_correct]
/// target_brightness = 0.55
/// percentile_low = 0.05
/// percentile_high = 0.95
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExposureCorrectConfig {
    /// Target mean luminance after correction, normalised to `[0, 1]`.
    /// `0.55` is a neutral choice for skin tones on video calls.
    /// Validated to be in `[MIN_TARGET_BRIGHTNESS, MAX_TARGET_BRIGHTNESS]`.
    #[serde(default = "default_target_brightness")]
    pub target_brightness: f32,
    /// Lower percentile of the luminance histogram mapped to `0` after
    /// the stretch. `0.05` rejects the darkest 5 % as noise.
    /// Validated to be in `[0.0, MAX_PERCENTILE_LOW]`.
    #[serde(default = "default_percentile_low")]
    pub percentile_low: f32,
    /// Upper percentile of the luminance histogram mapped to `255` after
    /// the stretch. `0.95` rejects the brightest 5 % as specular
    /// highlights. Validated to be in
    /// `[MIN_PERCENTILE_HIGH, 1.0]` and strictly greater than
    /// `percentile_low`.
    #[serde(default = "default_percentile_high")]
    pub percentile_high: f32,
}

fn default_target_brightness() -> f32 {
    DEFAULT_TARGET_BRIGHTNESS
}

fn default_percentile_low() -> f32 {
    DEFAULT_PERCENTILE_LOW
}

fn default_percentile_high() -> f32 {
    DEFAULT_PERCENTILE_HIGH
}

impl Default for ExposureCorrectConfig {
    fn default() -> Self {
        Self {
            target_brightness: default_target_brightness(),
            percentile_low: default_percentile_low(),
            percentile_high: default_percentile_high(),
        }
    }
}

/// `PlaneEffect` performing histogram-stretch + gamma exposure
/// correction.
pub struct ExposureCorrectEffect {
    config: ExposureCorrectConfig,
}

impl ExposureCorrectEffect {
    /// Effect name as registered in the plane registry.
    pub const NAME: &'static str = "exposure_correct";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Histogram stretch + gamma toward a target mean brightness.",
        params: &[
            ParamDescriptor {
                name: "target_brightness",
                kind: ParamKind::Float {
                    default: DEFAULT_TARGET_BRIGHTNESS,
                    min: MIN_TARGET_BRIGHTNESS,
                    max: MAX_TARGET_BRIGHTNESS,
                    step: 0.05,
                    scale: Scale::Linear,
                },
                help: "Target mean luminance after correction.",
                commit: CommitStrategy::Live {
                    debounce_ms: DEBOUNCE_STANDARD_MS,
                },
            },
            ParamDescriptor {
                name: "percentile_low",
                kind: ParamKind::Float {
                    default: DEFAULT_PERCENTILE_LOW,
                    min: 0.0,
                    max: MAX_PERCENTILE_LOW,
                    step: 0.01,
                    scale: Scale::Linear,
                },
                help: "Lower histogram percentile mapped to 0.",
                commit: CommitStrategy::Live {
                    debounce_ms: DEBOUNCE_STANDARD_MS,
                },
            },
            ParamDescriptor {
                name: "percentile_high",
                kind: ParamKind::Float {
                    default: DEFAULT_PERCENTILE_HIGH,
                    min: MIN_PERCENTILE_HIGH,
                    max: 1.0,
                    step: 0.01,
                    scale: Scale::Linear,
                },
                help: "Upper histogram percentile mapped to 255.",
                commit: CommitStrategy::Live {
                    debounce_ms: DEBOUNCE_STANDARD_MS,
                },
            },
        ],
    };

    /// Construct with default settings (target 0.55, percentiles 5/95)
    /// — must be `configure`d and `prepare`d before use.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ExposureCorrectConfig::default(),
        }
    }
}

impl Default for ExposureCorrectEffect {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the per-channel LUT applying the linear histogram stretch
/// followed by gamma correction toward `target_brightness`.
///
/// * Returns `None` when the frame is already well-exposed
///   (`p_high - p_low >= GATE_RANGE`); the caller short-circuits.
/// * When dynamic range is below `MIN_STRETCH_RANGE` the stretch is
///   skipped (would amplify noise on near-flat histograms) and only
///   the gamma adjustment runs.
fn build_lut(p_low: u8, p_high: u8, mean: f32, cfg: &ExposureCorrectConfig) -> Option<[u8; 256]> {
    let range = p_high.saturating_sub(p_low);
    if range >= GATE_RANGE {
        return None;
    }
    let do_stretch = range >= MIN_STRETCH_RANGE;

    // `mean_for_gamma` lives in `[0, 255]`. When the stretch is active,
    // estimate where the input mean lands after the stretch
    // analytically (saves a second histogram pass). When the stretch
    // is suppressed, work straight off the raw input mean.
    let mean_for_gamma = if do_stretch {
        ((mean - f32::from(p_low)) / f32::from(range) * 255.0).clamp(0.0, 255.0)
    } else {
        mean
    };

    let target = cfg.target_brightness;
    // Guard against pathological mean values that would explode the
    // logarithm; the clamp also handles a fully-black or fully-white
    // frame.
    let m_norm = (mean_for_gamma / 255.0).clamp(MEAN_LOG_GUARD, 1.0 - MEAN_LOG_GUARD);
    // gamma = log(target) / log(mean). gamma < 1 brightens midtones,
    // gamma > 1 darkens. Apply only when the (post-stretch) image is
    // darker than target — never pull bright frames down.
    let gamma = if m_norm < target {
        target.ln() / m_norm.ln()
    } else {
        1.0
    };

    let mut lut = [0u8; 256];
    for v in 0..=255u32 {
        let s = if do_stretch {
            // Linear stretch: `v → (v - p_low) / range`, clamped to
            // `[0, 1]`.
            let stretched = ((v as f32) - f32::from(p_low)) / f32::from(range);
            stretched.clamp(0.0, 1.0)
        } else {
            (v as f32) / 255.0
        };
        // Gamma toward target. `s.powf(gamma)` with `gamma < 1` lifts
        // midtones; `gamma == 1` is identity.
        let corrected = s.powf(gamma);
        lut[v as usize] = (corrected * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    Some(lut)
}

/// Compute the histogram, find percentile cut-offs and the mean
/// luminance.
///
/// Returns `(p_low, p_high, mean_luma)` where the percentiles are
/// 8-bit luminance values and `mean_luma` is the histogram's centre of
/// mass in `[0, 255]`.
fn analyse_histogram(data: &[u8], cfg: &ExposureCorrectConfig) -> (u8, u8, f32) {
    let mut hist = [0u32; 256];
    // Rec.601 luma weights are encapsulated in `luma_rec601`; the
    // helper guarantees output in `0..=255`, so the index is sound.
    for chunk in data.chunks_exact(3) {
        let y = luma_rec601([chunk[0], chunk[1], chunk[2]]);
        hist[y as usize] += 1;
    }
    let total: u32 = hist.iter().sum();
    if total == 0 {
        return (0, 255, 127.5);
    }
    let target_low = (f64::from(total) * f64::from(cfg.percentile_low)).round() as u32;
    let target_high = (f64::from(total) * f64::from(cfg.percentile_high)).round() as u32;
    let mut cumulative: u32 = 0;
    let mut p_low: u8 = 0;
    let mut p_high: u8 = 255;
    let mut p_low_found = false;
    for (i, &count) in hist.iter().enumerate() {
        cumulative += count;
        if !p_low_found && cumulative >= target_low {
            p_low = i as u8;
            p_low_found = true;
        }
        if cumulative >= target_high {
            p_high = i as u8;
            break;
        }
    }
    let weighted_sum: u64 = hist
        .iter()
        .enumerate()
        .map(|(i, &count)| u64::from(count) * (i as u64))
        .sum();
    let mean = (weighted_sum as f64) / f64::from(total);
    (p_low, p_high, mean as f32)
}

impl PlaneEffect for ExposureCorrectEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: ExposureCorrectConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        reject_out_of_range(
            Self::NAME,
            "target_brightness",
            cfg.target_brightness,
            MIN_TARGET_BRIGHTNESS,
            MAX_TARGET_BRIGHTNESS,
        )?;
        reject_out_of_range(
            Self::NAME,
            "percentile_low",
            cfg.percentile_low,
            0.0,
            MAX_PERCENTILE_LOW,
        )?;
        reject_out_of_range(
            Self::NAME,
            "percentile_high",
            cfg.percentile_high,
            MIN_PERCENTILE_HIGH,
            1.0,
        )?;
        // No cross-field check: per-field bounds (`MAX_PERCENTILE_LOW =
        // 0.2 < MIN_PERCENTILE_HIGH = 0.8`) make `percentile_low >=
        // percentile_high` unreachable.
        self.config = cfg;
        Ok(())
    }

    fn prepare(&mut self, _context: &ProcessingContext) -> Result<(), EffectError> {
        // Stateless: histogram and LUT are computed per-frame inside
        // `process` because the input statistics change every frame.
        Ok(())
    }

    fn process(
        &mut self,
        plane: &mut FramePlane<'_>,
        _ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
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
        let (p_low, p_high, mean) = analyse_histogram(plane.data, &self.config);
        let Some(lut) = build_lut(p_low, p_high, mean, &self.config) else {
            // Frame already well-exposed: short-circuit.
            return Ok(());
        };
        for byte in plane.data.iter_mut() {
            *byte = lut[*byte as usize];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(effect: &mut ExposureCorrectEffect, data: &mut [u8], w: u32, h: u32) {
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

    fn mean_luma(data: &[u8]) -> f32 {
        let mut sum = 0u64;
        let mut count = 0u64;
        for chunk in data.chunks_exact(3) {
            // Share the production Rec.601 formula so the test asserts
            // exactly what `analyse_histogram` measures.
            sum += u64::from(luma_rec601([chunk[0], chunk[1], chunk[2]]));
            count += 1;
        }
        (sum as f32) / (count as f32)
    }

    #[test]
    fn defaults_when_no_params() {
        let mut effect = ExposureCorrectEffect::new();
        let params: RawEffectParams = toml::Value::Table(toml::map::Map::new());
        effect.configure(params).expect("ok");
        assert!((effect.config.target_brightness - default_target_brightness()).abs() < 1e-6);
    }

    #[test]
    fn rejects_target_brightness_below_min() {
        let mut effect = ExposureCorrectEffect::new();
        let params: RawEffectParams = toml::from_str("target_brightness = 0.1").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn rejects_target_brightness_above_max() {
        let mut effect = ExposureCorrectEffect::new();
        let params: RawEffectParams = toml::from_str("target_brightness = 0.95").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn rejects_percentile_low_above_max() {
        let mut effect = ExposureCorrectEffect::new();
        let params: RawEffectParams = toml::from_str("percentile_low = 0.5").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn rejects_percentile_high_below_min() {
        let mut effect = ExposureCorrectEffect::new();
        let params: RawEffectParams = toml::from_str("percentile_high = 0.5").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn well_exposed_frame_is_noop() {
        // Synthetic frame whose luminance histogram spans 0..=255 in
        // 256 distinct values — dynamic range = 255 ≥ GATE_RANGE.
        let mut effect = ExposureCorrectEffect::new();
        effect
            .configure(toml::Value::Table(toml::map::Map::new()))
            .expect("configure");
        let mut data = Vec::with_capacity(256 * 3);
        for v in 0..=255u8 {
            data.extend_from_slice(&[v, v, v]);
        }
        let original = data.clone();
        run(&mut effect, &mut data, 256, 1);
        assert_eq!(data, original, "well-exposed frame must not be modified");
    }

    #[test]
    fn dark_frame_brightens() {
        // 32×32 frame filled with luma 60 (uniformly dark grey).
        // After exposure_correct the mean luma must move noticeably
        // toward the default target_brightness of 0.55 * 255 ≈ 140.
        let mut effect = ExposureCorrectEffect::new();
        effect
            .configure(toml::Value::Table(toml::map::Map::new()))
            .expect("configure");
        let mut data = vec![60u8; 32 * 32 * 3];
        let mean_before = mean_luma(&data);
        run(&mut effect, &mut data, 32, 32);
        let mean_after = mean_luma(&data);
        assert!(
            mean_after > mean_before + 10.0,
            "dark frame did not brighten: before={mean_before}, after={mean_after}"
        );
    }

    #[test]
    fn process_does_not_panic_on_constant_plane() {
        // Constant-luma frames have `range == 0`, which is below
        // `MIN_STRETCH_RANGE`. The stretch is suppressed and only the
        // gamma branch runs — verify it does not panic.
        let mut effect = ExposureCorrectEffect::new();
        effect
            .configure(toml::Value::Table(toml::map::Map::new()))
            .expect("configure");
        let mut data = vec![100u8; 16 * 16 * 3];
        run(&mut effect, &mut data, 16, 16);
        // The output should be a valid (possibly transformed) plane —
        // no panic and no NaN-driven garbage byte. All bytes must be
        // valid u8 (trivially true at the type level) — we just check
        // we got here.
        assert_eq!(data.len(), 16 * 16 * 3);
    }
}
