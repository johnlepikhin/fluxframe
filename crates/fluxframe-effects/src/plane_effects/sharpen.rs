//! Unsharp-mask sharpening. Applies a 3×3 separable Gaussian-style
//! blur to a working copy of the plane, then computes
//! `out = orig + amount * (orig - blurred)` channel-wise with `[0, 255]`
//! clamping.
//!
//! Use cases:
//! * `[foreground]` — restore detail on cheap webcams that smear faces.
//!   Mild amounts (0.2 – 0.5) read as "crisper" without the obvious
//!   "AI sharpened" halo.
//! * `[background]` — rarely useful; pairs poorly with `blur` which has
//!   already thrown detail away.
//!
//! Cost: `O(width × height)` per frame — three passes (horizontal blur,
//! vertical blur, blend).

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::{
    CommitStrategy, DEBOUNCE_FAST_MS, EffectMetadata, ParamDescriptor, ParamKind, Scale,
};
use fluxframe_core::plane::{FramePlane, PlaneEffect};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

use super::helpers::reject_out_of_range;
use crate::processing::parallel::{for_each_chunk_mut, for_each_row_mut};

/// Upper bound on `amount`. Past this, the unsharp formula produces
/// pronounced ringing halos rather than perceived sharpness; reject
/// loudly so the operator notices the misconfiguration.
const MAX_AMOUNT: f32 = 2.0;

/// Default sharpening strength for the `amount` field.
pub const DEFAULT_AMOUNT: f32 = 0.3;

/// TOML schema:
///
/// ```toml
/// [foreground.sharpen]
/// amount = 0.3
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharpenConfig {
    /// Strength of the sharpening kernel.
    ///
    /// * `0.0` is a no-op (the effect short-circuits and does no work).
    /// * `0.2 – 0.5` is the recommended range for video calls.
    /// * Values above `1.0` start to look artificially sharpened.
    ///
    /// Validated to be a finite number in `[0.0, MAX_AMOUNT]`.
    #[serde(default = "default_amount")]
    pub amount: f32,
}

fn default_amount() -> f32 {
    DEFAULT_AMOUNT
}

impl Default for SharpenConfig {
    fn default() -> Self {
        Self {
            amount: default_amount(),
        }
    }
}

/// `PlaneEffect` implementing a 3×3 separable-Gaussian unsharp mask.
pub struct SharpenEffect {
    config: SharpenConfig,
    /// Scratch for the horizontal-pass blur output. Reused as the
    /// source of the vertical pass.
    scratch_h: Vec<u8>,
    /// Scratch for the vertical-pass blur output — the "blurred" input
    /// to the unsharp combine.
    scratch_v: Vec<u8>,
    frame_w: u32,
    frame_h: u32,
}

impl SharpenEffect {
    /// Effect name as registered in the plane registry.
    pub const NAME: &'static str = "sharpen";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "3x3 separable unsharp-mask sharpening.",
        params: &[ParamDescriptor {
            name: "amount",
            kind: ParamKind::Float {
                default: DEFAULT_AMOUNT,
                min: 0.0,
                max: MAX_AMOUNT,
                step: 0.05,
                scale: Scale::Linear,
            },
            help: "Sharpening strength; 0 disables, 0.2-0.5 is typical.",
            commit: CommitStrategy::Live {
                debounce_ms: DEBOUNCE_FAST_MS,
            },
        }],
    };

    /// Construct with the default amount — must be `configure`d and
    /// `prepare`d before use.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SharpenConfig::default(),
            scratch_h: Vec::new(),
            scratch_v: Vec::new(),
            frame_w: 0,
            frame_h: 0,
        }
    }
}

impl Default for SharpenEffect {
    fn default() -> Self {
        Self::new()
    }
}

/// Horizontal 3-tap Gaussian-style blur with the `[1, 2, 1] / 4` kernel
/// and clamp-to-edge boundary. Output rows are independent (each reads
/// only its own `src` row), so they are dispatched row-parallel via the
/// shared primitive; `dst`'s row count fixes the height.
fn gauss_h(src: &[u8], dst: &mut [u8], width: usize) {
    let row_stride = width * 3;
    for_each_row_mut(dst, row_stride, |y, dst_row| {
        let src_row = &src[y * row_stride..y * row_stride + row_stride];
        for x in 0..width {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(width - 1);
            for c in 0..3 {
                let l = u16::from(src_row[xm * 3 + c]);
                let m = u16::from(src_row[x * 3 + c]);
                let r = u16::from(src_row[xp * 3 + c]);
                // `+ 2` is the standard rounding adjustment for
                // integer division by 4.
                dst_row[x * 3 + c] = ((l + 2 * m + r + 2) / 4) as u8;
            }
        }
    });
}

/// Vertical 3-tap Gaussian-style blur with the `[1, 2, 1] / 4` kernel
/// and clamp-to-edge boundary. Each output row reads three `src` rows
/// (`y-1`, `y`, `y+1`) from a separate scratch buffer, so rows are
/// independent and dispatched row-parallel; the inner byte-wise tap-sum
/// stays auto-vectorisable.
fn gauss_v(src: &[u8], dst: &mut [u8], width: usize, height: usize) {
    let row_stride = width * 3;
    for_each_row_mut(dst, row_stride, |y, dst_row| {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(height - 1);
        let top = &src[ym * row_stride..ym * row_stride + row_stride];
        let mid = &src[y * row_stride..y * row_stride + row_stride];
        let bot = &src[yp * row_stride..yp * row_stride + row_stride];
        for (i, out) in dst_row.iter_mut().enumerate() {
            let l = u16::from(top[i]);
            let m = u16::from(mid[i]);
            let r = u16::from(bot[i]);
            *out = ((l + 2 * m + r + 2) / 4) as u8;
        }
    });
}

/// Number of RGB bytes per parallel chunk in the unsharp blend. Each
/// byte is independent, so any chunking yields identical output; this
/// size mirrors the compositor's per-chunk granularity.
const COMBINE_CHUNK_BYTES: usize = 4_096 * 3;

/// Final unsharp blend: `out = orig + amount * (orig - blurred)`, with
/// channel-wise clamp to `[0, 255]`. Bytes are independent, so the blend
/// is dispatched in parallel chunks; `blurred` is sliced by the chunk's
/// global offset.
///
/// # Panics
///
/// Panics if `blurred.len() != plane_data.len()` (checked up front,
/// before any parallel work, to keep the panic out of the rayon worker).
fn unsharp_combine(plane_data: &mut [u8], blurred: &[u8], amount: f32) {
    assert_eq!(
        plane_data.len(),
        blurred.len(),
        "unsharp_combine: blurred length must equal plane_data",
    );
    for_each_chunk_mut(plane_data, COMBINE_CHUNK_BYTES, |off, chunk| {
        let blur_chunk = &blurred[off..off + chunk.len()];
        for (orig, blur) in chunk.iter_mut().zip(blur_chunk.iter()) {
            let o = f32::from(*orig);
            let b = f32::from(*blur);
            let mixed = o + amount * (o - b);
            *orig = mixed.clamp(0.0, 255.0) as u8;
        }
    });
}

impl PlaneEffect for SharpenEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: SharpenConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        reject_out_of_range(Self::NAME, "amount", cfg.amount, 0.0, MAX_AMOUNT)?;
        self.config = cfg;
        Ok(())
    }

    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError> {
        self.frame_w = context.width;
        self.frame_h = context.height;
        let frame_bytes = (context.width as usize) * (context.height as usize) * 3;
        self.scratch_h = vec![0u8; frame_bytes];
        self.scratch_v = vec![0u8; frame_bytes];
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
        // Dimension mismatch is a contract violation: the supervisor
        // re-runs `prepare` whenever the frame size changes.
        if plane.width != self.frame_w || plane.height != self.frame_h {
            return Err(EffectError::ProcessFailed {
                name: Self::NAME.to_string(),
                reason: format!(
                    "plane dimensions {}x{} differ from prepared {}x{}",
                    plane.width, plane.height, self.frame_w, self.frame_h
                ),
            });
        }
        // Short-circuit the no-op case so an `amount = 0.0` default does
        // not waste two passes plus a blend.
        if self.config.amount == 0.0 {
            return Ok(());
        }
        debug_assert_eq!(
            plane.data.len(),
            width * height * 3,
            "FramePlane invariant: data.len() must equal width * height * 3",
        );

        gauss_h(plane.data, &mut self.scratch_h, width);
        gauss_v(&self.scratch_h, &mut self.scratch_v, width, height);
        unsharp_combine(plane.data, &self.scratch_v, self.config.amount);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(effect: &mut SharpenEffect, data: &mut [u8], w: u32, h: u32) {
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
        let mut effect = SharpenEffect::new();
        let params: RawEffectParams = toml::Value::Table(toml::map::Map::new());
        effect.configure(params).expect("ok");
        assert!((effect.config.amount - default_amount()).abs() < 1e-6);
    }

    #[test]
    fn rejects_negative_amount() {
        let mut effect = SharpenEffect::new();
        let params: RawEffectParams = toml::from_str("amount = -0.5").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn rejects_excessive_amount() {
        let mut effect = SharpenEffect::new();
        let raw = format!("amount = {}", MAX_AMOUNT + 0.1);
        let params: RawEffectParams = toml::from_str(&raw).unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn rejects_nan_amount() {
        let mut effect = SharpenEffect::new();
        let params: RawEffectParams = toml::from_str("amount = nan").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn amount_zero_is_identity() {
        // `amount = 0.0` must short-circuit to a true no-op so a
        // chained-but-disabled sharpen pass costs nothing.
        let mut effect = SharpenEffect::new();
        let params: RawEffectParams = toml::from_str("amount = 0.0").unwrap();
        effect.configure(params).expect("configure");
        let mut data: Vec<u8> = vec![
            10, 20, 30, 40, 50, 60, //
            70, 80, 90, 100, 110, 120, //
        ];
        let original = data.clone();
        run(&mut effect, &mut data, 2, 2);
        assert_eq!(data, original, "amount=0 must be a no-op");
    }

    #[test]
    fn sharpening_increases_contrast_on_edge() {
        // 4×1 horizontal step edge — dark | dark | light | light.
        // After unsharp with amount=1.0, the boundary pixels should pull
        // further apart (dark side darkens, light side lightens).
        let mut effect = SharpenEffect::new();
        let params: RawEffectParams = toml::from_str("amount = 1.0").unwrap();
        effect.configure(params).expect("configure");
        let mut data: Vec<u8> = vec![
            50, 50, 50, 50, 50, 50, // x = 0, x = 1
            200, 200, 200, 200, 200, 200, // x = 2, x = 3
        ];
        run(&mut effect, &mut data, 4, 1);
        // Pixel at x=1 (dark side of the edge): R-channel at byte 3.
        assert!(data[3] < 50, "edge-dark pixel did not darken: {}", data[3]);
        // Pixel at x=2 (light side of the edge): R-channel at byte 6.
        assert!(
            data[6] > 200,
            "edge-light pixel did not lighten: {}",
            data[6]
        );
    }

    #[test]
    fn constant_plane_is_unchanged() {
        // A constant-coloured plane has zero high-frequency content, so
        // the unsharp diff is zero and the output equals the input even
        // at the maximum permitted amount.
        let mut effect = SharpenEffect::new();
        let params: RawEffectParams = toml::from_str("amount = 2.0").unwrap();
        effect.configure(params).expect("configure");
        let mut data_white: Vec<u8> = vec![255; 4 * 4 * 3];
        run(&mut effect, &mut data_white, 4, 4);
        assert!(
            data_white.iter().all(|&v| v == 255),
            "constant white must stay 255"
        );

        let mut effect2 = SharpenEffect::new();
        effect2
            .configure(toml::from_str("amount = 2.0").unwrap())
            .unwrap();
        let mut data_black: Vec<u8> = vec![0; 4 * 4 * 3];
        run(&mut effect2, &mut data_black, 4, 4);
        assert!(
            data_black.iter().all(|&v| v == 0),
            "constant black must stay 0"
        );
    }

    #[test]
    fn clamps_at_extremes_without_overflow() {
        // 4×1 with an isolated bright pixel at the end. With amount=2
        // the dark-neighbour update goes negative and the light-pixel
        // update overshoots 255; both must clamp without wrapping.
        let mut effect = SharpenEffect::new();
        let params: RawEffectParams = toml::from_str("amount = 2.0").unwrap();
        effect.configure(params).expect("configure");
        let mut data: Vec<u8> = vec![
            0, 0, 0, 0, 0, 0, //
            0, 0, 0, 255, 255, 255, //
        ];
        run(&mut effect, &mut data, 4, 1);
        // No wrap-around: every byte must remain a valid u8 channel.
        // (Implicit in u8 storage — the test verifies the clamp logic
        // produces 0/255 boundary values rather than panicking or
        // producing intermediate garbage.)
        assert_eq!(data[0], 0);
        assert_eq!(data[9], 255);
    }

    #[test]
    fn rejects_process_on_dimension_mismatch() {
        // Prepare for 8x8 then submit a 4x4 plane — must fail loudly
        // because the supervisor contract is to re-prepare on resize.
        let mut effect = SharpenEffect::new();
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
    fn configure_after_prepare_is_safe() {
        // Live-reconfig contract (Stage 13): `configure()` is re-callable
        // after `prepare()`. Verify the next `process()` reflects the
        // new `amount` without panicking.
        let mut effect = SharpenEffect::new();
        effect
            .configure(toml::from_str("amount = 0.0").unwrap())
            .expect("configure 1");
        run(&mut effect, &mut [128u8; 4 * 4 * 3].to_vec(), 4, 4);

        effect
            .configure(toml::from_str("amount = 1.5").unwrap())
            .expect("re-configure ok");
        // Edge case: 4×1 contrast bump exercised in the existing
        // `sharpening_increases_contrast_on_edge` test. Here we only
        // verify the re-configure path does not panic and process()
        // runs against the same prepared scratch.
        let mut data = vec![100u8; 4 * 4 * 3];
        let mut plane = FramePlane::new(&mut data, 4, 4);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("process ok");
        assert!((effect.config.amount - 1.5).abs() < 1e-6);
    }
}
