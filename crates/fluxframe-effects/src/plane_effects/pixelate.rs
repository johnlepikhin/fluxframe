//! Block-averaging pixelation. Each `block_size × block_size` square
//! of the plane is replaced with its mean RGB colour.
//!
//! Use cases:
//! * `[background]` — soft mosaic-style backdrop, cheaper than a
//!   Gaussian blur and visually similar at low frequencies.
//! * `[foreground]` — censorship-style "pixelated person" effect when
//!   the operator wants the subject obscured rather than replaced.
//!
//! Cost is `O(width × height)` per frame — one read pass to compute
//! per-block sums plus one write pass to broadcast the averages.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::plane::{FramePlane, PlaneEffect};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

/// Smallest accepted `block_size`. `1` would be a no-op (each pixel
/// is its own block); rejecting it surfaces the misconfiguration
/// instead of silently doing nothing.
const MIN_BLOCK_SIZE: u32 = 2;
/// Arbitrary safety cap.  Above this the output is indistinguishable
/// from a single solid colour even on 4K frames, and the `u32`
/// accumulator must stay well under `u32::MAX` (`256^2 * 255 ≈ 1.67e7`
/// today — see invariant note in `process`).
const MAX_BLOCK_SIZE: u32 = 256;

/// TOML schema:
///
/// ```toml
/// [background.pixelate]
/// block_size = 16
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PixelateConfig {
    /// Block edge length in pixels. Each `block_size × block_size`
    /// region of the plane is averaged into a single colour.
    /// Larger values give a chunkier mosaic.  Range
    /// `MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE` — see those constants for
    /// the rationale.
    #[serde(default = "default_block_size")]
    pub block_size: u32,
}

fn default_block_size() -> u32 {
    16
}

impl Default for PixelateConfig {
    fn default() -> Self {
        Self {
            block_size: default_block_size(),
        }
    }
}

/// `PlaneEffect` performing block-averaging pixelation.
pub struct PixelateEffect {
    block_size: u32,
}

impl PixelateEffect {
    /// Effect name as registered in the plane registry.
    pub const NAME: &'static str = "pixelate";

    /// Construct with the default block size.
    #[must_use]
    pub fn new() -> Self {
        Self {
            block_size: default_block_size(),
        }
    }
}

impl Default for PixelateEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl PlaneEffect for PixelateEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: PixelateConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        if !(MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE).contains(&cfg.block_size) {
            return Err(super::invalid_config(
                Self::NAME,
                format!(
                    "block_size must be in {MIN_BLOCK_SIZE}..={MAX_BLOCK_SIZE}, got {}",
                    cfg.block_size
                ),
            ));
        }
        self.block_size = cfg.block_size;
        Ok(())
    }

    fn prepare(&mut self, _context: &ProcessingContext) -> Result<(), EffectError> {
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
        let block = self.block_size as usize;
        debug_assert!(block >= MIN_BLOCK_SIZE as usize, "validated in configure");
        // Two passes per block: accumulate RGB sums, then broadcast
        // the mean. Each pixel is visited exactly twice across the
        // whole frame — O(W*H) total.
        //
        // `u32` accumulator invariant: a single block contributes at
        // most `MAX_BLOCK_SIZE^2 * 255 = 256 * 256 * 255 ≈ 1.67e7`,
        // well under `u32::MAX ≈ 4.29e9`. If `MAX_BLOCK_SIZE` is ever
        // raised above ~4095, widen the accumulator to `u64`.
        let row_stride = width * 3;
        for by in (0..height).step_by(block) {
            let block_h = block.min(height - by);
            for bx in (0..width).step_by(block) {
                let block_w = block.min(width - bx);
                let pixel_offset = bx * 3;
                let pixel_len = block_w * 3;
                // Accumulate. Hoist the per-row slice once so the
                // per-pixel inner loop sees a fresh small slice with
                // length known to the optimiser (and a single
                // bounds-check per row instead of per pixel).
                let mut sum_r: u32 = 0;
                let mut sum_g: u32 = 0;
                let mut sum_b: u32 = 0;
                for y in by..(by + block_h) {
                    let row_start = y * row_stride + pixel_offset;
                    let row = &plane.data[row_start..row_start + pixel_len];
                    for px in row.chunks_exact(3) {
                        sum_r += u32::from(px[0]);
                        sum_g += u32::from(px[1]);
                        sum_b += u32::from(px[2]);
                    }
                }
                let area = (block_w * block_h) as u32;
                // `sum_? / area <= 255` by construction (each pixel
                // channel is `u8`), so the `as u8` cast cannot truncate.
                let avg_r = (sum_r / area) as u8;
                let avg_g = (sum_g / area) as u8;
                let avg_b = (sum_b / area) as u8;
                // Broadcast — same row-slice hoisting as the accumulate
                // pass.
                for y in by..(by + block_h) {
                    let row_start = y * row_stride + pixel_offset;
                    let row = &mut plane.data[row_start..row_start + pixel_len];
                    for px in row.chunks_exact_mut(3) {
                        px[0] = avg_r;
                        px[1] = avg_g;
                        px[2] = avg_b;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(effect: &mut PixelateEffect, data: &mut [u8], w: u32, h: u32) {
        let mut plane = FramePlane::new(data, w, h);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("ok");
    }

    #[test]
    fn defaults_when_no_params() {
        let mut effect = PixelateEffect::new();
        let params: RawEffectParams = toml::Value::Table(toml::map::Map::new());
        effect.configure(params).expect("ok");
        assert_eq!(effect.block_size, default_block_size());
    }

    #[test]
    fn rejects_zero_block_size() {
        let mut effect = PixelateEffect::new();
        let params: RawEffectParams = toml::from_str("block_size = 0").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn rejects_excessive_block_size() {
        let mut effect = PixelateEffect::new();
        let raw = format!("block_size = {}", MAX_BLOCK_SIZE + 1);
        let params: RawEffectParams = toml::from_str(&raw).unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn rejects_block_size_one() {
        // `block_size = 1` would be a no-op (each pixel is its own
        // block); reject it loudly so the operator notices the
        // misconfiguration instead of silently getting an identity
        // transform.
        let mut effect = PixelateEffect::new();
        let params: RawEffectParams = toml::from_str("block_size = 1").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn accepts_min_block_size_boundary() {
        // `MIN_BLOCK_SIZE = 2` must validate successfully.
        let mut effect = PixelateEffect::new();
        let raw = format!("block_size = {MIN_BLOCK_SIZE}");
        let params: RawEffectParams = toml::from_str(&raw).unwrap();
        effect.configure(params).expect("min boundary accepted");
        assert_eq!(effect.block_size, MIN_BLOCK_SIZE);
    }

    #[test]
    fn accepts_max_block_size_boundary() {
        // `MAX_BLOCK_SIZE = 256` must validate successfully (off-by-one
        // guard for the `..=MAX` inclusive range).
        let mut effect = PixelateEffect::new();
        let raw = format!("block_size = {MAX_BLOCK_SIZE}");
        let params: RawEffectParams = toml::from_str(&raw).unwrap();
        effect.configure(params).expect("max boundary accepted");
        assert_eq!(effect.block_size, MAX_BLOCK_SIZE);
    }

    #[test]
    fn averages_2x2_block() {
        // 2x2 plane with four distinct pixels — one block at block_size=2.
        // R channel:   10  90        avg = (10+90+50+30)/4 = 45
        // G channel:   20 100        avg = 60
        // B channel:   30 110        avg = 70
        // Row 2 R:     50  30
        //       G:     60  40
        //       B:     70  50
        let mut effect = PixelateEffect::new();
        let params: RawEffectParams = toml::from_str("block_size = 2").unwrap();
        effect.configure(params).expect("configure ok");
        let mut data: Vec<u8> = vec![
            10, 20, 30, // (0,0)
            90, 100, 110, // (1,0)
            50, 60, 70, // (0,1)
            30, 40, 50, // (1,1)
        ];
        run(&mut effect, &mut data, 2, 2);
        // After averaging every pixel of the block is (45, 55, 65).
        // (sum_r = 180, avg = 45; sum_g = 220, avg = 55; sum_b = 260, avg = 65)
        for chunk in data.chunks_exact(3) {
            assert_eq!(chunk[0], 45, "R");
            assert_eq!(chunk[1], 55, "G");
            assert_eq!(chunk[2], 65, "B");
        }
    }

    #[test]
    fn partial_edge_block_handled() {
        // 3-wide plane with block_size = 2 — leftmost block is 2x1,
        // rightmost block is 1x1 (the orphan column on the right).
        let mut effect = PixelateEffect::new();
        let params: RawEffectParams = toml::from_str("block_size = 2").unwrap();
        effect.configure(params).expect("configure ok");
        // 3x1 plane, pixels (10,20,30), (50,60,70), (90,100,110).
        let mut data: Vec<u8> = vec![10, 20, 30, 50, 60, 70, 90, 100, 110];
        run(&mut effect, &mut data, 3, 1);
        // Left block (2x1): avg = ((10+50)/2, (20+60)/2, (30+70)/2) = (30, 40, 50).
        assert_eq!(&data[0..3], &[30, 40, 50]);
        assert_eq!(&data[3..6], &[30, 40, 50]);
        // Right orphan block (1x1) — keeps its own values.
        assert_eq!(&data[6..9], &[90, 100, 110]);
    }

    #[test]
    fn block_larger_than_plane_collapses_to_single_block() {
        let mut effect = PixelateEffect::new();
        let params: RawEffectParams = toml::from_str("block_size = 8").unwrap();
        effect.configure(params).expect("configure ok");
        // 2x2 plane — block_size = 8 collapses the whole plane to one
        // averaged colour.
        let mut data: Vec<u8> = vec![
            0, 0, 0, // black
            255, 0, 0, // red
            0, 255, 0, // green
            0, 0, 255, // blue
        ];
        run(&mut effect, &mut data, 2, 2);
        // Average across four pixels: R = (0+255+0+0)/4 = 63
        //                              G = (0+0+255+0)/4 = 63
        //                              B = (0+0+0+255)/4 = 63
        for chunk in data.chunks_exact(3) {
            assert_eq!(chunk, [63, 63, 63]);
        }
    }

    #[test]
    fn configure_after_prepare_is_safe() {
        // Live-reconfig contract (Stage 13): configure() may be re-called
        // after prepare(). Verify the next process() reflects the new
        // `block_size` (visible as a different averaging granularity).
        let mut effect = PixelateEffect::new();
        effect
            .configure(toml::from_str("block_size = 2").unwrap())
            .expect("configure 1");
        let context = ProcessingContext {
            width: 8,
            height: 8,
            format: fluxframe_core::PixelFormat::Rgb,
            fps: 30,
            counters: None,
        };
        effect.prepare(&context).expect("prepare");

        // Seed: 4x4 plane with a per-pixel gradient. With block_size=2
        // each 2x2 sub-block averages independently → 4 distinct colours
        // after process. With block_size=4 the whole plane collapses to
        // one colour.
        let make_data = || -> Vec<u8> {
            let mut v = Vec::with_capacity(4 * 4 * 3);
            for i in 0..16u8 {
                v.extend_from_slice(&[i * 16, i * 16, i * 16]);
            }
            v
        };

        let mut data = make_data();
        run(&mut effect, &mut data, 4, 4);
        // With block_size=2: at least the top-left block differs from
        // the bottom-right block — sanity check the partition.
        assert_ne!(&data[0..3], &data[(4 * 4 - 1) * 3..(4 * 4) * 3]);

        // Re-configure with bigger block.
        effect
            .configure(toml::from_str("block_size = 4").unwrap())
            .expect("re-configure ok");
        assert_eq!(effect.block_size, 4);

        let mut data = make_data();
        run(&mut effect, &mut data, 4, 4);
        // With block_size=4 the whole 4x4 plane collapses to one colour.
        let first = data[0..3].to_vec();
        for chunk in data.chunks_exact(3) {
            assert_eq!(
                chunk,
                first.as_slice(),
                "block_size=4 must paint one colour"
            );
        }
    }
}
