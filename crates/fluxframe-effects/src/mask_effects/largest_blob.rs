//! Keep only the largest connected component of the mask; zero out
//! every smaller blob and every below-threshold pixel.
//!
//! Use case: segmentation models occasionally emit small spurious
//! "islands" of foreground confidence outside the actual subject —
//! reflections in mirrors, a coat on a chair, the operator's elbow
//! sticking out of frame. The person itself is by far the largest
//! contiguous high-confidence region, so a connected-component
//! analysis with "keep the largest" rule removes the noise without
//! touching the subject.
//!
//! Algorithm: BFS labeling with 4-connectivity over the binary mask
//! `mask >= level`. After labeling, every pixel not in the largest
//! component is set to `0.0`; pixels in the largest component keep
//! their original confidence value (so a subsequent `feather` step
//! still produces soft edges).
//!
//! Cost is linear in the number of pixels — at 256×256 model
//! resolution the loop runs over 65 k entries, well below 1 ms on a
//! single core.
//!
//! Contract: `LargestBlobConfig::level` must lie in `[0.0, 1.0]`;
//! [`MaskEffect::configure`] returns [`EffectError::InvalidConfig`]
//! otherwise.
//!
//! Tie-breaking: when two components have equal size, the one
//! encountered first in row-major scan order wins. The result is
//! deterministic for a given frame, but the chosen component may
//! change between frames when the second-place component shifts.

use std::collections::VecDeque;

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::plane::{MaskEffect, MaskPlane};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

/// TOML schema: `level = 0.5` (default).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LargestBlobConfig {
    /// Pixels with `mask >= level` participate in the connected-component
    /// labeling; pixels below the level are treated as background and
    /// zeroed in the output. Default `0.5`, range `[0, 1]`.
    #[serde(default = "default_level")]
    pub level: f32,
}

/// Default segmentation threshold for the `level` field — pixels with
/// `mask >= DEFAULT_LEVEL` are treated as foreground.
pub const DEFAULT_LEVEL: f32 = 0.5;

fn default_level() -> f32 {
    DEFAULT_LEVEL
}

impl Default for LargestBlobConfig {
    fn default() -> Self {
        Self {
            level: default_level(),
        }
    }
}

/// `MaskEffect` that retains only the largest connected component.
///
/// Tie-breaking: equal-sized components are decided by row-major scan
/// order — the first component reached wins. See the module-level
/// docs for the implication on frame-to-frame stability.
#[derive(Default)]
pub struct LargestBlobMaskEffect {
    level: f32,
    /// Per-pixel component label. `0` = unlabeled (background or not
    /// yet visited); ≥ 1 = component id assigned during BFS.
    /// Allocated lazily on first `process` so we do not need the mask
    /// dimensions at `prepare` time.
    labels: Vec<u32>,
    /// BFS frontier. Reused across calls — `clear()` is O(1) and
    /// keeps capacity.
    queue: VecDeque<usize>,
}

impl LargestBlobMaskEffect {
    /// Effect name as registered in the mask registry.
    pub const NAME: &'static str = "largest_blob";

    /// Construct with the default threshold.
    #[must_use]
    pub fn new() -> Self {
        Self {
            level: default_level(),
            labels: Vec::new(),
            queue: VecDeque::new(),
        }
    }
}

impl MaskEffect for LargestBlobMaskEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: LargestBlobConfig = params
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
        let width = mask.width as usize;
        let height = mask.height as usize;
        let pixels = width * height;
        if pixels == 0 {
            return Ok(());
        }
        debug_assert_eq!(
            mask.data.len(),
            pixels,
            "MaskPlane invariant: data.len() must equal width * height",
        );

        // Lazy scratch allocation. When the buffer must grow, `resize`
        // already zeroes the newly appended slots; otherwise we only
        // need to reset the prefix we are about to use. Capacity is
        // preserved across calls so the steady-state hot loop does no
        // allocation.
        if self.labels.len() < pixels {
            self.labels.resize(pixels, 0);
        } else {
            self.labels[..pixels].fill(0);
        }

        let level = self.level;
        let mut next_label: u32 = 1;
        let mut largest_label: u32 = 0;
        let mut largest_size: usize = 0;

        // Pass 1: BFS labeling. `idx` walks the mask in row-major order;
        // each foreground pixel that has not been labeled yet seeds a
        // fresh component.
        for seed in 0..pixels {
            if mask.data[seed] < level || self.labels[seed] != 0 {
                continue;
            }
            self.queue.clear();
            self.queue.push_back(seed);
            self.labels[seed] = next_label;
            let mut size: usize = 0;

            while let Some(idx) = self.queue.pop_front() {
                size += 1;
                let x = idx % width;
                let y = idx / width;
                // 4-connectivity neighbours. The `x > 0` / `y > 0`
                // checks keep `idx - 1` and `idx - width` from
                // underflowing usize when the current pixel is on the
                // top/left edge; the upper-bound checks symmetrically
                // guard the right/bottom edges.
                let candidates = [
                    (x > 0).then(|| idx - 1),
                    (x + 1 < width).then(|| idx + 1),
                    (y > 0).then(|| idx - width),
                    (y + 1 < height).then(|| idx + width),
                ];
                for nidx in candidates.into_iter().flatten() {
                    if mask.data[nidx] >= level && self.labels[nidx] == 0 {
                        self.labels[nidx] = next_label;
                        self.queue.push_back(nidx);
                    }
                }
            }

            if size > largest_size {
                largest_size = size;
                largest_label = next_label;
            }
            // Pixel count fits in u32 on any conceivable mask
            // resolution (mask resolution * model resolution ≤ 4G px),
            // so the label counter cannot overflow in practice. Keep
            // an explicit debug assertion so a future codebase change
            // that introduces giant masks does not silently merge
            // components.
            debug_assert!(next_label < u32::MAX, "component label overflow");
            next_label += 1;
        }

        // Pass 2: zero out pixels that are not in the largest component.
        // `largest_label == 0` only when the seed loop ran zero times —
        // i.e. no input pixel was at or above `level`. In that case
        // every `labels[i]` is still 0 (unlabeled) too, so the `!=`
        // check still zeroes the whole mask. Equivalent to: "no
        // foreground → all-zero output".
        for (m, &label) in mask.data.iter_mut().zip(self.labels.iter()) {
            if label != largest_label {
                *m = 0.0;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::float_cmp,
    reason = "Test inputs use only exact 0.0 / 1.0 values; equality compares correctly."
)]
mod tests {
    use super::*;

    fn run(effect: &mut LargestBlobMaskEffect, data: &mut [f32], width: u32, height: u32) {
        let mut plane = MaskPlane::new(data, width, height);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("ok");
    }

    #[test]
    fn defaults_when_no_params() {
        let mut effect = LargestBlobMaskEffect::new();
        let params: RawEffectParams = toml::Value::Table(toml::map::Map::new());
        effect.configure(params).expect("ok");
        assert!((effect.level - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rejects_out_of_range_level() {
        let mut effect = LargestBlobMaskEffect::new();
        let params: RawEffectParams = toml::from_str("level = 1.5").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn all_zero_stays_zero() {
        let mut effect = LargestBlobMaskEffect::new();
        let mut data = vec![0.0_f32; 16];
        run(&mut effect, &mut data, 4, 4);
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn single_blob_is_preserved() {
        let mut effect = LargestBlobMaskEffect::new();
        // 4x4 with the centre 2x2 block above threshold.
        let mut data = vec![0.0_f32; 16];
        for &i in &[5, 6, 9, 10] {
            data[i] = 1.0;
        }
        let original = data.clone();
        run(&mut effect, &mut data, 4, 4);
        assert_eq!(data, original);
    }

    #[test]
    fn small_blob_removed_large_blob_kept() {
        let mut effect = LargestBlobMaskEffect::new();
        // 6x6 mask. Large blob (3x3 in top-left) + small isolated pixel
        // (bottom-right). With 4-connectivity they are NOT adjacent.
        // Layout (rows top-to-bottom, columns left-to-right):
        //   1 1 1 . . .
        //   1 1 1 . . .
        //   1 1 1 . . .
        //   . . . . . .
        //   . . . . . .
        //   . . . . . 1
        let mut data = vec![0.0_f32; 36];
        for r in 0..3 {
            for c in 0..3 {
                data[r * 6 + c] = 1.0;
            }
        }
        data[5 * 6 + 5] = 1.0; // isolated artefact
        run(&mut effect, &mut data, 6, 6);
        // Large blob still 1.0:
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(data[r * 6 + c], 1.0, "pixel ({r},{c}) lost");
            }
        }
        // Isolated artefact zeroed:
        assert_eq!(data[5 * 6 + 5], 0.0);
        // No spillage anywhere else:
        for r in 0..6 {
            for c in 0..6 {
                let in_blob = r < 3 && c < 3;
                if !in_blob {
                    assert_eq!(data[r * 6 + c], 0.0, "spurious value at ({r},{c})");
                }
            }
        }
    }

    #[test]
    fn preserves_original_values_in_largest() {
        let mut effect = LargestBlobMaskEffect::new();
        // 3x3, all foreground but each pixel has a distinct value
        // above level. After filtering — a single connected blob —
        // every value should survive unchanged (not binarised to 1.0).
        let mut data = vec![0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95];
        let original = data.clone();
        run(&mut effect, &mut data, 3, 3);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "got {a}, expected {b}");
        }
    }

    #[test]
    fn diagonal_connections_do_not_join_under_4_connectivity() {
        // Two foreground pixels touching only at corners. With
        // 4-connectivity they form two separate components of size 1.
        // The tie is broken by scan order — first labeled component
        // wins, so the top-left pixel is preserved.
        //   1 . .
        //   . 1 .
        //   . . .
        let mut effect = LargestBlobMaskEffect::new();
        let mut data = vec![0.0_f32; 9];
        data[0] = 1.0;
        data[4] = 1.0;
        run(&mut effect, &mut data, 3, 3);
        assert_eq!(data[0], 1.0, "first-scanned tie winner must survive");
        assert_eq!(data[4], 0.0, "second-scanned tie loser must be zeroed");
    }

    #[test]
    fn below_threshold_pixels_in_largest_are_kept_only_if_at_or_above_level() {
        // Confirm the contract: the "blob" is defined by `mask >= level`.
        // A pixel inside the bounding box of the blob but below level
        // does NOT join the component and is zeroed.
        let mut effect = LargestBlobMaskEffect::new();
        // 3x3 — eight strong pixels around a weak centre. The strong
        // pixels form a ring under 4-connectivity (corners not
        // diagonally connected — but the ring is connected through
        // edges), the centre stays separate.
        //   1 1 1
        //   1 . 1
        //   1 1 1
        let mut data = vec![1.0_f32; 9];
        data[4] = 0.1; // below default level 0.5
        run(&mut effect, &mut data, 3, 3);
        // Ring survives unchanged:
        for &i in &[0, 1, 2, 3, 5, 6, 7, 8] {
            assert_eq!(data[i], 1.0, "ring pixel {i} lost");
        }
        // Centre zeroed:
        assert_eq!(data[4], 0.0);
    }
}
