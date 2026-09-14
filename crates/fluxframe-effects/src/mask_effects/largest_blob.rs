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
//! Algorithm: run-based connected-component labeling with
//! 4-connectivity over the binary mask `mask >= level`. Each row is
//! split into horizontal runs of foreground pixels; a run is joined
//! (union-find) with every run of the previous row it overlaps in x.
//! Components are therefore sets of runs, and the whole analysis is
//! linear in the number of runs rather than pixels — on a real mask
//! that is a few hundred runs against 65 k pixels. After labeling,
//! every pixel not in the largest component is set to `0.0`; pixels
//! in the largest component keep their original confidence value (so
//! a subsequent `feather` step still produces soft edges).
//!
//! Cost: one pass over the pixels to find runs, one pass over the
//! runs to link them, one pass over the pixels to rewrite.
//!
//! Contract: `LargestBlobConfig::level` must lie in `[0.0, 1.0]`;
//! [`MaskEffect::configure`] returns [`EffectError::InvalidConfig`]
//! otherwise.
//!
//! Tie-breaking: when two components have equal size, the one
//! encountered first in row-major scan order wins. The result is
//! deterministic for a given frame, but the chosen component may
//! change between frames when the second-place component shifts.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::{
    CommitStrategy, DEBOUNCE_FAST_MS, EffectMetadata, ParamDescriptor, ParamKind, Scale,
};
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

/// One horizontal run of foreground pixels: row `y`, columns
/// `x0..=x1`, plus its union-find parent (an index into the run
/// vector).  Runs are appended in row-major scan order, so a run's
/// index also orders the first pixel of every component.
#[derive(Debug, Clone, Copy)]
struct Run {
    y: u32,
    x0: u32,
    x1: u32,
    parent: u32,
}

/// `MaskEffect` that retains only the largest connected component.
///
/// Tie-breaking: equal-sized components are decided by row-major scan
/// order — the first component reached wins. See the module-level
/// docs for the implication on frame-to-frame stability.
#[derive(Default)]
pub struct LargestBlobMaskEffect {
    level: f32,
    /// Runs of the current frame with their union-find links. Reused
    /// across calls — `clear()` keeps capacity, so the steady-state hot
    /// loop does no allocation.
    runs: Vec<Run>,
    /// Per-run component size, indexed by run; only root entries are
    /// meaningful after the count pass. Same reuse policy as `runs`.
    sizes: Vec<u32>,
}

/// Union-find root of run `i` with path halving.
fn find_root(runs: &mut [Run], mut i: usize) -> usize {
    while runs[i].parent as usize != i {
        let grand = runs[runs[i].parent as usize].parent;
        runs[i].parent = grand;
        i = grand as usize;
    }
    i
}

/// Join the components of runs `a` and `b`.  The smaller root index
/// stays the root, so a component's root is always its earliest run in
/// scan order — which is what makes the size tie-break below equal to
/// "first component reached in row-major order".
fn union_runs(runs: &mut [Run], a: usize, b: usize) {
    let ra = find_root(runs, a);
    let rb = find_root(runs, b);
    if ra != rb {
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        runs[hi].parent = lo as u32;
    }
}

impl LargestBlobMaskEffect {
    /// Effect name as registered in the mask registry.
    pub const NAME: &'static str = "largest_blob";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Keep only the largest connected mask component above a threshold.",
        params: &[ParamDescriptor {
            name: "level",
            kind: ParamKind::Float {
                default: DEFAULT_LEVEL,
                min: 0.0,
                max: 1.0,
                step: 0.01,
                scale: Scale::Linear,
            },
            help: "Pixels with value >= level are eligible for blob labeling.",
            commit: CommitStrategy::Live {
                debounce_ms: DEBOUNCE_FAST_MS,
            },
        }],
    };

    /// Construct with the default threshold.
    #[must_use]
    pub fn new() -> Self {
        Self {
            level: default_level(),
            runs: Vec::new(),
            sizes: Vec::new(),
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
        if width == 0 || mask.height == 0 {
            return Ok(());
        }
        debug_assert_eq!(
            mask.data.len(),
            width * mask.height as usize,
            "MaskPlane invariant: data.len() must equal width * height",
        );

        let level = self.level;
        let runs = &mut self.runs;
        runs.clear();

        // Pass 1: split every row into foreground runs and link each
        // run with the runs of the previous row it overlaps in x.
        // Both rows' runs are sorted by x, so the overlap search is a
        // two-pointer walk; `prev_start..prev_end` brackets the
        // previous row inside `runs`.
        let mut prev_start = 0usize;
        let mut prev_end = 0usize;
        for (y, row) in mask.data.chunks_exact(width).enumerate() {
            let row_start = runs.len();
            let mut x = 0usize;
            while x < width {
                if row[x] < level {
                    x += 1;
                    continue;
                }
                let x0 = x;
                while x < width && row[x] >= level {
                    x += 1;
                }
                runs.push(Run {
                    y: y as u32,
                    x0: x0 as u32,
                    x1: (x - 1) as u32,
                    parent: runs.len() as u32,
                });
            }
            let row_end = runs.len();

            let mut p = prev_start;
            for ri in row_start..row_end {
                let (x0, x1) = (runs[ri].x0, runs[ri].x1);
                // Skip previous-row runs that end before this one starts.
                while p < prev_end && runs[p].x1 < x0 {
                    p += 1;
                }
                // Every previous-row run that starts before this one
                // ends overlaps it.  Do not advance `p` past them: the
                // last one may also overlap the next run of this row.
                let mut q = p;
                while q < prev_end && runs[q].x0 <= x1 {
                    union_runs(runs, ri, q);
                    q += 1;
                }
            }
            prev_start = row_start;
            prev_end = row_end;
        }

        // Pass 2: component sizes by root, then the largest — strict
        // `>` with roots visited in increasing index order means an
        // equal-size tie goes to the component whose first run (and
        // hence first pixel) comes first in scan order.
        let sizes = &mut self.sizes;
        sizes.clear();
        sizes.resize(runs.len(), 0);
        for ri in 0..runs.len() {
            let root = find_root(runs, ri);
            sizes[root] += runs[ri].x1 - runs[ri].x0 + 1;
        }
        let mut winner: Option<usize> = None;
        let mut largest_size = 0u32;
        for (ri, &size) in sizes.iter().enumerate() {
            if size > largest_size {
                largest_size = size;
                winner = Some(ri);
            }
        }

        // Pass 3: rewrite. Every pixel outside the winner's runs — below
        // level, or in another component — becomes 0.0; the winner's
        // pixels keep their confidence values.  With no foreground at
        // all (`winner == None`) the whole mask is zeroed.
        let mut ri = 0usize;
        for (y, row) in mask.data.chunks_exact_mut(width).enumerate() {
            let mut cursor = 0usize;
            while ri < runs.len() && runs[ri].y as usize == y {
                let (x0, x1) = (runs[ri].x0 as usize, runs[ri].x1 as usize);
                row[cursor..x0].fill(0.0);
                if winner != Some(find_root(runs, ri)) {
                    row[x0..=x1].fill(0.0);
                }
                cursor = x1 + 1;
                ri += 1;
            }
            row[cursor..].fill(0.0);
        }
        debug_assert_eq!(ri, runs.len(), "every run belongs to a row of the mask");
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

    /// The pre-run-based implementation: BFS labeling with
    /// 4-connectivity, largest component by strict `>` in seed scan
    /// order.  Kept as the behavioural reference for the run-based
    /// kernel.
    fn bfs_reference(mask: &mut [f32], width: usize, height: usize, level: f32) {
        let pixels = width * height;
        let mut labels = vec![0u32; pixels];
        let mut queue = std::collections::VecDeque::new();
        let mut next_label = 1u32;
        let mut largest_label = 0u32;
        let mut largest_size = 0usize;
        for seed in 0..pixels {
            if mask[seed] < level || labels[seed] != 0 {
                continue;
            }
            queue.clear();
            queue.push_back(seed);
            labels[seed] = next_label;
            let mut size = 0usize;
            while let Some(idx) = queue.pop_front() {
                size += 1;
                let (x, y) = (idx % width, idx / width);
                let candidates = [
                    (x > 0).then(|| idx - 1),
                    (x + 1 < width).then(|| idx + 1),
                    (y > 0).then(|| idx - width),
                    (y + 1 < height).then(|| idx + width),
                ];
                for nidx in candidates.into_iter().flatten() {
                    if mask[nidx] >= level && labels[nidx] == 0 {
                        labels[nidx] = next_label;
                        queue.push_back(nidx);
                    }
                }
            }
            if size > largest_size {
                largest_size = size;
                largest_label = next_label;
            }
            next_label += 1;
        }
        for (m, &label) in mask.iter_mut().zip(&labels) {
            if label != largest_label {
                *m = 0.0;
            }
        }
    }

    #[test]
    fn run_based_labeling_matches_bfs_reference_on_noise() {
        // Dense pseudo-random noise produces hundreds of components,
        // many of equal size, and runs that straddle several runs of
        // the row above — every branch of the two-pointer link and the
        // scan-order tie-break gets exercised.  Also a sparse pattern
        // (few runs) and a striped one (runs spanning the full width).
        let cases: [(usize, usize, u32, f32); 4] = [
            (37, 23, 7, 0.5),
            (64, 64, 3, 0.5),
            (50, 9, 11, 0.3),
            (16, 40, 5, 0.7),
        ];
        for (width, height, density, level) in cases {
            let mut seed = 0x9E37_79B9u32;
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed
            };
            let src: Vec<f32> = (0..width * height)
                .map(|_| {
                    let r = next();
                    if r % 10 < density {
                        0.5 + (r % 50) as f32 / 100.0
                    } else {
                        (r % 30) as f32 / 100.0
                    }
                })
                .collect();

            let mut expected = src.clone();
            bfs_reference(&mut expected, width, height, level);

            let mut effect = LargestBlobMaskEffect::new();
            effect
                .configure(toml::from_str(&format!("level = {level}")).unwrap())
                .expect("configure");
            let mut got = src.clone();
            run(&mut effect, &mut got, width as u32, height as u32);
            assert_eq!(
                got, expected,
                "{width}x{height} density {density} level {level}"
            );
        }
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
