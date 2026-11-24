//! Stained glass: the plane is split into fixed, irregular cells (a
//! jittered-grid Voronoi pattern), each filled with the mean colour of
//! the pixels it covers, with thin dark "lead" lines between cells.
//!
//! Meant for `[background]` in video calls: the cell pattern never
//! changes and each cell's colour is smoothed over time, so movement
//! behind the speaker turns into a slow shift of tones instead of
//! flicker.  The room's own colours are kept, only averaged.
//!
//! With `ignore_person` (the default) each pixel counts in its cell's
//! average with weight `1 - mask`, so the person's clothes and skin do
//! not tint the glass around and behind them.  A cell the person covers
//! almost entirely keeps its last colour of the room.
//!
//! Cost per frame is `O(width × height)`: one read pass summing colours
//! per cell, one write pass.  The cell map (cell id and line weight per
//! pixel) is built once per geometry and kept until the frame size or a
//! pattern parameter changes.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::{
    CommitStrategy, DEBOUNCE_FAST_MS, DEBOUNCE_HEAVY_MS, DEBOUNCE_STANDARD_MS, EffectMetadata,
    ParamDescriptor, ParamKind,
};
use fluxframe_core::plane::{FramePlane, MaskPlane, PlaneEffect};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

use super::helpers::{
    check_mask_size, float_param, integer_param, reject_int_out_of_range, reject_out_of_range,
};
use crate::processing::parallel::{for_each_row_mut, map_reduce_chunks_indexed};

/// Smallest accepted `cell_size`: below it cells are a few pixels wide
/// and the pattern reads as noise.
const MIN_CELL_SIZE: i64 = 8;
/// Largest accepted `cell_size`: above it the glass is a handful of
/// panes even on 4K frames.
const MAX_CELL_SIZE: i64 = 256;
/// Largest accepted `line_width`, in pixels.
const MAX_LINE_WIDTH: i64 = 8;
/// Upper bound on `smoothing`; `1.0` would freeze the colours.
const MAX_SMOOTHING: f32 = 0.99;

/// Default mean cell size, in pixels.
pub const DEFAULT_CELL_SIZE: u32 = 48;
/// Default cell irregularity.
pub const DEFAULT_IRREGULARITY: f32 = 0.8;
/// Default per-frame inertia of the cell colours.
pub const DEFAULT_SMOOTHING: f32 = 0.9;
/// Default lead line width, in pixels.
pub const DEFAULT_LINE_WIDTH: u32 = 2;
/// Default darkening of the lead lines.
pub const DEFAULT_LINE_STRENGTH: f32 = 0.6;
/// Default for excluding the person from the cell colours.
pub const DEFAULT_IGNORE_PERSON: bool = true;
/// A cell whose visible room covers less than this share of its area
/// keeps its previous colour instead of averaging the few room pixels
/// left around the person.
const MIN_VISIBLE_SHARE: f32 = 0.1;

/// Packed per-pixel entry used while building the map: cell id in the
/// low 24 bits, line weight (0 = inside a cell, 255 = on a line) in the
/// high 8.
const ID_BITS: u32 = 24;
const ID_MASK: u32 = (1 << ID_BITS) - 1;
/// Rows per band when summing colours per cell.  Each band allocates
/// one sum vector, so bands are kept tall.
const SUM_BAND_ROWS: usize = 32;

/// TOML schema:
///
/// ```toml
/// [background.stained_glass]
/// cell_size = 48
/// irregularity = 0.8
/// smoothing = 0.9
/// line_width = 2
/// line_strength = 0.6
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StainedGlassConfig {
    /// Mean cell size in pixels: the step of the grid the cells are
    /// built on.
    #[serde(default = "default_cell_size")]
    pub cell_size: u32,
    /// How far each cell's centre may stray from its grid position:
    /// `0.0` gives a regular grid of squares, `1.0` the most irregular
    /// cells.
    #[serde(default = "default_irregularity")]
    pub irregularity: f32,
    /// Per-frame inertia of each cell's colour.  `0.0` follows the
    /// frame; higher values turn movement into slow colour shifts.
    #[serde(default = "default_smoothing")]
    pub smoothing: f32,
    /// Width of the lines between cells, in pixels.  `0` draws none.
    #[serde(default = "default_line_width")]
    pub line_width: u32,
    /// How much darker the lines are than the cells: `0.0` invisible,
    /// `1.0` black.
    #[serde(default = "default_line_strength")]
    pub line_strength: f32,
    /// Leave the person out of the cell colours (uses the mask).  Turn
    /// off when the effect runs on `[foreground]`, where the plane is the
    /// person.
    #[serde(default = "default_ignore_person")]
    pub ignore_person: bool,
}

fn default_cell_size() -> u32 {
    DEFAULT_CELL_SIZE
}
fn default_irregularity() -> f32 {
    DEFAULT_IRREGULARITY
}
fn default_smoothing() -> f32 {
    DEFAULT_SMOOTHING
}
fn default_line_width() -> u32 {
    DEFAULT_LINE_WIDTH
}
fn default_line_strength() -> f32 {
    DEFAULT_LINE_STRENGTH
}
fn default_ignore_person() -> bool {
    DEFAULT_IGNORE_PERSON
}

impl Default for StainedGlassConfig {
    fn default() -> Self {
        Self {
            cell_size: default_cell_size(),
            irregularity: default_irregularity(),
            smoothing: default_smoothing(),
            line_width: default_line_width(),
            line_strength: default_line_strength(),
            ignore_person: default_ignore_person(),
        }
    }
}

/// Everything the cell map depends on; the map is rebuilt when it
/// changes.
#[derive(Debug, Clone, Copy, PartialEq)]
struct MapKey {
    width: u32,
    height: u32,
    cell_size: u32,
    irregularity: f32,
    line_width: u32,
}

/// A horizontal run of pixels with the same cell and line weight.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Run {
    id: u32,
    /// Line weight, `0..=255`.
    weight: u32,
    len: u32,
}

/// Cell ids and line weights for one geometry, as runs per row: cells
/// are wide, so a row holds a few dozen runs instead of `width` entries
/// and the per-frame passes work per run, not per pixel.
struct CellMap {
    key: MapKey,
    runs: Vec<Run>,
    /// `height + 1` offsets into `runs`: row `y` is
    /// `runs[row_runs[y]..row_runs[y + 1]]`.
    row_runs: Vec<usize>,
    cells: usize,
}

impl CellMap {
    fn row(&self, y: usize) -> &[Run] {
        &self.runs[self.row_runs[y]..self.row_runs[y + 1]]
    }

    /// Collapse packed per-pixel entries into runs.
    fn from_packed(key: MapKey, packed: &[u32], cells: usize) -> Self {
        let width = key.width as usize;
        let mut runs = Vec::new();
        let mut row_runs = Vec::with_capacity(key.height as usize + 1);
        row_runs.push(0);
        for row in packed.chunks_exact(width.max(1)) {
            let mut x = 0;
            while x < row.len() {
                let entry = row[x];
                let len = row[x..].iter().take_while(|&&e| e == entry).count();
                runs.push(Run {
                    id: entry & ID_MASK,
                    weight: entry >> ID_BITS,
                    len: len as u32,
                });
                x += len;
            }
            row_runs.push(runs.len());
        }
        Self {
            key,
            runs,
            row_runs,
            cells,
        }
    }
}

/// Colour sums of one cell for one frame.
#[derive(Debug, Clone, Copy, Default)]
struct CellSums {
    plain: [u64; 3],
    count: u64,
    /// RGB sums weighted by `255 - 255 * mask` per pixel.
    weighted: [u64; 3],
    /// Sum of the weights; `255 × count` when no pixel is the person.
    weight: u64,
}

/// Mask values below this count as pure room, above `1 - ROOM_EPS` as
/// pure person: both round to the end weights `255` and `0`.
const ROOM_EPS: f32 = 1.0 / 510.0;

impl CellSums {
    /// Add a run of pixels with `weight` (`0..=255`) each.
    fn add_uniform<'a>(&mut self, pixels: impl Iterator<Item = &'a [u8]>, weight: u64) {
        // `u32` per run: a run lies within one frame row, so even an
        // 8K row sums to `7680 × 255 ≈ 2e6`.
        let mut rgb = [0_u32; 3];
        let mut n = 0_u64;
        for px in pixels {
            rgb[0] += u32::from(px[0]);
            rgb[1] += u32::from(px[1]);
            rgb[2] += u32::from(px[2]);
            n += 1;
        }
        for ((plain, weighted), sum) in self.plain.iter_mut().zip(&mut self.weighted).zip(rgb) {
            *plain += u64::from(sum);
            *weighted += u64::from(sum) * weight;
        }
        self.count += n;
        self.weight += n * weight;
    }

    /// Add a run of pixels weighted by how much of each is room
    /// (`255 - 255 * mask`). Runs entirely in the room or entirely on
    /// the person — most of them — take the uniform path.
    fn add_masked<'a>(&mut self, pixels: impl Iterator<Item = &'a [u8]>, mask: &[f32]) {
        if mask.iter().all(|&m| m < ROOM_EPS) {
            return self.add_uniform(pixels, 255);
        }
        if mask.iter().all(|&m| m > 1.0 - ROOM_EPS) {
            return self.add_uniform(pixels, 0);
        }
        // `u32` per run: an 8K row weighs at most
        // `7680 × 255 × 255 ≈ 5e8 < u32::MAX`.
        let mut plain = [0_u32; 3];
        let mut weighted = [0_u32; 3];
        let mut weight = 0_u32;
        let mut n = 0_u64;
        for (px, &m) in pixels.zip(mask) {
            let w = 255 - (m.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
            for ((p, wsum), &v) in plain.iter_mut().zip(&mut weighted).zip(px) {
                *p += u32::from(v);
                *wsum += u32::from(v) * w;
            }
            weight += w;
            n += 1;
        }
        for c in 0..3 {
            self.plain[c] += u64::from(plain[c]);
            self.weighted[c] += u64::from(weighted[c]);
        }
        self.count += n;
        self.weight += u64::from(weight);
    }

    fn merge(&mut self, other: &Self) {
        for c in 0..3 {
            self.plain[c] += other.plain[c];
            self.weighted[c] += other.weighted[c];
        }
        self.count += other.count;
        self.weight += other.weight;
    }
}

/// `PlaneEffect` drawing the plane as stained glass.
pub struct StainedGlassEffect {
    config: StainedGlassConfig,
    map: Option<CellMap>,
    /// Smoothed colour per cell; empty until the first frame after a
    /// (re)build or a reset, which seeds it.
    colors: Vec<[f32; 3]>,
    /// Whether the cell's colour comes from the room.  A cell the person
    /// covers from the start is seeded with its plain mean and takes the
    /// room's colour at once when it first becomes visible.
    seen: Vec<bool>,
    /// `colors` rounded for the write pass.
    colors_u8: Vec<[u8; 3]>,
}

impl StainedGlassEffect {
    /// Effect name as registered in the plane registry.
    pub const NAME: &'static str = "stained_glass";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Fixed irregular glass cells filled with the averaged colours of the plane.",
        params: &[
            integer_param(
                "cell_size",
                DEFAULT_CELL_SIZE as i64,
                MIN_CELL_SIZE,
                MAX_CELL_SIZE,
                DEBOUNCE_HEAVY_MS,
                "Mean cell size, in pixels.",
            ),
            float_param(
                "irregularity",
                DEFAULT_IRREGULARITY,
                0.0,
                1.0,
                DEBOUNCE_HEAVY_MS,
                "0 gives a regular grid, 1 the most irregular cells.",
            ),
            float_param(
                "smoothing",
                DEFAULT_SMOOTHING,
                0.0,
                MAX_SMOOTHING,
                DEBOUNCE_STANDARD_MS,
                "Per-frame inertia of the cell colours.",
            ),
            integer_param(
                "line_width",
                DEFAULT_LINE_WIDTH as i64,
                0,
                MAX_LINE_WIDTH,
                DEBOUNCE_HEAVY_MS,
                "Width of the lines between cells, in pixels; 0 draws none.",
            ),
            float_param(
                "line_strength",
                DEFAULT_LINE_STRENGTH,
                0.0,
                1.0,
                DEBOUNCE_STANDARD_MS,
                "How much darker the lines are: 0 invisible, 1 black.",
            ),
            ParamDescriptor {
                name: "ignore_person",
                kind: ParamKind::Bool {
                    default: DEFAULT_IGNORE_PERSON,
                },
                help: "Leave the person out of the cell colours; off on the foreground.",
                commit: CommitStrategy::Live {
                    debounce_ms: DEBOUNCE_FAST_MS,
                },
            },
        ],
    };

    /// Construct with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: StainedGlassConfig::default(),
            map: None,
            colors: Vec::new(),
            seen: Vec::new(),
            colors_u8: Vec::new(),
        }
    }

    fn map_key(&self, width: u32, height: u32) -> MapKey {
        MapKey {
            width,
            height,
            cell_size: self.config.cell_size,
            irregularity: self.config.irregularity,
            line_width: self.config.line_width,
        }
    }

    /// Fold this frame's sums into the smoothed cell colours.
    ///
    /// A cell takes the room's colour at once the first time it sees
    /// the room; a cell hidden behind the person keeps its colour.  A
    /// cell hidden from the start shows the mean colour of the whole
    /// visible room until it is revealed (its own plain mean would be
    /// the person's colour); when no room is visible at all, the plain
    /// mean is all there is.
    fn update_colors(&mut self, sums: &[CellSums]) {
        let cells = sums.len();
        let total = sums.iter().fold(CellSums::default(), |mut acc, s| {
            acc.merge(s);
            acc
        });
        let room_mean =
            (total.weight > 0).then(|| total.weighted.map(|sum| sum as f32 / total.weight as f32));
        if self.colors.len() != cells {
            self.colors = vec![[0.0; 3]; cells];
            self.seen = vec![false; cells];
        }
        let smoothing = self.config.smoothing;
        self.colors_u8.resize(cells, [0; 3]);
        for (((color, seen), out), s) in self
            .colors
            .iter_mut()
            .zip(&mut self.seen)
            .zip(&mut self.colors_u8)
            .zip(sums)
        {
            // A cell whose centre lies past the frame edge may cover no
            // pixels; it is never drawn either.
            if s.count > 0 {
                let visible = s.weight as f32 / (s.count as f32 * 255.0);
                if visible >= MIN_VISIBLE_SHARE {
                    let keep = if *seen { smoothing } else { 0.0 };
                    let w = s.weight as f32;
                    for (channel, &sum) in color.iter_mut().zip(&s.weighted) {
                        let mean = sum as f32 / w;
                        *channel = keep * *channel + (1.0 - keep) * mean;
                    }
                    *seen = true;
                } else if !*seen {
                    let n = s.count as f32;
                    *color = room_mean.unwrap_or(s.plain.map(|sum| sum as f32 / n));
                }
            }
            *out = color.map(|v| v.round().clamp(0.0, 255.0) as u8);
        }
    }

    /// Build the cell map for `key` unless the current one matches.
    /// A new map starts the colours afresh.
    fn ensure_map(&mut self, key: MapKey) {
        if self.map.as_ref().is_some_and(|m| m.key == key) {
            return;
        }
        self.map = Some(build_map(key));
        self.colors.clear();
    }
}

impl Default for StainedGlassEffect {
    fn default() -> Self {
        Self::new()
    }
}

/// Deterministic hash of a grid cell to two offsets in `[-1, 1]`
/// (splitmix64 finaliser).  Fixed, so the pattern is the same on every
/// run.
fn jitter(gx: usize, gy: usize) -> (f32, f32) {
    let mut z = ((gx as u64) << 32 | gy as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let unit = |bits: u64| (bits & 0xFF_FFFF) as f32 / 0xFF_FFFF as f32 * 2.0 - 1.0;
    (unit(z), unit(z >> 32))
}

/// Cell centres of a `cols × rows` grid with step `cell`: each centre
/// strays from its grid cell's middle by up to half a step times
/// `irregularity`, so it stays inside its grid cell.
fn cell_centres(cols: usize, rows: usize, cell: f32, irregularity: f32) -> Vec<(f32, f32)> {
    let reach = 0.5 * cell * irregularity;
    let mut centres = Vec::with_capacity(cols * rows);
    for gy in 0..rows {
        for gx in 0..cols {
            let (jx, jy) = jitter(gx, gy);
            centres.push((
                (gx as f32 + 0.5) * cell + jx * reach,
                (gy as f32 + 0.5) * cell + jy * reach,
            ));
        }
    }
    centres
}

fn build_map(key: MapKey) -> CellMap {
    let (width, height) = (key.width as usize, key.height as usize);
    let cell = key.cell_size as f32;
    let step = key.cell_size as usize;
    let cols = width.div_ceil(step).max(1);
    let rows = height.div_ceil(step).max(1);
    let centres = cell_centres(cols, rows, cell, key.irregularity);
    let half_line = key.line_width as f32 * 0.5;
    let mut packed = vec![0_u32; width * height];
    if width == 0 || height == 0 {
        return CellMap::from_packed(key, &packed, centres.len());
    }
    for_each_row_mut(&mut packed, width, |y, row| {
        let py = y as f32 + 0.5;
        let gy = y / step;
        for (x, out) in row.iter_mut().enumerate() {
            let px = x as f32 + 0.5;
            let gx = x / step;
            // Centres stay inside their grid cells, so the nearest two
            // lie in the surrounding 3 × 3 grid cells.
            let (mut best, mut best_d) = (0_usize, f32::INFINITY);
            let (mut second, mut second_d) = (0_usize, f32::INFINITY);
            for ny in gy.saturating_sub(1)..=(gy + 1).min(rows - 1) {
                for nx in gx.saturating_sub(1)..=(gx + 1).min(cols - 1) {
                    let id = ny * cols + nx;
                    let (cx, cy) = centres[id];
                    let d = (px - cx).powi(2) + (py - cy).powi(2);
                    if d < best_d {
                        (second, second_d) = (best, best_d);
                        (best, best_d) = (id, d);
                    } else if d < second_d {
                        (second, second_d) = (id, d);
                    }
                }
            }
            let weight = if half_line > 0.0 && second_d.is_finite() {
                // Distance from the pixel to the boundary between the two
                // nearest cells (their perpendicular bisector).
                let (ax, ay) = centres[best];
                let (bx, by) = centres[second];
                let span = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt().max(1e-3);
                let to_edge = (second_d - best_d) / (2.0 * span);
                // One pixel of anti-aliasing around the line's edge.
                (half_line + 0.5 - to_edge).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let weight = (weight * 255.0).round() as u32;
            *out = (weight << ID_BITS) | best as u32;
        }
    });
    CellMap::from_packed(key, &packed, centres.len())
}

impl PlaneEffect for StainedGlassEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn reset_state(&mut self) {
        self.colors.clear();
        self.seen.clear();
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: StainedGlassConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        reject_int_out_of_range(
            Self::NAME,
            "cell_size",
            cfg.cell_size.into(),
            MIN_CELL_SIZE,
            MAX_CELL_SIZE,
        )?;
        reject_int_out_of_range(
            Self::NAME,
            "line_width",
            cfg.line_width.into(),
            0,
            MAX_LINE_WIDTH,
        )?;
        reject_out_of_range(Self::NAME, "irregularity", cfg.irregularity, 0.0, 1.0)?;
        reject_out_of_range(Self::NAME, "smoothing", cfg.smoothing, 0.0, MAX_SMOOTHING)?;
        reject_out_of_range(Self::NAME, "line_strength", cfg.line_strength, 0.0, 1.0)?;
        // A changed pattern is rebuilt lazily by the next `process`.
        self.config = cfg;
        Ok(())
    }

    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError> {
        let key = self.map_key(context.width, context.height);
        self.ensure_map(key);
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
        let height = plane.height as usize;
        if width == 0 || height == 0 {
            return Ok(());
        }
        debug_assert_eq!(
            plane.data.len(),
            width * height * 3,
            "FramePlane invariant: data.len() must equal width * height * 3",
        );
        self.ensure_map(self.map_key(plane.width, plane.height));
        let map = self.map.as_ref().expect("built above");
        let mask = self.config.ignore_person.then_some(&*mask.data);
        let sums = cell_sums(map, plane.data, width, mask);
        self.update_colors(&sums);
        let map = self.map.as_ref().expect("built above");
        fill(
            map,
            plane.data,
            width,
            &self.colors_u8,
            self.config.line_strength,
        );
        Ok(())
    }
}

/// Colour sums per cell over the plane. With `mask`, pixels are weighted
/// by how much they are room (`255 - 255 * mask`); without, every pixel
/// has the full weight.
fn cell_sums(map: &CellMap, data: &[u8], width: usize, mask: Option<&[f32]>) -> Vec<CellSums> {
    let cells = map.cells;
    let row_stride = width * 3;
    map_reduce_chunks_indexed(
        data,
        SUM_BAND_ROWS * row_stride,
        || vec![CellSums::default(); cells],
        |offset, band| {
            let mut acc = vec![CellSums::default(); cells];
            let first_row = offset / row_stride;
            for (i, row) in band.chunks_exact(row_stride).enumerate() {
                let y = first_row + i;
                let mut x = 0;
                for run in map.row(y) {
                    let len = run.len as usize;
                    let pixels = row[x * 3..(x + len) * 3].chunks_exact(3);
                    let cell = &mut acc[run.id as usize];
                    match mask {
                        Some(mask) => cell.add_masked(pixels, &mask[y * width + x..][..len]),
                        None => cell.add_uniform(pixels, 255),
                    }
                    x += len;
                }
            }
            acc
        },
        |mut a, b| {
            for (x, y) in a.iter_mut().zip(&b) {
                x.merge(y);
            }
            a
        },
    )
}

/// Fill each pixel with its cell's colour, darkened on the lines (8.8
/// fixed point).
fn fill(map: &CellMap, data: &mut [u8], width: usize, colors: &[[u8; 3]], line_strength: f32) {
    let strength_q8 = (line_strength * 256.0).round() as u32;
    for_each_row_mut(data, width * 3, |y, row| {
        let mut x = 0;
        for run in map.row(y) {
            let len = run.len as usize;
            let color = colors[run.id as usize];
            let gain = 256 - ((run.weight * strength_q8 + 127) / 255);
            let shaded = color.map(|c| ((u32::from(c) * gain + 128) >> 8) as u8);
            for px in row[x * 3..(x + len) * 3].chunks_exact_mut(3) {
                px.copy_from_slice(&shaded);
            }
            x += len;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processing::parallel::MIN_PARALLEL_ELEMS;
    use crate::test_support::{filled_mask, process_plane};

    fn ctx(w: u32, h: u32) -> ProcessingContext {
        ProcessingContext {
            width: w,
            height: h,
            format: fluxframe_core::PixelFormat::Rgb,
            fps: 30,
            counters: None,
        }
    }

    fn parse(text: &str) -> RawEffectParams {
        toml::from_str(text).unwrap()
    }

    fn prepared(cfg: &str, w: u32, h: u32) -> StainedGlassEffect {
        let mut e = StainedGlassEffect::new();
        e.configure(parse(cfg)).expect("configure");
        e.prepare(&ctx(w, h)).expect("prepare");
        e
    }

    fn run(e: &mut StainedGlassEffect, data: &mut [u8], w: u32, h: u32) {
        let mut plane = FramePlane::new(data, w, h);
        process_plane(e, &mut plane, &mut FrameContext::default()).expect("process");
    }

    /// Per-pixel packed entries, expanded back from the runs.
    fn packed(e: &StainedGlassEffect) -> Vec<u32> {
        let map = e.map.as_ref().unwrap();
        map.runs
            .iter()
            .flat_map(|r| std::iter::repeat_n((r.weight << ID_BITS) | r.id, r.len as usize))
            .collect()
    }

    fn ids(e: &StainedGlassEffect) -> Vec<u32> {
        packed(e).iter().map(|p| p & ID_MASK).collect()
    }

    /// Deterministic busy test image.
    fn noise(w: u32, h: u32) -> Vec<u8> {
        let mut s = 0x1234_5678_u32;
        (0..w * h * 3)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 24) as u8
            })
            .collect()
    }

    /// Straightforward per-pixel reference of one seeded frame.
    fn naive(e: &StainedGlassEffect, input: &[u8]) -> Vec<u8> {
        let map = e.map.as_ref().unwrap();
        let packed = packed(e);
        let mut sums = vec![[0_u64; 4]; map.cells];
        for (i, px) in input.chunks_exact(3).enumerate() {
            let id = (packed[i] & ID_MASK) as usize;
            for c in 0..3 {
                sums[id][c] += u64::from(px[c]);
            }
            sums[id][3] += 1;
        }
        let strength_q8 = (e.config.line_strength * 256.0).round() as u32;
        let mut out = Vec::with_capacity(input.len());
        for entry in &packed {
            let s = sums[(entry & ID_MASK) as usize];
            let gain = 256 - (((entry >> ID_BITS) * strength_q8 + 127) / 255);
            for c in 0..3 {
                let mean = (s[c] as f32 / s[3] as f32).round() as u32;
                out.push(((mean * gain + 128) >> 8) as u8);
            }
        }
        out
    }

    #[test]
    fn defaults_when_no_params() {
        let mut e = StainedGlassEffect::new();
        e.configure(parse("")).expect("ok");
        assert_eq!(e.config.cell_size, DEFAULT_CELL_SIZE);
        assert_eq!(e.config.line_width, DEFAULT_LINE_WIDTH);
        assert!((e.config.irregularity - DEFAULT_IRREGULARITY).abs() < 1e-6);
        assert!((e.config.smoothing - DEFAULT_SMOOTHING).abs() < 1e-6);
        assert!((e.config.line_strength - DEFAULT_LINE_STRENGTH).abs() < 1e-6);
    }

    #[test]
    fn rejects_out_of_range_values_and_keeps_the_config() {
        let mut e = StainedGlassEffect::new();
        e.configure(parse("cell_size = 20")).expect("ok");
        for bad in [
            "cell_size = 7",
            "cell_size = 257",
            "line_width = 9",
            "irregularity = 1.1",
            "irregularity = -0.1",
            "smoothing = 1.0",
            "line_strength = 1.5",
            "unknown = 1",
        ] {
            assert!(e.configure(parse(bad)).is_err(), "{bad} accepted");
            assert_eq!(e.config.cell_size, 20, "{bad} changed the config");
        }
        for good in [
            "cell_size = 8",
            "cell_size = 256",
            "line_width = 0",
            "line_width = 8",
            "irregularity = 0.0",
            "smoothing = 0.99",
            "line_strength = 1.0",
        ] {
            e.configure(parse(good))
                .unwrap_or_else(|err| panic!("{good}: {err}"));
        }
    }

    #[test]
    fn the_pattern_is_the_same_on_every_build() {
        let a = prepared("", 120, 80);
        let b = prepared("", 120, 80);
        assert_eq!(packed(&a), packed(&b));
    }

    #[test]
    fn regular_grid_cells_are_squares() {
        // With no irregularity every pixel belongs to its own grid cell.
        let e = prepared("cell_size = 10\nirregularity = 0.0", 40, 30);
        let cols = 4;
        for (i, id) in ids(&e).into_iter().enumerate() {
            let (x, y) = (i % 40, i / 40);
            assert_eq!(id as usize, (y / 10) * cols + x / 10, "pixel {x},{y}");
        }
    }

    #[test]
    fn a_flat_frame_stays_flat_inside_the_cells() {
        let (w, h) = (96, 64);
        let mut e = prepared("", w, h);
        let mut data = [40_u8, 120, 200].repeat((w * h) as usize);
        run(&mut e, &mut data, w, h);
        for (px, entry) in data.chunks_exact(3).zip(&packed(&e)) {
            if entry >> ID_BITS == 0 {
                assert_eq!(px, [40, 120, 200]);
            } else {
                assert!(px[2] < 200, "a line must be darker: {px:?}");
            }
        }
    }

    #[test]
    fn zero_line_width_draws_no_lines() {
        let e = prepared("line_width = 0", 96, 64);
        assert!(packed(&e).iter().all(|p| p >> ID_BITS == 0));
    }

    #[test]
    fn every_drawn_cell_has_pixels_and_ids_are_in_range() {
        let e = prepared("cell_size = 16", 100, 70);
        let map = e.map.as_ref().unwrap();
        let mut seen = vec![false; map.cells];
        for id in ids(&e) {
            assert!((id as usize) < map.cells);
            seen[id as usize] = true;
        }
        // Cells inside the frame are never empty; only cells of the
        // partial last column / row may be.
        let cols = 100_usize.div_ceil(16);
        for gy in 0..70 / 16 {
            for gx in 0..100 / 16 {
                assert!(seen[gy * cols + gx], "cell {gx},{gy} is empty");
            }
        }
    }

    #[test]
    fn parallel_path_matches_naive_reference() {
        let (w, h) = (200_u32, 150_u32);
        assert!((w * h * 3) as usize > MIN_PARALLEL_ELEMS);
        let mut e = prepared("cell_size = 24", w, h);
        let input = noise(w, h);
        let expected = naive(&e, &input);
        let mut data = input.clone();
        run(&mut e, &mut data, w, h);
        assert_eq!(data, expected);
    }

    #[test]
    fn colours_follow_a_change_with_the_configured_inertia() {
        let (w, h) = (32, 32);
        let mut e = prepared("smoothing = 0.5\nline_width = 0", w, h);
        let mut black = vec![0_u8; (w * h * 3) as usize];
        run(&mut e, &mut black, w, h);
        let white = vec![200_u8; (w * h * 3) as usize];
        let mut frame = white.clone();
        run(&mut e, &mut frame, w, h);
        assert_eq!(frame[0], 100, "half way after one frame");
        let mut frame = white.clone();
        run(&mut e, &mut frame, w, h);
        assert_eq!(frame[0], 150);
        // A reset seeds from the current frame at once.
        e.reset_state();
        let mut frame = black.clone();
        run(&mut e, &mut frame, w, h);
        assert_eq!(frame[0], 0);
    }

    #[test]
    fn reconfiguring_the_pattern_after_prepare_rebuilds_the_map() {
        let (w, h) = (64, 48);
        let mut e = prepared("cell_size = 32", w, h);
        let before = e.map.as_ref().unwrap().cells;
        e.configure(parse("cell_size = 16")).expect("ok");
        let mut data = noise(w, h);
        run(&mut e, &mut data, w, h);
        let after = e.map.as_ref().unwrap().cells;
        assert!(after > before, "{before} -> {after}");
        // A smoothing-only change keeps the map and the colours.
        e.configure(parse("cell_size = 16\nsmoothing = 0.5"))
            .expect("ok");
        run(&mut e, &mut data, w, h);
        assert_eq!(e.colors.len(), after);
    }

    #[test]
    fn a_different_plane_size_rebuilds_the_map() {
        let mut e = prepared("", 64, 48);
        let mut data = noise(80, 40);
        run(&mut e, &mut data, 80, 40);
        assert_eq!(packed(&e).len(), 80 * 40);
    }
    // Square 16-px cells without lines: cell 0 is x, y in 0..16.
    const GRID: &str = "cell_size = 16\nirregularity = 0.0\nline_width = 0\nsmoothing = 0.0";
    const GW: u32 = 64;
    const GH: u32 = 32;
    const ROOM: [u8; 3] = [0, 0, 200];
    const SHIRT: [u8; 3] = [200, 0, 0];

    /// Frame of `ROOM` with the person (`SHIRT`, mask 1) at `x < person_w`
    /// in the top 16 rows; returns the output pixel at (0, 0).
    fn run_with_person(e: &mut StainedGlassEffect, person_w: u32) -> [u8; 3] {
        let mut data = Vec::with_capacity((GW * GH * 3) as usize);
        let mut mask = Vec::with_capacity((GW * GH) as usize);
        for y in 0..GH {
            for x in 0..GW {
                let person = x < person_w && y < 16;
                data.extend_from_slice(if person { &SHIRT } else { &ROOM });
                mask.push(if person { 1.0 } else { 0.0 });
            }
        }
        let mut plane = FramePlane::new(&mut data, GW, GH);
        let mask = MaskPlane::new(&mut mask, GW, GH);
        e.process(&mut plane, &mask, &mut FrameContext::default())
            .expect("process");
        [data[0], data[1], data[2]]
    }

    #[test]
    fn the_person_does_not_tint_the_cells() {
        let mut e = prepared(GRID, GW, GH);
        assert_eq!(run_with_person(&mut e, 8), ROOM);
    }

    #[test]
    fn without_ignore_person_the_person_is_averaged_in() {
        let mut e = prepared(&format!("{GRID}\nignore_person = false"), GW, GH);
        assert_eq!(run_with_person(&mut e, 8), [100, 0, 100]);
    }

    #[test]
    fn soft_mask_values_weight_the_average() {
        // Every pixel partly person: the shirt with mask 0.75 (weight
        // 64), the room with mask 0.25 (weight 191).
        let mut e = prepared(GRID, GW, GH);
        let mut data = Vec::new();
        let mut mask = Vec::new();
        for _ in 0..GH {
            for x in 0..GW {
                let shirt = x % 16 < 8;
                data.extend_from_slice(if shirt { &SHIRT } else { &ROOM });
                mask.push(if shirt { 0.75 } else { 0.25 });
            }
        }
        let mut plane = FramePlane::new(&mut data, GW, GH);
        let mask = MaskPlane::new(&mut mask, GW, GH);
        e.process(&mut plane, &mask, &mut FrameContext::default())
            .expect("process");
        // R = 200 × 64 / 255, B = 200 × 191 / 255.
        assert_eq!([data[0], data[1], data[2]], [50, 0, 150]);
    }

    #[test]
    fn a_covered_cell_keeps_the_colour_of_the_room() {
        let mut e = prepared(GRID, GW, GH);
        assert_eq!(run_with_person(&mut e, 0), ROOM);
        // The person now covers cell 0 entirely (and a bit more).
        assert_eq!(run_with_person(&mut e, 20), ROOM);
    }

    #[test]
    fn a_cell_covered_from_the_start_shows_the_room_until_revealed() {
        const WALL: [u8; 3] = [0, 160, 0];
        let mut e = prepared(
            "cell_size = 16\nirregularity = 0.0\nline_width = 0\nsmoothing = 0.9",
            GW,
            GH,
        );
        // Never seen its own room: the mean colour of the visible room,
        // not the shirt.
        assert_eq!(run_with_person(&mut e, 16), ROOM);
        // Revealed, with a wall of another colour behind: that colour at
        // once, despite the smoothing.
        let mut data = Vec::new();
        for y in 0..GH {
            for x in 0..GW {
                data.extend_from_slice(if x < 16 && y < 16 { &WALL } else { &ROOM });
            }
        }
        let mut buf = Vec::new();
        let mask = filled_mask(&mut buf, GW, GH, 0.0);
        let mut plane = FramePlane::new(&mut data, GW, GH);
        e.process(&mut plane, &mask, &mut FrameContext::default())
            .expect("process");
        assert_eq!([data[0], data[1], data[2]], WALL);
    }

    #[test]
    fn with_no_room_visible_the_plain_mean_is_used() {
        let mut e = prepared(GRID, GW, GH);
        let mut data = SHIRT.repeat((GW * GH) as usize);
        let mut buf = Vec::new();
        let mask = filled_mask(&mut buf, GW, GH, 1.0);
        let mut plane = FramePlane::new(&mut data, GW, GH);
        e.process(&mut plane, &mask, &mut FrameContext::default())
            .expect("process");
        assert_eq!([data[0], data[1], data[2]], SHIRT);
    }
}
