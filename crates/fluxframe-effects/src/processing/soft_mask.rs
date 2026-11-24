//! A blurred copy of the silhouette mask, kept at a quarter of the
//! frame size and sampled back at frame resolution.
//!
//! Effects that light or shade the room around the person (a drop
//! shadow, a halo) need the mask blurred by tens of pixels. Blurring at
//! frame size would cost several milliseconds; at a quarter size the
//! blur is ~16× cheaper and the result is smooth anyway, so bilinear
//! sampling hides the lower resolution.

use super::mask::feather;
use super::parallel::for_each_row_mut;

/// Downscale factor per axis.
pub const SOFT_MASK_FACTOR: u32 = 4;
/// Box-blur passes; two passes approximate a Gaussian.
const BLUR_PASSES: usize = 2;

/// Bilinear tap along one axis: the two source indices and the weight
/// of the second one, as a float and in 8.8 fixed point.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Tap {
    i0: usize,
    i1: usize,
    w1: f32,
    w1_q8: u32,
}

/// Blurred, downscaled silhouette mask. Buffers are reused across frames
/// and reallocated only when the frame size changes.
#[derive(Debug, Default)]
pub struct SoftMask {
    frame_w: usize,
    frame_h: usize,
    small_w: usize,
    small_h: usize,
    small: Vec<f32>,
    scratch: Vec<f32>,
    /// One frame row of column sums over a block's rows.
    row_sum: Vec<f32>,
    x_taps: Vec<Tap>,
    y_taps: Vec<Tap>,
}

/// Taps mapping each of `len` frame pixels to the `small_len` grid
/// (pixel-centre aligned, clamp-to-edge).
fn axis_taps(len: usize, small_len: usize) -> Vec<Tap> {
    let factor = SOFT_MASK_FACTOR as f32;
    let last = small_len.saturating_sub(1);
    (0..len)
        .map(|i| {
            let pos = ((i as f32 + 0.5) / factor - 0.5).clamp(0.0, last as f32);
            let i0 = pos.floor() as usize;
            let i1 = (i0 + 1).min(last);
            let w1 = pos - i0 as f32;
            Tap {
                i0,
                i1,
                w1,
                w1_q8: (w1 * 256.0 + 0.5) as u32,
            }
        })
        .collect()
}

impl SoftMask {
    /// Empty; the first [`SoftMask::update`] sizes it.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn resize(&mut self, width: usize, height: usize) {
        if (width, height) == (self.frame_w, self.frame_h) {
            return;
        }
        let factor = SOFT_MASK_FACTOR as usize;
        self.frame_w = width;
        self.frame_h = height;
        self.small_w = width.div_ceil(factor);
        self.small_h = height.div_ceil(factor);
        self.small = vec![0.0; self.small_w * self.small_h];
        self.scratch = vec![0.0; self.small.len()];
        self.row_sum = vec![0.0; width];
        self.x_taps = axis_taps(width, self.small_w);
        self.y_taps = axis_taps(height, self.small_h);
    }

    /// Rebuild from a frame-sized `mask`: average each 4×4 block
    /// (values clamped to `[0, 1]`), then blur by about `radius` frame
    /// pixels. `radius = 0` keeps only the block average.
    ///
    /// # Panics
    ///
    /// Panics if `mask.len() != width * height`.
    pub fn update(&mut self, mask: &[f32], width: u32, height: u32, radius: u32) {
        let (w, h) = (width as usize, height as usize);
        assert_eq!(mask.len(), w * h, "mask size must match the frame");
        self.resize(w, h);
        if w == 0 || h == 0 {
            return;
        }
        let factor = SOFT_MASK_FACTOR as usize;
        let small_w = self.small_w;
        // Sum each block row by row, then divide by the block's pixel
        // count (the last column and row of blocks may be partial) and
        // clamp the mean, which bounds out-of-range mask values without
        // a per-pixel clamp.
        // The rows of a block are first added column-wise (a
        // vectorisable pass), then each group of `factor` columns.
        let row_sum = &mut self.row_sum;
        for (sy, out) in self.small.chunks_exact_mut(small_w).enumerate() {
            let y0 = sy * factor;
            let rows = factor.min(h - y0);
            row_sum.copy_from_slice(&mask[y0 * w..(y0 + 1) * w]);
            for row in mask[(y0 + 1) * w..(y0 + rows) * w].chunks_exact(w) {
                for (acc, &v) in row_sum.iter_mut().zip(row) {
                    *acc += v;
                }
            }
            for (v, block) in out.iter_mut().zip(row_sum.chunks(factor)) {
                let area = (rows * block.len()) as f32;
                *v = (block.iter().sum::<f32>() / area).clamp(0.0, 1.0);
            }
        }
        let small_radius = radius.div_ceil(SOFT_MASK_FACTOR);
        for _ in 0..BLUR_PASSES {
            feather(
                &mut self.small,
                &mut self.scratch,
                small_w as u32,
                self.small_h as u32,
                small_radius,
            );
        }
    }

    /// Sampler for frame row `y - shift`; rows outside the frame clamp
    /// to the nearest edge row.
    ///
    /// # Panics
    ///
    /// Panics if called before the first [`SoftMask::update`] with a
    /// non-empty frame.
    #[must_use]
    pub fn row(&self, y: usize, shift: i32) -> SoftMaskRow<'_> {
        let tap = self.y_taps[shifted(y, shift, self.frame_h)];
        let row = |i: usize| &self.small[i * self.small_w..(i + 1) * self.small_w];
        SoftMaskRow {
            top: row(tap.i0),
            bottom: row(tap.i1),
            wy: tap.w1,
            x_taps: &self.x_taps,
        }
    }
}

/// `i - shift`, clamped to `0..len`.
fn shifted(i: usize, shift: i32, len: usize) -> usize {
    let s = shift.unsigned_abs() as usize;
    let j = if shift >= 0 {
        i.saturating_sub(s)
    } else {
        i.saturating_add(s)
    };
    j.min(len - 1)
}

/// Bilinear sampler over one frame row of a [`SoftMask`].
#[derive(Debug, Clone, Copy)]
pub struct SoftMaskRow<'a> {
    top: &'a [f32],
    bottom: &'a [f32],
    wy: f32,
    x_taps: &'a [Tap],
}

impl SoftMaskRow<'_> {
    /// Blurred mask value at frame column `x - shift`, in `[0, 1]`;
    /// columns outside the frame clamp to the nearest edge column.
    #[must_use]
    pub fn at(&self, x: usize, shift: i32) -> f32 {
        let tap = self.x_taps[shifted(x, shift, self.x_taps.len())];
        let top = self.top[tap.i0] + (self.top[tap.i1] - self.top[tap.i0]) * tap.w1;
        let bottom = self.bottom[tap.i0] + (self.bottom[tap.i1] - self.bottom[tap.i0]) * tap.w1;
        top + (bottom - top) * self.wy
    }
}

/// Frame-column taps for `x - shift` over `0..taps.len()`, clamped to
/// the edge columns.
fn shifted_taps(taps: &[Tap], shift: i32) -> impl Iterator<Item = &Tap> {
    let len = taps.len();
    let s = (shift.unsigned_abs() as usize).min(len);
    let (lead, body, trail) = if shift >= 0 {
        (s, &taps[..len - s], 0)
    } else {
        (0, &taps[s..], s)
    };
    std::iter::repeat_n(&taps[0], lead)
        .chain(body)
        .chain(std::iter::repeat_n(&taps[len - 1], trail))
}

/// How close a pixel is to the person, from a blurred mask value: the
/// blurred mask is about one half right at the silhouette's edge, so it
/// is doubled — `1` at the edge and inside, falling to `0` at the blur
/// radius outside.
#[must_use]
pub fn closeness(soft: f32) -> f32 {
    (2.0 * soft).min(1.0)
}

/// Apply a per-pixel operation to a packed-RGB `plane` (`width` pixels
/// wide, same size as the frame `soft` was built from), driven by the
/// soft mask sampled at `(x - shift_x, y - shift_y)`.
///
/// `eval` maps a mask value in `[0, 1]` to an integer parameter (at
/// most `2^23`); it runs on the quarter-size grid, once per grid column
/// and frame row, and the results are interpolated along the row in
/// fixed point. `apply` then receives each pixel (3 bytes) with its
/// interpolated parameter, so the per-pixel work stays a few integer
/// operations.
///
/// # Panics
///
/// Panics if `plane` does not match the size of the last
/// [`SoftMask::update`].
pub fn map_rows<E, A>(
    plane: &mut [u8],
    width: usize,
    soft: &SoftMask,
    shift: (i32, i32),
    eval: E,
    apply: A,
) where
    E: Fn(f32) -> u32 + Sync,
    A: Fn(&mut [u8], u32) + Sync,
{
    assert_eq!(width, soft.frame_w, "plane width must match the mask");
    assert_eq!(plane.len(), soft.frame_w * soft.frame_h * 3);
    if plane.is_empty() {
        return;
    }
    let (shift_x, shift_y) = shift;
    // Compact form for the per-pixel loop, materialised once per call (a
    // chained iterator in the loop would defeat the optimiser): index of
    // the left value and the 8.8 weight of the right one. The values row
    // gets one padding entry, so `i + 1` is always valid (clamped taps
    // have weight 0).
    let x_taps: Vec<(u32, u32)> = shifted_taps(&soft.x_taps, shift_x)
        .map(|t| (t.i0 as u32, t.w1_q8))
        .collect();
    for_each_row_mut(plane, width * 3, |y, row| {
        let tap = soft.y_taps[shifted(y, shift_y, soft.frame_h)];
        let small_row = |i: usize| &soft.small[i * soft.small_w..(i + 1) * soft.small_w];
        let mut values: Vec<u32> = small_row(tap.i0)
            .iter()
            .zip(small_row(tap.i1))
            .map(|(&top, &bottom)| eval(top + (bottom - top) * tap.w1))
            .collect();
        values.push(*values.last().expect("non-empty row"));
        // Interpolate first, then apply in a plain loop the compiler can
        // keep tight.
        let row_values: Vec<u32> = x_taps
            .iter()
            .map(|&(i, w)| {
                let i = i as usize;
                (values[i] * (256 - w) + values[i + 1] * w + 128) >> 8
            })
            .collect();
        for (px, &v) in row.chunks_exact_mut(3).zip(&row_values) {
            apply(px, v);
        }
    });
}

/// Multiply every pixel by an 8.8 fixed-point gain derived from the soft
/// mask (see [`map_rows`]): `gain` maps a mask value in `[0, 1]` to the
/// factor (`256` = unchanged); results above `255` saturate.
///
/// # Panics
///
/// Panics if `plane` does not match the size of the last
/// [`SoftMask::update`].
pub fn shade<G>(plane: &mut [u8], width: usize, soft: &SoftMask, shift: (i32, i32), gain: G)
where
    G: Fn(f32) -> u32 + Sync,
{
    map_rows(plane, width, soft, shift, gain, |px, g| {
        for c in px {
            *c = ((u32::from(*c) * g + 128) >> 8).min(255) as u8;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(soft: &SoftMask, x: usize, y: usize) -> f32 {
        soft.row(y, 0).at(x, 0)
    }

    #[test]
    fn uniform_mask_stays_uniform() {
        for value in [0.0_f32, 0.4, 1.0] {
            let (w, h) = (37_u32, 23_u32);
            let mask = vec![value; (w * h) as usize];
            let mut soft = SoftMask::new();
            soft.update(&mask, w, h, 12);
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let v = sample(&soft, x, y);
                    assert!((v - value).abs() < 1e-5, "{value}: ({x},{y}) = {v}");
                }
            }
        }
    }

    #[test]
    fn shifts_beyond_the_frame_clamp_to_the_edge() {
        // Person in the leftmost column only.
        let (w, h) = (16_u32, 8_u32);
        let mask: Vec<f32> = (0..w * h)
            .map(|i| if i % w == 0 { 1.0 } else { 0.0 })
            .collect();
        let mut soft = SoftMask::new();
        soft.update(&mask, w, h, 0);
        let edge = sample(&soft, 0, 0);
        assert!((soft.row(0, 40).at(3, 40) - edge).abs() < 1e-6);
        assert!((soft.row(7, -40).at(0, 0) - edge).abs() < 1e-6);
        assert!(soft.row(0, 0).at(0, -40) < edge);
    }

    #[test]
    fn shade_applies_the_gain_and_saturates() {
        let (w, h) = (8_u32, 4_u32);
        let mut soft = SoftMask::new();
        soft.update(&[1.0; 32], w, h, 0);
        let mut plane = vec![200_u8; 96];
        shade(&mut plane, 8, &soft, (0, 0), |s| (s * 128.0) as u32);
        assert!(plane.iter().all(|&v| v == 100));
        shade(&mut plane, 8, &soft, (0, 0), |_| 1024);
        assert!(plane.iter().all(|&v| v == 255));
    }

    #[test]
    fn shifted_taps_match_per_pixel_clamping() {
        let taps = axis_taps(23, 6);
        for shift in [-30, -5, -1, 0, 1, 7, 30] {
            let fast: Vec<Tap> = shifted_taps(&taps, shift).copied().collect();
            let slow: Vec<Tap> = (0..23).map(|x| taps[shifted(x, shift, 23)]).collect();
            assert_eq!(fast, slow, "shift {shift}");
        }
    }

    #[test]
    fn map_rows_passes_the_interpolated_value_to_each_pixel() {
        let (w, h) = (12_u32, 8_u32);
        let mut soft = SoftMask::new();
        soft.update(&[0.5; 96], w, h, 0);
        let mut plane = vec![0_u8; 12 * 8 * 3];
        map_rows(
            &mut plane,
            12,
            &soft,
            (0, 0),
            |s| (s * 200.0) as u32,
            |px, v| px.fill(v as u8),
        );
        assert!(plane.iter().all(|&v| v == 100));
    }

    #[test]
    fn closeness_is_full_at_the_edge_and_zero_far_away() {
        assert!((closeness(0.5) - 1.0).abs() < 1e-6);
        assert!((closeness(0.9) - 1.0).abs() < 1e-6);
        assert!((closeness(0.25) - 0.5).abs() < 1e-6);
        assert!(closeness(0.0).abs() < 1e-6);
    }

    #[test]
    fn values_are_clamped() {
        let mask = vec![3.0_f32; 64];
        let mut soft = SoftMask::new();
        soft.update(&mask, 8, 8, 0);
        assert!((sample(&soft, 3, 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_hard_edge_becomes_a_smooth_monotonic_ramp() {
        // Left half person, right half room.
        let (w, h) = (128_u32, 16_u32);
        let mask: Vec<f32> = (0..w * h)
            .map(|i| if i % w < w / 2 { 1.0 } else { 0.0 })
            .collect();
        let mut soft = SoftMask::new();
        soft.update(&mask, w, h, 16);
        let row = soft.row(8, 0);
        let values: Vec<f32> = (0..w as usize).map(|x| row.at(x, 0)).collect();
        for pair in values.windows(2) {
            assert!(pair[1] <= pair[0] + 1e-6, "not monotonic: {pair:?}");
        }
        assert!(values[0] > 0.99 && values[w as usize - 1] < 0.01);
        // The ramp is wide: well inside the blur radius both sides are
        // partial.
        let mid = (w / 2) as usize;
        assert!(
            values[mid - 6] < 0.95 && values[mid + 6] > 0.05,
            "{values:?}"
        );
    }

    #[test]
    fn frame_size_not_divisible_by_the_factor() {
        // A single person pixel in the last, partial block.
        let (w, h) = (10_u32, 6_u32);
        let mut mask = vec![0.0_f32; 60];
        mask[(h as usize - 1) * w as usize + w as usize - 1] = 1.0;
        let mut soft = SoftMask::new();
        soft.update(&mask, w, h, 0);
        // The last block is 2×2 pixels, so its mean is 1/4.
        assert!((soft.small[soft.small.len() - 1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn resizing_the_frame_reallocates() {
        let mut soft = SoftMask::new();
        soft.update(&[1.0; 16], 4, 4, 0);
        soft.update(&[0.0; 40], 8, 5, 0);
        assert_eq!((soft.small_w, soft.small_h), (2, 2));
        assert!(sample(&soft, 7, 4).abs() < 1e-6);
    }
}
