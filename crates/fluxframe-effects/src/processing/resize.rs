//! Bilinear resize for packed RGB (`&[u8]`) and single-channel mask
//! (`&[f32]`) buffers.  Letterbox mapping for non-matching aspect
//! ratios.

use super::parallel::for_each_row_mut;

/// Description of how a model-input rectangle was placed inside a
/// frame after letterboxing.
///
/// `scale` is the multiplier applied to source coordinates to land in
/// the dst rectangle; `pad_x`/`pad_y` are the leading offsets (in
/// dst-pixel units) where the scaled rectangle starts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    /// Multiplier applied to source coordinates to land in the dst rectangle.
    pub scale: f32,
    /// Leading horizontal offset (in dst pixels) where the scaled rectangle starts.
    pub pad_x: u32,
    /// Leading vertical offset (in dst pixels) where the scaled rectangle starts.
    pub pad_y: u32,
    /// Width of the scaled rectangle (in dst pixels).
    pub inner_width: u32,
    /// Height of the scaled rectangle (in dst pixels).
    pub inner_height: u32,
}

/// Compute the letterbox that fits `src_w × src_h` into `dst_w × dst_h`
/// while preserving aspect ratio.  Pad regions stay zero.
#[must_use]
pub fn fit_letterbox(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Letterbox {
    let scale_x = dst_w as f32 / src_w as f32;
    let scale_y = dst_h as f32 / src_h as f32;
    let scale = scale_x.min(scale_y);
    let inner_w = ((src_w as f32) * scale).round() as u32;
    let inner_h = ((src_h as f32) * scale).round() as u32;
    let pad_x = (dst_w - inner_w) / 2;
    let pad_y = (dst_h - inner_h) / 2;
    Letterbox {
        scale,
        pad_x,
        pad_y,
        inner_width: inner_w,
        inner_height: inner_h,
    }
}

/// One bilinear sample position along an axis: the two source
/// indices to blend and the weight of the second one.
#[derive(Debug, Clone, Copy, PartialEq)]
struct AxisTap {
    /// Lower source index.
    i0: usize,
    /// Upper source index (`i0 + 1`, clamped to the last element).
    i1: usize,
    /// Weight of `i1`; `i0` gets `1 - w`.
    w: f32,
}

/// Pixel-centre bilinear mapping of `dst_len` output positions onto
/// `src_len` source positions, precomputed once per resize call so the
/// per-pixel loop does no division / floor / clamping.
///
/// The mapping is `s = (d + 0.5) * src / dst - 0.5`, clamped at 0 —
/// the same convention the original per-pixel formula used, so the
/// sampled positions are unchanged.
fn axis_taps(src_len: u32, dst_len: u32) -> Vec<AxisTap> {
    let last = (src_len as usize).saturating_sub(1);
    let (sl, dl) = (src_len as f32, dst_len as f32);
    (0..dst_len)
        .map(|d| {
            // Evaluated exactly as the former per-pixel expression so
            // the sampled positions are bit-identical to before.
            let s = ((d as f32 + 0.5) * sl / dl - 0.5).max(0.0);
            let i0 = (s.floor() as usize).min(last);
            let i1 = (i0 + 1).min(last);
            AxisTap {
                i0,
                i1,
                w: s - i0 as f32,
            }
        })
        .collect()
}

/// Blend `a` toward `b` by `w` — one multiply instead of the two the
/// expanded `(1-w)*a + w*b` form needs.
#[inline]
fn lerp(a: f32, b: f32, w: f32) -> f32 {
    a + w * (b - a)
}

/// Bilinear resize for packed RGB (3 bytes per pixel).
///
/// `src` length must equal `src_w * src_h * 3`, `dst` length must
/// equal `dst_w * dst_h * 3`.  Allocates only the two small per-axis
/// tap tables (`dst_w + dst_h` entries).
///
/// # Panics
///
/// Panics in debug builds if slice lengths don't match the dimensions.
pub fn resize_rgb_bilinear(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
) {
    debug_assert_eq!(src.len(), (src_w * src_h * 3) as usize);
    debug_assert_eq!(dst.len(), (dst_w * dst_h * 3) as usize);
    if dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 {
        return;
    }
    let xs = axis_taps(src_w, dst_w);
    let ys = axis_taps(src_h, dst_h);
    let src_row = src_w as usize * 3;
    // Output rows are independent — each reads the shared `src` and
    // writes only its own row — so dispatch them row-parallel.
    for_each_row_mut(dst, dst_w as usize * 3, |y, dst_row| {
        let ty = ys[y];
        let row0 = &src[ty.i0 * src_row..ty.i0 * src_row + src_row];
        let row1 = &src[ty.i1 * src_row..ty.i1 * src_row + src_row];
        for (tx, out) in xs.iter().zip(dst_row.chunks_exact_mut(3)) {
            let p00 = &row0[tx.i0 * 3..tx.i0 * 3 + 3];
            let p01 = &row0[tx.i1 * 3..tx.i1 * 3 + 3];
            let p10 = &row1[tx.i0 * 3..tx.i0 * 3 + 3];
            let p11 = &row1[tx.i1 * 3..tx.i1 * 3 + 3];
            for c in 0..3 {
                let top = lerp(f32::from(p00[c]), f32::from(p01[c]), tx.w);
                let bot = lerp(f32::from(p10[c]), f32::from(p11[c]), tx.w);
                let v = lerp(top, bot, ty.w);
                out[c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    });
}

/// Nearest-neighbour resize for a packed RGB buffer.  Fast and
/// allocation-free — used on the upscale leg of the downscaled-blur
/// pipeline where the input is already low-frequency content and the
/// blocky output is hidden by the feathered alpha mask.  Bilinear here
/// would dominate the blur stage's CPU cost (a 4× upscale runs ~30
/// float ops × output pixels — the upscale alone outweighed the cost
/// of the box-blur it was meant to make cheap).
///
/// # Panics
///
/// Panics in debug builds if slice lengths don't match the dimensions.
pub fn resize_rgb_nearest(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
) {
    debug_assert_eq!(src.len(), (src_w * src_h * 3) as usize);
    debug_assert_eq!(dst.len(), (dst_w * dst_h * 3) as usize);
    if dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 {
        return;
    }
    let src_row_pixels = src_w as usize;
    let dst_row_pixels = dst_w as usize;
    let dst_rows = dst_h as usize;
    // 16.16 fixed-point step keeps the inner loop branch-free and
    // avoids a float→int per pixel.
    let x_step = (u64::from(src_w) << 16) / u64::from(dst_w).max(1);
    let y_step = (u64::from(src_h) << 16) / u64::from(dst_h).max(1);
    let max_col_idx = u64::from(src_w) - 1;
    let max_row_idx = u64::from(src_h) - 1;
    for y in 0..dst_rows {
        let row_idx = (((y as u64).wrapping_mul(y_step) >> 16).min(max_row_idx)) as usize;
        let src_row = row_idx * src_row_pixels * 3;
        let dst_row = y * dst_row_pixels * 3;
        for x in 0..dst_row_pixels {
            let col_idx = (((x as u64).wrapping_mul(x_step) >> 16).min(max_col_idx)) as usize;
            let s = src_row + col_idx * 3;
            let d = dst_row + x * 3;
            dst[d] = src[s];
            dst[d + 1] = src[s + 1];
            dst[d + 2] = src[s + 2];
        }
    }
}

/// Bilinear resize for a single-channel `f32` mask.
///
/// `src` length must equal `src_w * src_h`; same for `dst`.
///
/// # Panics
///
/// Panics in debug builds if slice lengths don't match the dimensions.
pub fn resize_mask_bilinear(
    src: &[f32],
    src_w: u32,
    src_h: u32,
    dst: &mut [f32],
    dst_w: u32,
    dst_h: u32,
) {
    debug_assert_eq!(src.len(), (src_w * src_h) as usize);
    debug_assert_eq!(dst.len(), (dst_w * dst_h) as usize);
    if dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 {
        return;
    }
    let xs = axis_taps(src_w, dst_w);
    let ys = axis_taps(src_h, dst_h);
    let src_row = src_w as usize;
    // Output rows are independent (each reads the shared `src`), so
    // dispatch them row-parallel — same structure as the RGB resize.
    // This is the per-frame model→frame mask upscale in the composite.
    for_each_row_mut(dst, dst_w as usize, |y, dst_row| {
        let ty = ys[y];
        let row0 = &src[ty.i0 * src_row..ty.i0 * src_row + src_row];
        let row1 = &src[ty.i1 * src_row..ty.i1 * src_row + src_row];
        for (tx, out) in xs.iter().zip(dst_row.iter_mut()) {
            let top = lerp(row0[tx.i0], row0[tx.i1], tx.w);
            let bot = lerp(row1[tx.i0], row1[tx.i1], tx.w);
            *out = lerp(top, bot, ty.w);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_preserves_aspect_ratio_wider_src() {
        // 4:3 source into 1:1 target — height becomes inner, width pad.
        let lb = fit_letterbox(400, 300, 100, 100);
        assert!((lb.scale - 0.25).abs() < 1e-6);
        assert_eq!(lb.inner_width, 100);
        assert_eq!(lb.inner_height, 75);
        assert_eq!(lb.pad_x, 0);
        assert_eq!(lb.pad_y, 12);
    }

    #[test]
    fn letterbox_preserves_aspect_ratio_taller_src() {
        // 3:4 source into 1:1 target — width becomes inner, height pad.
        let lb = fit_letterbox(300, 400, 100, 100);
        assert!((lb.scale - 0.25).abs() < 1e-6);
        assert_eq!(lb.inner_width, 75);
        assert_eq!(lb.inner_height, 100);
        assert_eq!(lb.pad_x, 12);
        assert_eq!(lb.pad_y, 0);
    }

    #[test]
    fn letterbox_equal_aspect_no_padding() {
        let lb = fit_letterbox(256, 144, 128, 72);
        assert_eq!(lb.pad_x, 0);
        assert_eq!(lb.pad_y, 0);
        assert_eq!(lb.inner_width, 128);
        assert_eq!(lb.inner_height, 72);
    }

    #[test]
    fn rgb_identity_resize_preserves_pixels() {
        let src: Vec<u8> = (0..(4u8 * 4 * 3)).collect();
        let mut dst = vec![0u8; 4 * 4 * 3];
        resize_rgb_bilinear(&src, 4, 4, &mut dst, 4, 4);
        assert_eq!(dst, src);
    }

    #[test]
    fn rgb_downsample_2x_averages_neighbours() {
        // 2x2 RGB where each row has two pixels with known channel values.
        // Pixel (0,0) = (10,20,30), (0,1) = (50,60,70)
        // Pixel (1,0) = (90,100,110), (1,1) = (130,140,150)
        let src: Vec<u8> = vec![10, 20, 30, 50, 60, 70, 90, 100, 110, 130, 140, 150];
        let mut dst = vec![0u8; 3];
        resize_rgb_bilinear(&src, 2, 2, &mut dst, 1, 1);
        // 1×1 result samples around center → average of all 4
        assert_eq!(dst[0], 70); // (10+50+90+130)/4
        assert_eq!(dst[1], 80);
        assert_eq!(dst[2], 90);
    }

    #[test]
    fn mask_upsample_interpolates() {
        let src = vec![0.0_f32, 1.0, 1.0, 0.0];
        let mut dst = vec![0.0_f32; 16];
        resize_mask_bilinear(&src, 2, 2, &mut dst, 4, 4);
        // Centre values should interpolate; corners stay close to original.
        assert!((dst[0] - 0.0).abs() < 1e-3); // top-left
        assert!((dst[15] - 0.0).abs() < 1e-3); // bottom-right
        assert!((dst[3] - 1.0).abs() < 1e-3); // top-right
        assert!((dst[12] - 1.0).abs() < 1e-3); // bottom-left
    }

    #[test]
    fn axis_taps_match_pixel_centre_formula() {
        // Reference: the per-pixel formula the kernels used before the
        // tables were introduced.
        let (src_len, dst_len) = (256u32, 800u32);
        let taps = axis_taps(src_len, dst_len);
        assert_eq!(taps.len(), dst_len as usize);
        for (d, tap) in taps.iter().enumerate() {
            let s = ((d as f32 + 0.5) * src_len as f32 / dst_len as f32 - 0.5).max(0.0);
            let i0 = (s.floor() as usize).min(src_len as usize - 1);
            assert_eq!(tap.i0, i0);
            assert_eq!(tap.i1, (i0 + 1).min(src_len as usize - 1));
            assert!((tap.w - (s - i0 as f32)).abs() < 1e-6);
            assert!((0.0..1.0).contains(&tap.w));
        }
        // Last tap clamps to the final source element.
        let last = taps[dst_len as usize - 1];
        assert_eq!(last.i1, src_len as usize - 1);
    }

    #[test]
    fn axis_taps_single_source_element_is_degenerate() {
        for tap in axis_taps(1, 5) {
            assert_eq!((tap.i0, tap.i1), (0, 0));
        }
    }

    #[test]
    fn mask_upsample_matches_reference_bilinear() {
        // Pseudo-random 16×9 mask upscaled 3× — the two-lerp form must
        // agree with the expanded four-term weights to float noise.
        let (sw, sh, dw, dh) = (16u32, 9u32, 48u32, 27u32);
        let src: Vec<f32> = (0..sw * sh)
            .map(|i| (i.wrapping_mul(2_654_435_761) % 1000) as f32 / 1000.0)
            .collect();
        let mut dst = vec![0.0_f32; (dw * dh) as usize];
        resize_mask_bilinear(&src, sw, sh, &mut dst, dw, dh);
        for y in 0..dh {
            let sy = ((y as f32 + 0.5) * sh as f32 / dh as f32 - 0.5).max(0.0);
            let y0 = (sy.floor() as u32).min(sh - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let wy = sy - y0 as f32;
            for x in 0..dw {
                let sx = ((x as f32 + 0.5) * sw as f32 / dw as f32 - 0.5).max(0.0);
                let x0 = (sx.floor() as u32).min(sw - 1);
                let x1 = (x0 + 1).min(sw - 1);
                let wx = sx - x0 as f32;
                let at = |yy: u32, xx: u32| src[(yy * sw + xx) as usize];
                let expected = (1.0 - wx) * (1.0 - wy) * at(y0, x0)
                    + wx * (1.0 - wy) * at(y0, x1)
                    + (1.0 - wx) * wy * at(y1, x0)
                    + wx * wy * at(y1, x1);
                let got = dst[(y * dw + x) as usize];
                assert!(
                    (got - expected).abs() < 1e-5,
                    "({x},{y}) {got} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn mask_identity_resize_preserves_values() {
        let src: Vec<f32> = (0..9).map(|i| i as f32 / 8.0).collect();
        let mut dst = vec![0.0_f32; 9];
        resize_mask_bilinear(&src, 3, 3, &mut dst, 3, 3);
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!((s - d).abs() < 1e-3, "src={s} dst={d}");
        }
    }
}
