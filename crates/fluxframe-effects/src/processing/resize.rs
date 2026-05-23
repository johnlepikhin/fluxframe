//! Bilinear resize for packed RGB (`&[u8]`) and single-channel mask
//! (`&[f32]`) buffers.  Letterbox mapping for non-matching aspect
//! ratios.

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

/// Bilinear resize for packed RGB (3 bytes per pixel).
///
/// `src` length must equal `src_w * src_h * 3`, `dst` length must
/// equal `dst_w * dst_h * 3`.  No allocation.
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
    if dst_w == 0 || dst_h == 0 {
        return;
    }
    let sw = src_w as f32;
    let sh = src_h as f32;
    let dw = dst_w as f32;
    let dh = dst_h as f32;
    for y in 0..dst_h {
        let sy = ((y as f32 + 0.5) * sh / dh - 0.5).max(0.0);
        let y0 = (sy.floor() as u32).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = sy - y0 as f32;
        for x in 0..dst_w {
            let sx = ((x as f32 + 0.5) * sw / dw - 0.5).max(0.0);
            let x0 = (sx.floor() as u32).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = sx - x0 as f32;
            let i00 = ((y0 * src_w + x0) * 3) as usize;
            let i01 = ((y0 * src_w + x1) * 3) as usize;
            let i10 = ((y1 * src_w + x0) * 3) as usize;
            let i11 = ((y1 * src_w + x1) * 3) as usize;
            let dst_idx = ((y * dst_w + x) * 3) as usize;
            for c in 0..3 {
                let v = (1.0 - wx) * (1.0 - wy) * f32::from(src[i00 + c])
                    + wx * (1.0 - wy) * f32::from(src[i01 + c])
                    + (1.0 - wx) * wy * f32::from(src[i10 + c])
                    + wx * wy * f32::from(src[i11 + c]);
                dst[dst_idx + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
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
    if dst_w == 0 || dst_h == 0 {
        return;
    }
    let sw = src_w as f32;
    let sh = src_h as f32;
    let dw = dst_w as f32;
    let dh = dst_h as f32;
    for y in 0..dst_h {
        let sy = ((y as f32 + 0.5) * sh / dh - 0.5).max(0.0);
        let y0 = (sy.floor() as u32).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = sy - y0 as f32;
        for x in 0..dst_w {
            let sx = ((x as f32 + 0.5) * sw / dw - 0.5).max(0.0);
            let x0 = (sx.floor() as u32).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = sx - x0 as f32;
            let v00 = src[(y0 * src_w + x0) as usize];
            let v01 = src[(y0 * src_w + x1) as usize];
            let v10 = src[(y1 * src_w + x0) as usize];
            let v11 = src[(y1 * src_w + x1) as usize];
            let dst_idx = (y * dst_w + x) as usize;
            dst[dst_idx] = (1.0 - wx) * (1.0 - wy) * v00
                + wx * (1.0 - wy) * v01
                + (1.0 - wx) * wy * v10
                + wx * wy * v11;
        }
    }
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
    fn mask_identity_resize_preserves_values() {
        let src: Vec<f32> = (0..9).map(|i| i as f32 / 8.0).collect();
        let mut dst = vec![0.0_f32; 9];
        resize_mask_bilinear(&src, 3, 3, &mut dst, 3, 3);
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!((s - d).abs() < 1e-3, "src={s} dst={d}");
        }
    }
}
