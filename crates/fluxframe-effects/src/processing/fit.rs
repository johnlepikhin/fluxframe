//! Aspect-preserving "fit" helpers that resize a packed RGB source
//! into a fixed-size destination buffer.
//!
//! Two semantics are supported:
//!
//! * `fit_cover_rgb` — scale so the source fully covers the destination
//!   rectangle (max-axis scale), then centre-crop the overflow. Aspect
//!   ratio is preserved; the long axis is clipped.
//! * `fit_contain_rgb` — scale so the source fits entirely inside the
//!   destination rectangle (min-axis scale), then letterbox the unused
//!   border with a caller-supplied RGB colour. Aspect ratio is
//!   preserved; one axis carries bars.
//!
//! Both operate on byte buffers (`&[u8]`) with explicit dimensions so
//! the `processing/` module stays free of the `image` crate types in
//! its public surface, even though the implementation uses
//! `image::imageops::resize` for the high-quality bilinear/triangle
//! filter.
//!
//! This module is feature-gated behind `image-fill` because it depends
//! on the optional `image` crate.

use image::imageops::FilterType;
use image::{GenericImageView, RgbImage};

/// Scale `src` (packed RGB, `src_w × src_h`) into a `dst_w × dst_h` RGB
/// buffer using "cover" semantics: max-axis scale followed by
/// centre-crop. Returns a freshly allocated `dst_w * dst_h * 3` byte
/// vector.
///
/// Returns an empty `Vec` if either destination dimension is zero.
///
/// # Panics
///
/// Panics if `src.len() != (src_w * src_h * 3) as usize` or if either
/// `src_w` or `src_h` is zero.
#[must_use]
pub fn fit_cover_rgb(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    assert_eq!(
        src.len(),
        (src_w as usize) * (src_h as usize) * 3,
        "fit_cover_rgb: src buffer length mismatch"
    );
    assert!(
        src_w > 0 && src_h > 0,
        "fit_cover_rgb: src dimensions must be non-zero"
    );
    if dst_w == 0 || dst_h == 0 {
        return Vec::new();
    }
    let src_img = RgbImage::from_raw(src_w, src_h, src.to_vec())
        .expect("RgbImage::from_raw with validated length cannot fail");
    let scale_x = f64::from(dst_w) / f64::from(src_w);
    let scale_y = f64::from(dst_h) / f64::from(src_h);
    let scale = scale_x.max(scale_y);
    // `.ceil() + .max(dst_w/h)` ensures the scaled buffer is at least as
    // large as the destination on both axes so the centre-crop never
    // walks past the buffer end when a sub-pixel rounding pushed the
    // scaled extent just below the destination.
    let scaled_w = ((f64::from(src_w) * scale).ceil() as u32).max(dst_w);
    let scaled_h = ((f64::from(src_h) * scale).ceil() as u32).max(dst_h);
    let scaled = image::imageops::resize(&src_img, scaled_w, scaled_h, FilterType::Triangle);
    let crop_x = (scaled_w - dst_w) / 2;
    let crop_y = (scaled_h - dst_h) / 2;
    scaled
        .view(crop_x, crop_y, dst_w, dst_h)
        .to_image()
        .into_raw()
}

/// Scale `src` (packed RGB, `src_w × src_h`) into a `dst_w × dst_h` RGB
/// buffer using "contain" semantics: min-axis scale followed by
/// letterbox fill on the unused border. Returns a freshly allocated
/// `dst_w * dst_h * 3` byte vector.
///
/// Returns an empty `Vec` if either destination dimension is zero.
///
/// # Panics
///
/// Panics if `src.len() != (src_w * src_h * 3) as usize` or if either
/// `src_w` or `src_h` is zero.
#[must_use]
pub fn fit_contain_rgb(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    letterbox_rgb: [u8; 3],
) -> Vec<u8> {
    assert_eq!(
        src.len(),
        (src_w as usize) * (src_h as usize) * 3,
        "fit_contain_rgb: src buffer length mismatch"
    );
    assert!(
        src_w > 0 && src_h > 0,
        "fit_contain_rgb: src dimensions must be non-zero"
    );
    if dst_w == 0 || dst_h == 0 {
        return Vec::new();
    }
    let src_img = RgbImage::from_raw(src_w, src_h, src.to_vec())
        .expect("RgbImage::from_raw with validated length cannot fail");
    let scale_x = f64::from(dst_w) / f64::from(src_w);
    let scale_y = f64::from(dst_h) / f64::from(src_h);
    let scale = scale_x.min(scale_y);
    let scaled_w = ((f64::from(src_w) * scale).floor() as u32).clamp(1, dst_w);
    let scaled_h = ((f64::from(src_h) * scale).floor() as u32).clamp(1, dst_h);
    let scaled = image::imageops::resize(&src_img, scaled_w, scaled_h, FilterType::Triangle);

    let dst_bytes = (dst_w as usize) * (dst_h as usize) * 3;
    // Pre-fill the whole buffer with the letterbox colour via a
    // `chunks_exact_mut(3)` sweep — cheaper than `extend_from_slice` in
    // a per-pixel loop because the compiler can unroll the 3-byte copy.
    let mut out = vec![0u8; dst_bytes];
    for chunk in out.chunks_exact_mut(3) {
        chunk.copy_from_slice(&letterbox_rgb);
    }
    let offset_x = ((dst_w - scaled_w) / 2) as usize;
    let offset_y = ((dst_h - scaled_h) / 2) as usize;
    let scaled_raw = scaled.as_raw();
    let src_stride = (scaled_w as usize) * 3;
    let dst_stride = (dst_w as usize) * 3;
    for y in 0..(scaled_h as usize) {
        let src_row_start = y * src_stride;
        let dst_row_start = (offset_y + y) * dst_stride + offset_x * 3;
        out[dst_row_start..dst_row_start + src_stride]
            .copy_from_slice(&scaled_raw[src_row_start..src_row_start + src_stride]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
        let n = (width as usize) * (height as usize);
        let mut out = Vec::with_capacity(n * 3);
        for _ in 0..n {
            out.extend_from_slice(&rgb);
        }
        out
    }

    #[test]
    fn cover_monochrome_preserves_colour() {
        let src = solid_rgb(2, 4, [50, 150, 250]);
        let out = fit_cover_rgb(&src, 2, 4, 4, 4);
        assert_eq!(out.len(), 4 * 4 * 3);
        for chunk in out.chunks_exact(3) {
            assert_eq!(chunk, [50, 150, 250]);
        }
    }

    #[test]
    fn contain_letterboxes_with_configured_colour() {
        // 4×1 source into a 4×4 frame: one row of source, rest letterbox.
        let src = solid_rgb(4, 1, [255, 0, 0]);
        let out = fit_contain_rgb(&src, 4, 1, 4, 4, [10, 20, 30]);
        assert_eq!(out.len(), 4 * 4 * 3);
        // Top row letterbox.
        for chunk in out[0..12].chunks_exact(3) {
            assert_eq!(chunk, [10, 20, 30]);
        }
        // Bottom row letterbox.
        for chunk in out[36..48].chunks_exact(3) {
            assert_eq!(chunk, [10, 20, 30]);
        }
        // At least one middle row should be source colour.
        let middle = [&out[12..24], &out[24..36]];
        let any_source = middle
            .iter()
            .any(|row| row.chunks_exact(3).all(|c| c == [255, 0, 0]));
        assert!(any_source, "expected at least one row of source colour");
    }

    #[test]
    fn cover_zero_destination_returns_empty() {
        let src = solid_rgb(2, 2, [1, 2, 3]);
        assert!(fit_cover_rgb(&src, 2, 2, 0, 4).is_empty());
        assert!(fit_cover_rgb(&src, 2, 2, 4, 0).is_empty());
    }

    #[test]
    fn contain_zero_destination_returns_empty() {
        let src = solid_rgb(2, 2, [1, 2, 3]);
        assert!(fit_contain_rgb(&src, 2, 2, 0, 4, [0, 0, 0]).is_empty());
        assert!(fit_contain_rgb(&src, 2, 2, 4, 0, [0, 0, 0]).is_empty());
    }
}
