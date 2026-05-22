//! Alpha compositing for packed RGB buffers with an `f32` mask.

/// Composite `fg` over `bg` using `mask` as per-pixel alpha (1.0 → fg,
/// 0.0 → bg).  `dst` receives the result.
///
/// All RGB slices must be `mask.len() * 3` bytes long.  The mask is
/// clamped to `[0, 1]` per pixel.
///
/// # Panics
///
/// Panics in debug builds on slice length mismatch.
pub fn alpha_composite_rgb(fg: &[u8], bg: &[u8], dst: &mut [u8], mask: &[f32]) {
    debug_assert_eq!(fg.len(), mask.len() * 3);
    debug_assert_eq!(bg.len(), mask.len() * 3);
    debug_assert_eq!(dst.len(), mask.len() * 3);
    for (i, &alpha) in mask.iter().enumerate() {
        let alpha = alpha.clamp(0.0, 1.0);
        let one_minus = 1.0 - alpha;
        let base = i * 3;
        for c in 0..3 {
            let v = alpha * f32::from(fg[base + c]) + one_minus * f32::from(bg[base + c]);
            dst[base + c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Composite `fg_dst` (foreground, written in-place) over `bg` using
/// `mask`.  Result lands in `fg_dst`.  Useful when the caller wants to
/// avoid an extra full-frame `dst` buffer.
///
/// `fg_dst` and `bg` must both be `mask.len() * 3` bytes long.
///
/// # Panics
///
/// Panics in debug builds on slice length mismatch.
pub fn alpha_composite_rgb_in_place(fg_dst: &mut [u8], bg: &[u8], mask: &[f32]) {
    debug_assert_eq!(fg_dst.len(), mask.len() * 3);
    debug_assert_eq!(bg.len(), mask.len() * 3);
    for (i, &alpha) in mask.iter().enumerate() {
        let alpha = alpha.clamp(0.0, 1.0);
        let one_minus = 1.0 - alpha;
        let base = i * 3;
        for c in 0..3 {
            let fg_v = f32::from(fg_dst[base + c]);
            let bg_v = f32::from(bg[base + c]);
            let v = alpha * fg_v + one_minus * bg_v;
            fg_dst[base + c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_zero_returns_background() {
        let fg = vec![255u8, 0, 0];
        let bg = vec![0u8, 0, 255];
        let mut dst = vec![0u8; 3];
        let mask = vec![0.0_f32];
        alpha_composite_rgb(&fg, &bg, &mut dst, &mask);
        assert_eq!(dst, bg);
    }

    #[test]
    fn mask_one_returns_foreground() {
        let fg = vec![255u8, 0, 0];
        let bg = vec![0u8, 0, 255];
        let mut dst = vec![0u8; 3];
        let mask = vec![1.0_f32];
        alpha_composite_rgb(&fg, &bg, &mut dst, &mask);
        assert_eq!(dst, fg);
    }

    #[test]
    fn mask_half_blends_evenly() {
        let fg = vec![200u8, 100, 50];
        let bg = vec![0u8, 0, 0];
        let mut dst = vec![0u8; 3];
        let mask = vec![0.5_f32];
        alpha_composite_rgb(&fg, &bg, &mut dst, &mask);
        assert_eq!(dst[0], 100);
        assert_eq!(dst[1], 50);
        assert_eq!(dst[2], 25);
    }

    #[test]
    fn out_of_range_mask_is_clamped() {
        let fg = vec![100u8, 100, 100];
        let bg = vec![0u8, 0, 0];
        let mut dst = vec![0u8; 3];
        let mask = vec![2.5_f32]; // > 1 should clamp to 1
        alpha_composite_rgb(&fg, &bg, &mut dst, &mask);
        assert_eq!(dst, fg);
    }

    #[test]
    fn in_place_mask_zero_returns_background() {
        let mut fg_dst = vec![255u8, 0, 0];
        let bg = vec![0u8, 0, 255];
        let mask = vec![0.0_f32];
        alpha_composite_rgb_in_place(&mut fg_dst, &bg, &mask);
        assert_eq!(fg_dst, bg);
    }

    #[test]
    fn in_place_mask_one_keeps_foreground() {
        let original_fg = vec![255u8, 0, 0];
        let mut fg_dst = original_fg.clone();
        let bg = vec![0u8, 0, 255];
        let mask = vec![1.0_f32];
        alpha_composite_rgb_in_place(&mut fg_dst, &bg, &mask);
        assert_eq!(fg_dst, original_fg);
    }

    #[test]
    fn in_place_mask_half_blends_evenly() {
        let mut fg_dst = vec![200u8, 100, 50];
        let bg = vec![0u8, 0, 0];
        let mask = vec![0.5_f32];
        alpha_composite_rgb_in_place(&mut fg_dst, &bg, &mask);
        assert_eq!(fg_dst[0], 100);
        assert_eq!(fg_dst[1], 50);
        assert_eq!(fg_dst[2], 25);
    }

    #[test]
    fn in_place_out_of_range_mask_is_clamped() {
        let mut fg_dst = vec![100u8, 100, 100];
        let bg = vec![0u8, 0, 0];
        let mask = vec![2.5_f32];
        alpha_composite_rgb_in_place(&mut fg_dst, &bg, &mask);
        assert_eq!(fg_dst, vec![100u8, 100, 100]);
    }

    #[test]
    fn in_place_multi_pixel_matches_out_of_place() {
        let fg = vec![200u8, 100, 50, 10, 20, 30];
        let bg = vec![0u8, 0, 0, 100, 100, 100];
        let mask = vec![0.25_f32, 0.75_f32];
        let mut dst_oop = vec![0u8; 6];
        alpha_composite_rgb(&fg, &bg, &mut dst_oop, &mask);
        let mut fg_dst = fg.clone();
        alpha_composite_rgb_in_place(&mut fg_dst, &bg, &mask);
        assert_eq!(fg_dst, dst_oop);
    }
}
