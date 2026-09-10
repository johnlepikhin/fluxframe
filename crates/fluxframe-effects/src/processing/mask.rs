//! Mask post-processing: threshold, temporal smoothing, dilation,
//! edge feathering (separable box blur on the mask).
//!
//! All operations are in-place or use caller-supplied scratch buffers.

/// Threshold the mask: values >= `threshold` become 1.0, others 0.0.
pub fn threshold(mask: &mut [f32], threshold: f32) {
    for v in mask.iter_mut() {
        *v = if *v >= threshold { 1.0 } else { 0.0 };
    }
}

/// Temporal smoothing — exponential moving average between the
/// previous frame's mask (`prev`) and the current one (`curr`).
///
/// `prev` is updated in place: `prev = alpha * prev + (1 - alpha) * curr`.
/// `alpha` in `[0, 1]`; closer to 1 → more inertia (smoother but
/// laggier), 0 → no smoothing.
///
/// # Panics
///
/// Panics in debug builds if `prev.len() != curr.len()`.
pub fn smooth_temporal(prev: &mut [f32], curr: &[f32], alpha: f32) {
    debug_assert_eq!(prev.len(), curr.len());
    let alpha = alpha.clamp(0.0, 1.0);
    let one_minus = 1.0 - alpha;
    for (p, c) in prev.iter_mut().zip(curr.iter()) {
        *p = alpha * *p + one_minus * *c;
    }
}

/// Morphological dilation with a 3x3 max kernel.
///
/// `iterations` controls how many passes: each pass expands foreground
/// regions by one pixel.  `scratch` must be the same length as `mask`
/// and is used as an intermediate buffer.
///
/// # Panics
///
/// Panics in debug builds if `mask.len() != width * height` or
/// `scratch.len() != mask.len()`.
pub fn dilate(mask: &mut [f32], scratch: &mut [f32], width: u32, height: u32, iterations: u32) {
    debug_assert_eq!(mask.len(), (width as usize) * (height as usize));
    debug_assert_eq!(scratch.len(), mask.len());
    if iterations == 0 || width == 0 || height == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    for _ in 0..iterations {
        for y in 0..h {
            let y_start = y.saturating_sub(1);
            let y_end = (y + 1).min(h - 1);
            for x in 0..w {
                let x_start = x.saturating_sub(1);
                let x_end = (x + 1).min(w - 1);
                let mut max_val = 0.0_f32;
                for ny in y_start..=y_end {
                    for nx in x_start..=x_end {
                        let v = mask[ny * w + nx];
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
                scratch[y * w + x] = max_val;
            }
        }
        mask.copy_from_slice(scratch);
    }
}

/// Edge feathering — soft-edge blur on the mask via a separable box
/// blur of `radius` pixels with clamp-to-edge borders.  Single pass;
/// for stronger blur, increase `radius`.
///
/// Both passes are sliding-window running sums, so the cost is
/// `O(width × height)` independent of `radius`.  The vertical pass is
/// evaluated row-major against a per-column sum so every inner loop
/// walks contiguous memory.
///
/// `scratch` must be the same length as `mask`.
///
/// # Panics
///
/// Panics in debug builds on dimension/length mismatch.
pub fn feather(mask: &mut [f32], scratch: &mut [f32], width: u32, height: u32, radius: u32) {
    debug_assert_eq!(mask.len(), (width as usize) * (height as usize));
    debug_assert_eq!(scratch.len(), mask.len());
    if radius == 0 || width == 0 || height == 0 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let r = radius as usize;
    let inv_kernel = 1.0 / (2 * r + 1) as f32;
    // Index of the element `r` before `i`, clamped to the edge; and of
    // the element `r + 1` after `i`, clamped to `len - 1`.
    let leave_at = |i: usize| i.saturating_sub(r);
    let enter_at = |i: usize, len: usize| (i + r + 1).min(len - 1);

    // Horizontal pass: mask → scratch, one running sum per row.
    for (src_row, dst_row) in mask.chunks_exact(w).zip(scratch.chunks_exact_mut(w)) {
        // Window centred on x = 0: `r + 1` copies of the edge element
        // (clamp-to-edge) plus the next `r` elements.
        let mut sum =
            src_row[0] * (r + 1) as f32 + (1..=r).map(|dx| src_row[dx.min(w - 1)]).sum::<f32>();
        for (x, out) in dst_row.iter_mut().enumerate() {
            *out = sum * inv_kernel;
            sum += src_row[enter_at(x, w)] - src_row[leave_at(x)];
        }
    }

    // Vertical pass: scratch → mask, one running sum per column,
    // advanced a whole row at a time.
    let row = |y: usize| &scratch[y * w..y * w + w];
    let mut col_sum: Vec<f32> = row(0).iter().map(|v| v * (r + 1) as f32).collect();
    for dy in 1..=r {
        for (acc, v) in col_sum.iter_mut().zip(row(dy.min(h - 1))) {
            *acc += v;
        }
    }
    for (y, dst_row) in mask.chunks_exact_mut(w).enumerate() {
        for (out, &acc) in dst_row.iter_mut().zip(&col_sum) {
            *out = acc * inv_kernel;
        }
        let enter = row(enter_at(y, h));
        let leave = row(leave_at(y));
        for ((acc, &e), &l) in col_sum.iter_mut().zip(enter).zip(leave) {
            *acc += e - l;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_binarises_mask() {
        let mut mask = vec![0.0_f32, 0.3, 0.5, 0.7, 1.0];
        threshold(&mut mask, 0.5);
        let expected = [0.0_f32, 0.0, 1.0, 1.0, 1.0];
        for (a, b) in mask.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {a}, expected {b}");
        }
    }

    #[test]
    fn smooth_temporal_blends() {
        let mut prev = vec![0.0_f32, 1.0, 0.0, 1.0];
        let curr = vec![1.0_f32, 0.0, 1.0, 0.0];
        smooth_temporal(&mut prev, &curr, 0.5);
        for v in &prev {
            assert!((v - 0.5).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn smooth_temporal_alpha_one_keeps_prev() {
        let mut prev = vec![0.0_f32, 1.0];
        let curr = vec![1.0_f32, 0.0];
        smooth_temporal(&mut prev, &curr, 1.0);
        let expected = [0.0_f32, 1.0];
        for (a, b) in prev.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn smooth_temporal_alpha_zero_takes_curr() {
        let mut prev = vec![0.0_f32, 1.0];
        let curr = vec![0.7_f32, 0.3];
        smooth_temporal(&mut prev, &curr, 0.0);
        for (a, b) in prev.iter().zip(curr.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn dilate_expands_single_pixel() {
        // 3x3 grid with a single foreground in the centre.
        let mut mask = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let mut scratch = vec![0.0_f32; 9];
        dilate(&mut mask, &mut scratch, 3, 3, 1);
        // All cells reachable by 3x3 kernel from centre → all should be 1.0.
        for v in &mask {
            assert!((v - 1.0).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn dilate_zero_iterations_is_identity() {
        let mut mask = vec![0.5_f32; 9];
        let mut scratch = vec![0.0_f32; 9];
        dilate(&mut mask, &mut scratch, 3, 3, 0);
        for v in &mask {
            assert!((v - 0.5).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn feather_smooths_edges() {
        // Sharp step in horizontal direction.  After feather radius 1,
        // edge pixels become mid-values.
        let mut mask = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let mut scratch = vec![0.0_f32; 8];
        feather(&mut mask, &mut scratch, 4, 2, 1);
        // Interior values should be 0 or 1 (kernel doesn't reach the
        // edge fully); near the step we expect interpolated values.
        // Pixel (1,0): kernel spans cols 0..=2, rows 0..=1 →
        //   horizontal pass (mask):  (0+0+1)/3 = 0.333... at (1,0).
        //   vertical pass on scratch column at x=1: (0.333 + 0.333)/3 = 0.222
        //   etc. We just sanity-check that top-left stays 0 and top-right stays high.
        // Layout is row-major: mask[0] = (x=0, y=0), mask[3] = (x=3, y=0).
        assert!(mask[0] < 0.5); // top-left corner
        assert!(mask[3] > 0.5); // top-right corner
    }

    #[test]
    fn feather_matches_naive_box_blur_reference() {
        // Direct O(radius) evaluation with clamp-to-edge — the sliding
        // window must agree to float noise, including a radius larger
        // than the image so every window is fully clamped.
        for &(w, h, r) in &[(9usize, 6usize, 1usize), (9, 6, 3), (5, 4, 7)] {
            let src: Vec<f32> = (0..w * h)
                .map(|i| ((i * 37 + 11) % 101) as f32 / 100.0)
                .collect();
            // Window `[i - r, i + r]` as offsets `i + d - r`, clamped to
            // `[0, len - 1]` — evaluated with unsigned arithmetic only.
            let window = |i: usize, len: usize| {
                (0..=2 * r).map(move |d| (i + d).saturating_sub(r).min(len - 1))
            };
            let k = (2 * r + 1) as f32;
            let mut hpass = vec![0.0_f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let s: f32 = window(x, w).map(|xx| src[y * w + xx]).sum();
                    hpass[y * w + x] = s / k;
                }
            }
            let mut expected = vec![0.0_f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let s: f32 = window(y, h).map(|yy| hpass[yy * w + x]).sum();
                    expected[y * w + x] = s / k;
                }
            }

            let mut mask = src.clone();
            let mut scratch = vec![0.0_f32; w * h];
            feather(&mut mask, &mut scratch, w as u32, h as u32, r as u32);
            for (i, (got, exp)) in mask.iter().zip(&expected).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-5,
                    "{w}x{h} r={r} idx {i}: {got} vs {exp}"
                );
            }
        }
    }

    #[test]
    fn feather_radius_zero_is_identity() {
        let original = vec![0.0_f32, 0.5, 1.0, 0.5];
        let mut mask = original.clone();
        let mut scratch = vec![0.0_f32; 4];
        feather(&mut mask, &mut scratch, 2, 2, 0);
        for (a, b) in mask.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
