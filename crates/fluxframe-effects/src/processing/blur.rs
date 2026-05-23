//! CPU box blur for packed RGB buffers.
//!
//! Three-pass separable box blur approximates a Gaussian and is
//! tunable via `passes`.  `radius` is the half-kernel size, so the
//! kernel covers `2*radius + 1` pixels.

use rayon::prelude::*;

/// Apply a separable box blur to a packed RGB buffer.
///
/// `src` and `dst` must be the same length: `width * height * 3`.
/// `scratch` is used as an intermediate (also `width * height * 3`).
/// Result lands in `dst`.
///
/// `passes` ≥ 1: more passes → closer to a Gaussian.
///
/// # Panics
///
/// Panics in debug builds if slice lengths don't match the dimensions.
pub fn box_blur_rgb(
    src: &[u8],
    dst: &mut [u8],
    scratch: &mut [u8],
    width: u32,
    height: u32,
    radius: u32,
    passes: u32,
) {
    debug_assert_eq!(src.len(), (width * height * 3) as usize);
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(scratch.len(), src.len());
    if passes == 0 || radius == 0 {
        dst.copy_from_slice(src);
        return;
    }
    // First pass: src → scratch (horizontal) → dst (vertical).
    // Subsequent passes ping-pong: dst → scratch → dst.
    blur_horizontal(src, scratch, width, height, radius);
    blur_vertical(scratch, dst, width, height, radius);
    for _ in 1..passes {
        blur_horizontal(dst, scratch, width, height, radius);
        blur_vertical(scratch, dst, width, height, radius);
    }
}

/// Sliding-window horizontal box blur with clamp-to-edge boundaries.
///
/// O(width) per row regardless of `radius`: the running sum tracks one
/// kernel-sized window; sliding one pixel right means subtracting the
/// pixel leaving on the left and adding the pixel entering on the
/// right (both clamped to `[0, w-1]`).
fn blur_horizontal(src: &[u8], dst: &mut [u8], width: u32, _height: u32, radius: u32) {
    let w = width as usize;
    let r = radius as usize;
    let r_i32 = i32::try_from(r).expect("blur radius fits in i32");
    let kernel = 2 * r_i32 + 1;
    let w_last = w - 1;
    let row_bytes = w * 3;
    // Rows are independent — each output row only reads its own input
    // row.  Parallelise across rows so multi-core CPUs absorb the cost
    // and a heavy `box_blur_rgb` call does not serialise the entire
    // effect chain on one thread.
    dst.par_chunks_mut(row_bytes)
        .zip(src.par_chunks(row_bytes))
        .for_each(|(dst_row, src_row)| {
            blur_row_horizontal(src_row, dst_row, w, w_last, r, r_i32, kernel);
        });
}

fn blur_row_horizontal(
    src_row: &[u8],
    dst_row: &mut [u8],
    w: usize,
    w_last: usize,
    r: usize,
    r_i32: i32,
    kernel: i32,
) {
    // Build the initial sum for the window centred at x = 0.
    // Kernel slots in [-r, r]; negative slots clamp to src[0],
    // slots beyond w-1 clamp to src[w-1].
    let mut sum = [0i32; 3];
    for k in 0..=r {
        let nx = k.min(w_last);
        let idx = nx * 3;
        sum[0] += i32::from(src_row[idx]);
        sum[1] += i32::from(src_row[idx + 1]);
        sum[2] += i32::from(src_row[idx + 2]);
    }
    // Add `r` extra copies of src[0] to cover negative kernel slots.
    sum[0] += i32::from(src_row[0]) * r_i32;
    sum[1] += i32::from(src_row[1]) * r_i32;
    sum[2] += i32::from(src_row[2]) * r_i32;

    dst_row[0] = (sum[0] / kernel) as u8;
    dst_row[1] = (sum[1] / kernel) as u8;
    dst_row[2] = (sum[2] / kernel) as u8;

    for x in 1..w {
        let leave_x = (x - 1).saturating_sub(r).min(w_last);
        let enter_x = (x + r).min(w_last);
        let leave_idx = leave_x * 3;
        let enter_idx = enter_x * 3;
        sum[0] += i32::from(src_row[enter_idx]) - i32::from(src_row[leave_idx]);
        sum[1] += i32::from(src_row[enter_idx + 1]) - i32::from(src_row[leave_idx + 1]);
        sum[2] += i32::from(src_row[enter_idx + 2]) - i32::from(src_row[leave_idx + 2]);

        let dst_idx = x * 3;
        dst_row[dst_idx] = (sum[0] / kernel) as u8;
        dst_row[dst_idx + 1] = (sum[1] / kernel) as u8;
        dst_row[dst_idx + 2] = (sum[2] / kernel) as u8;
    }
}

/// Sliding-window vertical box blur with clamp-to-edge boundaries.
///
/// Mirror of `blur_horizontal`: column-major running sum slides down
/// one row at a time, subtracting the topmost row and adding the new
/// bottom row (both clamped to `[0, h-1]`).
fn blur_vertical(src: &[u8], dst: &mut [u8], width: u32, height: u32, radius: u32) {
    let w = width as usize;
    let h = height as usize;
    let r = radius as usize;
    let r_i32 = i32::try_from(r).expect("blur radius fits in i32");
    let kernel = 2 * r_i32 + 1;
    let h_last = h - 1;
    for x in 0..w {
        let col_offset = x * 3;
        // Build initial sum for the window centred at y = 0.
        let mut sum = [0i32; 3];
        for k in 0..=r {
            let ny = k.min(h_last);
            let idx = ny * w * 3 + col_offset;
            sum[0] += i32::from(src[idx]);
            sum[1] += i32::from(src[idx + 1]);
            sum[2] += i32::from(src[idx + 2]);
        }
        // Add `r` extra copies of src[y=0, x] for negative kernel slots.
        let top_idx0 = col_offset;
        sum[0] += i32::from(src[top_idx0]) * r_i32;
        sum[1] += i32::from(src[top_idx0 + 1]) * r_i32;
        sum[2] += i32::from(src[top_idx0 + 2]) * r_i32;

        // Emit row 0 pixel.
        let dst_idx = col_offset;
        dst[dst_idx] = (sum[0] / kernel) as u8;
        dst[dst_idx + 1] = (sum[1] / kernel) as u8;
        dst[dst_idx + 2] = (sum[2] / kernel) as u8;

        for y in 1..h {
            let leave_y = (y - 1).saturating_sub(r).min(h_last);
            let enter_y = (y + r).min(h_last);
            let leave_idx = leave_y * w * 3 + col_offset;
            let enter_idx = enter_y * w * 3 + col_offset;
            sum[0] += i32::from(src[enter_idx]) - i32::from(src[leave_idx]);
            sum[1] += i32::from(src[enter_idx + 1]) - i32::from(src[leave_idx + 1]);
            sum[2] += i32::from(src[enter_idx + 2]) - i32::from(src[leave_idx + 2]);

            let dst_idx = y * w * 3 + col_offset;
            dst[dst_idx] = (sum[0] / kernel) as u8;
            dst[dst_idx + 1] = (sum[1] / kernel) as u8;
            dst[dst_idx + 2] = (sum[2] / kernel) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radius_zero_returns_input() {
        let src = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let mut dst = vec![0u8; 12];
        let mut scratch = vec![0u8; 12];
        box_blur_rgb(&src, &mut dst, &mut scratch, 2, 2, 0, 1);
        assert_eq!(dst, src);
    }

    #[test]
    fn zero_passes_returns_input() {
        let src = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let mut dst = vec![0u8; 12];
        let mut scratch = vec![0u8; 12];
        box_blur_rgb(&src, &mut dst, &mut scratch, 2, 2, 3, 0);
        assert_eq!(dst, src);
    }

    #[test]
    fn uniform_input_remains_uniform() {
        let src = vec![128u8; 4 * 4 * 3];
        let mut dst = vec![0u8; 4 * 4 * 3];
        let mut scratch = vec![0u8; 4 * 4 * 3];
        box_blur_rgb(&src, &mut dst, &mut scratch, 4, 4, 1, 2);
        for v in &dst {
            assert_eq!(*v, 128);
        }
    }

    #[test]
    fn step_function_softens() {
        // 8x1 row, left half black, right half white.
        let mut src = vec![0u8; 8 * 3];
        for i in 4..8 {
            src[i * 3] = 255;
            src[i * 3 + 1] = 255;
            src[i * 3 + 2] = 255;
        }
        let mut dst = vec![0u8; src.len()];
        let mut scratch = vec![0u8; src.len()];
        box_blur_rgb(&src, &mut dst, &mut scratch, 8, 1, 1, 1);
        // Near the boundary, values should be intermediate.
        let mid_left = dst[3 * 3]; // pixel 3
        let mid_right = dst[4 * 3]; // pixel 4
        assert!(mid_left > 0 && mid_left < 255, "got {mid_left}");
        assert!(mid_right > 0 && mid_right < 255, "got {mid_right}");
    }
}
