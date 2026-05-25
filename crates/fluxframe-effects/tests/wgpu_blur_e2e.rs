//! Stage 7 step 3 end-to-end test: compare the `wgpu` GPU blur output
//! against the CPU `box_blur_rgb` reference on the same input.
//!
//! `#[ignore]` because the test requires a working Vulkan adapter on
//! the host — Mesa on Intel/AMD or a vendor driver elsewhere.  Run
//! locally with `cargo test -- --ignored`.  CI without a GPU stays
//! green by skipping this file.
//!
//! Tolerance: ±4 LSB per channel.  The CPU primitive accumulates u8
//! samples into i32 and divides by the integer kernel size with
//! truncation; the GPU shader accumulates into f32 and writes back
//! through an `Rgba8Unorm` storage texture with round-to-nearest.
//! With multiple passes each pass adds the same u8 round-trip, so
//! the residual compounds.  Empirically the divergence stays within
//! ~3 LSB on the test images; ±4 leaves a small cushion.  Visually
//! this is well below the JPEG quantisation noise floor and far
//! below the camera-sensor noise that this blur backs onto in
//! production, so it is acceptable as the formal tolerance.  If a
//! future stage tightens the visual budget, the risk register in
//! `doc/plan/stage-7-wgpu-blur.md` notes the upgrade path: an
//! `Rgba16Float` intermediate texture between the H and V passes
//! removes the per-pass u8 quantisation entirely.

#![cfg(all(feature = "ml", feature = "wgpu"))]

use fluxframe_effects::backend::{BlurBackend, CpuBlurBackend, WgpuBlurBackend};
use fluxframe_effects::processing::box_blur_rgb;

/// Synthetic 64×48 RGB image with a step gradient — sharp transitions
/// stress the boundary clamp and the kernel-sum rounding in both
/// implementations.
fn build_test_image(width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut buf = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            // R: horizontal step at midline.
            buf[i] = if x < w / 2 { 32 } else { 224 };
            // G: vertical ramp.
            buf[i + 1] = ((y * 255) / h.max(1)) as u8;
            // B: diagonal high-frequency noise (deterministic).
            buf[i + 2] = (((x * 17) ^ (y * 31)) as u8).wrapping_add(64);
        }
    }
    buf
}

fn max_channel_delta(a: &[u8], b: &[u8]) -> u16 {
    a.iter()
        .zip(b.iter())
        .map(|(p, q)| u16::from(p.abs_diff(*q)))
        .max()
        .unwrap_or(0)
}

fn run_compare(width: u32, height: u32, radius: u32, passes: u32, tolerance: u16) {
    let src = build_test_image(width, height);

    // CPU reference.
    let mut cpu_dst = vec![0u8; src.len()];
    let mut cpu_scratch = vec![0u8; src.len()];
    box_blur_rgb(
        &src,
        &mut cpu_dst,
        &mut cpu_scratch,
        width,
        height,
        radius,
        passes,
    );

    // GPU candidate.
    let mut gpu = WgpuBlurBackend::probe().expect("Vulkan adapter required for this test");
    gpu.prepare(width, height).expect("gpu prepare");
    let mut gpu_dst = vec![0u8; src.len()];
    gpu.blur(&src, &mut gpu_dst, width, height, radius, passes)
        .expect("gpu blur");

    let delta = max_channel_delta(&cpu_dst, &gpu_dst);
    assert!(
        delta <= tolerance,
        "wgpu blur diverges from CPU reference by {delta} > tolerance {tolerance} \
         (width={width}, height={height}, radius={radius}, passes={passes})"
    );
}

#[test]
#[ignore = "requires Vulkan ICD on the host"]
fn wgpu_matches_cpu_small_radius_one_pass() {
    run_compare(64, 48, 3, 1, 4);
}

#[test]
#[ignore = "requires Vulkan ICD on the host"]
fn wgpu_matches_cpu_medium_radius_two_passes() {
    // Two passes is the production setting in `fluxframe.toml`.
    run_compare(64, 48, 5, 2, 4);
}

#[test]
#[ignore = "requires Vulkan ICD on the host"]
fn wgpu_passthrough_when_radius_zero() {
    let src = build_test_image(16, 16);
    let mut gpu = WgpuBlurBackend::probe().expect("Vulkan adapter required");
    gpu.prepare(16, 16).expect("prepare");
    let mut dst = vec![0u8; src.len()];
    gpu.blur(&src, &mut dst, 16, 16, 0, 1).expect("blur");
    assert_eq!(dst, src, "radius=0 must passthrough");
}

#[test]
#[ignore = "requires Vulkan ICD on the host"]
fn wgpu_passthrough_when_passes_zero() {
    let src = build_test_image(16, 16);
    let mut gpu = WgpuBlurBackend::probe().expect("Vulkan adapter required");
    gpu.prepare(16, 16).expect("prepare");
    let mut dst = vec![0u8; src.len()];
    gpu.blur(&src, &mut dst, 16, 16, 5, 0).expect("blur");
    assert_eq!(dst, src, "passes=0 must passthrough");
}

#[test]
#[ignore = "requires Vulkan ICD on the host"]
fn wgpu_rejects_unprepared_blur() {
    let mut gpu = WgpuBlurBackend::probe().expect("Vulkan adapter required");
    let src = vec![0u8; 4 * 4 * 3];
    let mut dst = vec![0u8; 4 * 4 * 3];
    // No `prepare()` call → blur must fail with a structured error.
    let err = gpu
        .blur(&src, &mut dst, 4, 4, 1, 1)
        .expect_err("blur before prepare must fail");
    assert!(format!("{err}").to_lowercase().contains("prepared"));
}

#[test]
#[ignore = "requires Vulkan ICD on the host"]
fn wgpu_reprepare_to_new_size() {
    let mut gpu = WgpuBlurBackend::probe().expect("Vulkan adapter required");
    gpu.prepare(32, 32).expect("prepare 32x32");
    // Switch to a different size — must succeed and the next blur
    // call must use the new dims.
    gpu.prepare(48, 32).expect("re-prepare 48x32");
    let src = build_test_image(48, 32);
    let mut cpu_dst = vec![0u8; src.len()];
    let mut cpu_scratch = vec![0u8; src.len()];
    box_blur_rgb(&src, &mut cpu_dst, &mut cpu_scratch, 48, 32, 2, 1);
    let mut gpu_dst = vec![0u8; src.len()];
    gpu.blur(&src, &mut gpu_dst, 48, 32, 2, 1)
        .expect("blur 48x32");
    let delta = max_channel_delta(&cpu_dst, &gpu_dst);
    assert!(delta <= 4, "delta {delta} > 4 after reprepare");
}

#[test]
fn cpu_backend_matches_self_sanity() {
    // Cheap, no-GPU sanity that the comparison harness itself is
    // correct: `CpuBlurBackend` wraps `box_blur_rgb`, so they must
    // agree exactly (delta == 0).
    let src = build_test_image(32, 24);
    let mut backend_dst = vec![0u8; src.len()];
    let mut ref_dst = vec![0u8; src.len()];
    let mut ref_scratch = vec![0u8; src.len()];

    let mut backend = CpuBlurBackend::new();
    backend.prepare(32, 24).expect("prepare");
    backend
        .blur(&src, &mut backend_dst, 32, 24, 3, 1)
        .expect("blur");
    box_blur_rgb(&src, &mut ref_dst, &mut ref_scratch, 32, 24, 3, 1);
    assert_eq!(backend_dst, ref_dst);
}
