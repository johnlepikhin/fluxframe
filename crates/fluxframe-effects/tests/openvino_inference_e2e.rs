//! End-to-end inference test for [`OpenVinoEngine`] (Stage 8 step 4).
//!
//! Loads the production SelfieSegmentation ONNX through OpenVINO's
//! CPU plugin, runs a synthetic NHWC RGB input through it, and
//! sanity-checks the output mask (shape, value range, basic
//! non-uniformity).  The point is to prove that the OpenVINO seam
//! is end-to-end functional through the **public** crate API
//! (`fluxframe_effects::ml::OpenVinoEngine`).
//!
//! ## What this test deliberately does NOT do
//!
//! Initially this test also loaded the model through
//! `OnnxEngine` (ORT) and compared the mask numerically with
//! `OpenVinoEngine`.  Loading both runtimes in the same process
//! hangs in `futex_do_wait` — both ORT and OpenVINO ship their own
//! TBB-based threadpools and contention for all available cores
//! deadlocks the test before either engine produces a result.  The
//! cross-runtime comparison stays in `cargo bench`-style work
//! (Stage 8.7 or later), not in `cargo test`.
//!
//! `#[ignore]` because the test requires:
//!
//! * the `fluxframe-effects/openvino` Cargo feature (gated via
//!   `required-features` in `Cargo.toml`);
//! * the OpenVINO runtime reachable through `OPENVINO_INSTALL_DIR`
//!   or `LD_LIBRARY_PATH` — on this developer machine wired by the
//!   `johnlepikhin` Guix Home channel after `guix home reconfigure`;
//! * the model file at `<workspace>/models/selfie_segmentation.onnx`.
//!
//! Run locally with:
//!
//! ```sh
//! guix shell -m manifest.scm -- bash -lc '
//!   cargo test -p fluxframe-effects --features openvino \
//!     --test openvino_inference_e2e -- --ignored --nocapture
//! '
//! ```

#![cfg(feature = "openvino")]

use std::path::PathBuf;

use fluxframe_core::traits::{InferenceEngine, InferenceInput};
use fluxframe_effects::ml::{ModelConfig, OpenVinoEngine, load_sidecar_or_placeholder};

/// Workspace-rooted path to the production SelfieSegmentation model.
fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("models")
        .join("selfie_segmentation.onnx")
}

fn model_config() -> ModelConfig {
    load_sidecar_or_placeholder(&model_path()).expect("sidecar TOML present in repo")
}

/// 256×256 NHWC RGB f32 buffer with a deterministic non-uniform
/// pattern.  Solid-grey inputs collapse SelfieSegmentation to a
/// constant mask, defeating the "non-uniform output" check.
fn synthetic_input() -> Vec<f32> {
    let (h, w) = (256_usize, 256);
    let mut buf = vec![0.0_f32; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            // R: horizontal gradient.
            buf[i] = (x as f32) / (w as f32);
            // G: vertical gradient.
            buf[i + 1] = (y as f32) / (h as f32);
            // B: diagonal hash, breaks symmetry.
            buf[i + 2] = (((x ^ y) & 0xFF) as f32) / 255.0;
        }
    }
    buf
}

#[test]
#[ignore = "requires OpenVINO runtime + the SelfieSegmentation model"]
fn openvino_cpu_infers_selfie_segmentation_end_to_end() {
    let path = model_path();
    assert!(
        path.exists(),
        "model file missing at {} — populate before running",
        path.display()
    );
    let cfg = model_config();

    let mut engine = OpenVinoEngine::load(&path, cfg, "CPU").expect("OpenVINO loads");

    let data = synthetic_input();
    let shape = [1_usize, 256, 256, 3];
    let out = engine
        .infer(InferenceInput {
            data: &data,
            shape: &shape,
        })
        .expect("OpenVINO infer");

    eprintln!(
        "openvino output: shape={:?} len={}",
        out.shape,
        out.data.len()
    );

    // 1. Output must be a non-trivial tensor.
    assert!(!out.data.is_empty(), "empty output");
    assert!(!out.shape.is_empty(), "empty output shape");

    // 2. Output rank ≥ 2 (mask + at least one extra dim) — sanity
    // check that the ONNX frontend resolved the segmentation output
    // and didn't downgrade to a scalar.
    assert!(
        out.shape.iter().filter(|&&d| d > 1).count() >= 2,
        "expected at least a 2-D mask in the output shape, got {:?}",
        out.shape
    );

    // 3. Mask values must lie in [0, 1] (SelfieSegmentation outputs
    // sigmoid'ed probabilities).  Tolerate ±0.01 for f32 rounding.
    let min = out.data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = out.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("openvino mask range: [{min}, {max}]");
    assert!(
        (-0.01..=1.01).contains(&min) && (-0.01..=1.01).contains(&max),
        "OpenVINO mask outside [0, 1]: min={min} max={max}"
    );

    // 4. Output must NOT be a flat constant tensor.  The
    // SelfieSegmentation model on an abstract gradient input
    // produces a mostly-zero mask (no «person» detected), so the
    // variance is tiny (~1e-7) but `max - min > 0` proves the
    // network actually responded to spatial structure and we
    // didn't accidentally hand back an all-zeros pre-init buffer.
    let mean: f32 = out.data.iter().sum::<f32>() / (out.data.len() as f32);
    eprintln!("openvino mask mean={mean} spread={}", max - min);
    assert!(
        max - min > 1e-3,
        "output mask is flat: max-min={} < 1e-3 — likely the wrong tensor",
        max - min
    );
}
