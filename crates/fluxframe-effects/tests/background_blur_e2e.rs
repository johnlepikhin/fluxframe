//! Stage 4 `background_blur` end-to-end smoke test.
//!
//! Requires a real ONNX person-segmentation model plus a working
//! `libonnxruntime.so`; both are supplied via environment variables to
//! keep the test inert during plain `cargo test --workspace` runs.
//!
//! Environment:
//! * `FLUXFRAME_ONNX_TEST_MODEL` — absolute path to a `.onnx`
//!   segmentation model.  Should sit next to a `<model>.toml` sidecar
//!   describing its input/output layout (see
//!   `fluxframe_effects::ml::ModelConfig`).
//! * `ORT_DYLIB_PATH` — absolute path to `libonnxruntime.so` (the
//!   `ort` crate is built with `load-dynamic`).
//!
//! Run with:
//!
//! ```ignore
//! FLUXFRAME_ONNX_TEST_MODEL=/path/to/seg.onnx \
//! ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
//! cargo test -p fluxframe-effects --tests -- \
//!     --ignored background_blur_e2e
//! ```

#![cfg(feature = "ml")]

use std::path::PathBuf;

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::frame::{FrameBuffer, FrameMeta, PixelFormat, VideoFrame};
use fluxframe_core::traits::VideoEffect;
use fluxframe_effects::BackgroundBlurEffect;

#[test]
#[ignore = "requires FLUXFRAME_ONNX_TEST_MODEL + ORT_DYLIB_PATH; run with --ignored"]
fn background_blur_processes_a_frame() {
    // Locating the model is the gate keeping this test inert when the
    // operator has not opted in.  We bail with a clear message rather
    // than panic, so the test is also runnable in `--include-ignored`
    // CI passes without spurious failures.
    let Some(model_os) = std::env::var_os("FLUXFRAME_ONNX_TEST_MODEL") else {
        eprintln!("FLUXFRAME_ONNX_TEST_MODEL not set; skipping");
        return;
    };
    let model: PathBuf = PathBuf::from(model_os);

    let width: u32 = 64;
    let height: u32 = 64;

    let mut effect = BackgroundBlurEffect::new();

    // Build the configure() payload via TOML so the test exercises the
    // same path that the CLI's `apply()` populates from `--model`.
    // `toml::Value::try_from` on a plain `String` would produce a
    // bare string, not a table, so build the table explicitly.
    let mut table = toml::map::Map::new();
    table.insert(
        "model".into(),
        toml::Value::String(model.display().to_string()),
    );
    table.insert("blur_radius".into(), toml::Value::Integer(5));
    table.insert("blur_passes".into(), toml::Value::Integer(1));
    table.insert("mask_threshold".into(), toml::Value::Float(0.5));
    table.insert("mask_smoothing".into(), toml::Value::Float(0.0));
    table.insert("mask_feather_radius".into(), toml::Value::Integer(1));
    table.insert("mask_dilate".into(), toml::Value::Integer(0));
    table.insert("fallback_threshold".into(), toml::Value::Integer(1));
    let params = toml::Value::Table(table);

    effect.configure(params).expect("configure ok");

    let ctx = ProcessingContext {
        width,
        height,
        format: PixelFormat::Rgb,
        fps: 30,
        counters: None,
    };
    effect
        .prepare(&ctx)
        .expect("prepare ok (model + sidecar must be loadable)");

    // Solid mid-grey RGB frame — content is irrelevant, we only want
    // the per-frame pipeline to exercise resize + inference + compose.
    let buf = vec![128u8; (width as usize) * (height as usize) * 3];
    let mut frame = VideoFrame::new_packed(
        FrameBuffer::Owned(buf),
        width,
        height,
        PixelFormat::Rgb,
        FrameMeta::default(),
    )
    .expect("frame builds");

    let mut frame_ctx = FrameContext::default();
    effect
        .process(&mut frame, &mut frame_ctx)
        .expect("process ok");

    assert!(
        !frame_ctx.fallback_active,
        "expected successful inference on the test model; \
         fallback_active set indicates the engine reported a transient \
         failure on this frame",
    );
}
