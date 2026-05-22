//! End-to-end smoke test for [`fluxframe_effects::ml::OnnxEngine`].
//!
//! Requires a real ONNX model on disk plus a working
//! `libonnxruntime.so`.  Both must be supplied through environment
//! variables to keep the test hermetic enough for CI / `cargo test
//! --workspace` runs without the runtime installed:
//!
//! * `FLUXFRAME_ONNX_TEST_MODEL` — absolute path to a `.onnx` file.
//!   The test only opens a session and (optionally) runs a single
//!   forward pass, so any tiny model with at least one f32 input is
//!   enough.
//! * `ORT_DYLIB_PATH` — absolute path to `libonnxruntime.so`.  The
//!   `ort` crate is built with `load-dynamic`, so this must point at
//!   a shared library at runtime.
//! * `FLUXFRAME_ONNX_TEST_SHAPE` — optional, comma-separated input
//!   shape (e.g. `"1,3,224,224"`).  When set the test also calls
//!   `infer()` with a zero tensor of that shape.
//!
//! Test is `#[ignore]`d by default — run with `cargo test -p
//! fluxframe-effects --test onnx_inference_e2e -- --ignored`.

use std::path::PathBuf;

use fluxframe_core::traits::{InferenceEngine, InferenceInput};
use fluxframe_effects::ml::{ModelConfig, OnnxEngine};

fn model_path_from_env() -> Option<PathBuf> {
    std::env::var_os("FLUXFRAME_ONNX_TEST_MODEL").map(PathBuf::from)
}

fn shape_from_env() -> Option<Vec<usize>> {
    let raw = std::env::var("FLUXFRAME_ONNX_TEST_SHAPE").ok()?;
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().ok())
        .collect::<Option<Vec<_>>>()
}

#[test]
#[ignore = "requires ORT_DYLIB_PATH and FLUXFRAME_ONNX_TEST_MODEL"]
fn loads_real_model_from_env() {
    let Some(model_path) = model_path_from_env() else {
        panic!("set FLUXFRAME_ONNX_TEST_MODEL to a real .onnx file path");
    };

    let cfg = ModelConfig::from_toml_str(
        r#"
name = "e2e-smoke"
input_width = 256
input_height = 144
"#,
    )
    .expect("model config parses");

    let mut engine = OnnxEngine::load(&model_path, cfg).expect("engine loads with valid ORT path");

    let info = engine.model_info();
    assert_eq!(&*info.name, "e2e-smoke");
    assert!(
        engine.input_count() >= 1,
        "model exposes at least one input"
    );

    if let Some(shape) = shape_from_env() {
        let elems: usize = shape.iter().product();
        let data = vec![0.0_f32; elems];
        let _ = engine
            .infer(InferenceInput {
                data: &data,
                shape: &shape,
            })
            .expect("zero-tensor inference round-trips");
    }
}
