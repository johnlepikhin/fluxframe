//! `fluxframe benchmark` — Stage 3 implements the inference-only path.
//!
//! Full end-to-end pipeline benchmarking (capture + chain + sink)
//! lands in Stage 5.

use std::path::Path;
use std::time::Instant;

use fluxframe_core::FluxError;
use fluxframe_core::traits::{InferenceEngine, InferenceInput};
use fluxframe_effects::ml::{ModelConfig, OnnxEngine, TensorDType, TensorLayout};
use tracing::{info, warn};

use crate::cli::BenchmarkArgs;

/// Entry point for `fluxframe benchmark`.
///
/// # Errors
///
/// Returns [`FluxError`] when the model file or its sidecar config
/// cannot be loaded, or when inference fails.
pub fn run(args: BenchmarkArgs) -> Result<(), FluxError> {
    let Some(model_path) = args.common.model else {
        return Err(FluxError::Config {
            reason: "Stage 3 benchmark requires --model <path>".into(),
            hint: Some("pass --model ./models/<name>.onnx with a sibling .toml config".into()),
        });
    };

    if !model_path.exists() {
        return Err(FluxError::Config {
            reason: format!("model file not found: {}", model_path.display()),
            hint: Some("verify the path".into()),
        });
    }

    let config = load_or_default_config(&model_path)?;

    // Snapshot fields that we still need after `config` is moved into
    // `OnnxEngine::load`.  This avoids relying on `ModelConfig: Clone`
    // semantics in the hot loop (the engine owns the config anyway).
    let layout = config.input_layout;
    let dtype = config.input_dtype;
    let (shape, elements) = synthetic_input_shape(&config);

    let mut engine = OnnxEngine::load(&model_path, config)?;
    let info = engine.model_info();
    info!(
        model = %info.name,
        w = info.input_width,
        h = info.input_height,
        ?layout,
        ?dtype,
        elements,
        "synthetic input tensor",
    );

    let data: Vec<f32> = vec![0.0; elements];
    // Stage 3: `--duration` doubles as iteration count.  Stage 5 will
    // switch to a wall-clock budget when the full pipeline benchmark
    // lands.
    let runs = u32::max(args.duration, 1);
    let mut timings: Vec<u128> = Vec::with_capacity(runs as usize);
    for _ in 0..runs {
        let t = Instant::now();
        let _ = engine.infer(InferenceInput {
            data: &data,
            shape: &shape,
        })?;
        timings.push(t.elapsed().as_micros());
    }
    timings.sort_unstable();

    let p50 = percentile(&timings, 50);
    let p95 = percentile(&timings, 95);
    let max = *timings.last().unwrap_or(&0);

    println!("model: {}", info.name);
    println!(
        "input: {}x{} {} {:?}",
        info.input_width,
        info.input_height,
        dtype_label(dtype),
        layout,
    );
    println!("runs:  {runs}");
    println!("p50: {p50} us   p95: {p95} us   max: {max} us");
    Ok(())
}

/// Load a `<model>.toml` sidecar next to `model_path`, falling back to
/// a 1x1 placeholder config when the sidecar is missing.
///
/// TODO(stage-4): consolidate with `commands::check::check_model_file`.
fn load_or_default_config(model_path: &Path) -> Result<ModelConfig, FluxError> {
    let sidecar = model_path.with_extension("toml");
    if sidecar.exists() {
        Ok(ModelConfig::load(&sidecar)?)
    } else {
        warn!(
            sidecar = %sidecar.display(),
            "no model config sidecar found; falling back to 1x1 placeholder",
        );
        Ok(ModelConfig::from_toml_str(
            r#"
name = "<unknown>"
input_width = 1
input_height = 1
"#,
        )?)
    }
}

/// Build the synthetic input shape `(shape, total_elements)` for the
/// given model config.  Channel count is hard-coded to 3 (RGB-typical);
/// Stage 5 will derive it from `config.input_color`.
fn synthetic_input_shape(cfg: &ModelConfig) -> (Vec<usize>, usize) {
    let (n, c, h, w) = (
        1_usize,
        3_usize,
        cfg.input_height as usize,
        cfg.input_width as usize,
    );
    let shape = match cfg.input_layout {
        TensorLayout::Nhwc => vec![n, h, w, c],
        TensorLayout::Nchw => vec![n, c, h, w],
    };
    let elements = shape.iter().product();
    (shape, elements)
}

fn percentile(sorted: &[u128], p: usize) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (sorted.len() * p / 100).min(sorted.len() - 1);
    sorted[idx]
}

fn dtype_label(dtype: TensorDType) -> &'static str {
    match dtype {
        TensorDType::F32 => "f32",
        TensorDType::U8 => "u8",
        TensorDType::I8 => "i8",
    }
}
