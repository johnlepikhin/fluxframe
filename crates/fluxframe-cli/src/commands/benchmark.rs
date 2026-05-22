//! `fluxframe benchmark` — Stage 3 implements the inference-only path.
//!
//! Full end-to-end pipeline benchmarking (capture + chain + sink)
//! lands in Stage 5.

use std::time::Instant;

use fluxframe_core::FluxError;
use fluxframe_core::traits::{InferenceEngine, InferenceInput};
use fluxframe_effects::ml::{Layout, ModelConfig, OnnxEngine, TensorDType};
use tracing::info;

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

    let config = fluxframe_effects::ml::load_sidecar_or_placeholder(&model_path)?;

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

    let duration = std::time::Duration::from_secs(args.duration.into());
    let start = Instant::now();
    let mut timings: Vec<u128> = Vec::with_capacity(1024);
    while start.elapsed() < duration {
        let t = Instant::now();
        engine.infer(InferenceInput {
            data: &data,
            shape: &shape,
        })?;
        timings.push(t.elapsed().as_micros());
    }
    timings.sort_unstable();

    let total = timings.len();
    if total == 0 {
        return Err(FluxError::Config {
            reason: "duration too short — no inferences completed".into(),
            hint: Some("increase --duration to at least 1 second".into()),
        });
    }
    let p50 = percentile(&timings, 50);
    let p95 = percentile(&timings, 95);
    let max = *timings.last().expect("non-empty by check above");

    println!("model: {}", info.name);
    println!(
        "input: {}x{} {} {:?}",
        info.input_width,
        info.input_height,
        dtype_label(dtype),
        layout,
    );
    println!("runs:  {} (in {:?})", total, start.elapsed());
    println!("p50: {p50} us   p95: {p95} us   max: {max} us");
    Ok(())
}

/// Build the synthetic input shape `(shape, total_elements)` for the
/// given model config.  Channel count is derived from `input_color`;
/// Stage 5 will refine the planar (YUY2/NV12) formats.
fn synthetic_input_shape(cfg: &ModelConfig) -> (Vec<usize>, usize) {
    let channels = match cfg.input_color {
        fluxframe_core::frame::PixelFormat::Gray8 => 1usize,
        fluxframe_core::frame::PixelFormat::Rgba => 4,
        // Rgb/Bgr/Yuy2/Nv12 — Stage 5 will refine planar formats.
        _ => 3,
    };
    let (n, c, h, w) = (
        1_usize,
        channels,
        cfg.input_height as usize,
        cfg.input_width as usize,
    );
    let shape = match cfg.input_layout {
        Layout::Nchw => vec![n, c, h, w],
        // `Hw` is invalid for inputs (validated by `ModelConfig`), but
        // synthesise something sensible anyway as a Stage 5 edge case.
        Layout::Hw => vec![h, w],
        // `Layout::Nhwc` plus any future `#[non_exhaustive]` variants
        // default to NHWC, the most common layout.
        Layout::Nhwc | _ => vec![n, h, w, c],
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
        // `TensorDType` is `#[non_exhaustive]`.
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxframe_core::frame::PixelFormat;
    use fluxframe_effects::ml::{Layout, ModelConfig};

    fn cfg_for(layout: Layout, color: PixelFormat, w: u32, h: u32) -> ModelConfig {
        let mut c = ModelConfig::new("test".into(), w, h);
        c.input_layout = layout;
        c.input_color = color;
        c
    }

    #[test]
    fn nhwc_rgb_shape_matches_dimensions() {
        let cfg = cfg_for(Layout::Nhwc, PixelFormat::Rgb, 4, 3);
        let (shape, elements) = synthetic_input_shape(&cfg);
        assert_eq!(shape, vec![1, 3, 4, 3]);
        assert_eq!(elements, 36);
    }

    #[test]
    fn nchw_gray8_shape_one_channel() {
        let cfg = cfg_for(Layout::Nchw, PixelFormat::Gray8, 4, 3);
        let (shape, elements) = synthetic_input_shape(&cfg);
        assert_eq!(shape, vec![1, 1, 3, 4]);
        assert_eq!(elements, 12);
    }

    #[test]
    fn rgba_yields_four_channels() {
        let cfg = cfg_for(Layout::Nhwc, PixelFormat::Rgba, 2, 2);
        let (shape, _) = synthetic_input_shape(&cfg);
        assert_eq!(shape, vec![1, 2, 2, 4]);
    }

    #[test]
    fn percentile_returns_zero_on_empty_slice() {
        assert_eq!(percentile(&[], 50), 0);
    }

    #[test]
    fn percentile_picks_correct_indices() {
        // `percentile` uses `(len * p / 100).min(len - 1)`, so for a
        // length-100 sorted slice the index is `p` (capped at 99).  With
        // `sorted[i] = i + 1` this means the 50th-percentile value is
        // 51, the 95th is 96, and the 100th saturates to 100.
        let sorted: Vec<u128> = (1..=100).collect();
        assert_eq!(percentile(&sorted, 50), 51);
        assert_eq!(percentile(&sorted, 95), 96);
        assert_eq!(percentile(&sorted, 100), 100);
    }
}
