//! `fluxframe benchmark` — Stage 3 implements the inference-only path.
//!
//! Full end-to-end pipeline benchmarking (capture + chain + sink)
//! lands in Stage 5.

use std::time::{Duration, Instant};

use fluxframe_core::FluxError;
use fluxframe_core::traits::{InferenceEngine, InferenceInput};
use fluxframe_effects::ml::{InputLayout, ModelConfig, OnnxEngine, TensorDType};
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

    // Note: OnnxEngine::load already canonicalises and returns ModelNotFound
    // for missing files.  We don't pre-check exists() here — it would just
    // duplicate the syscall.

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
    let duration = Duration::from_secs(args.duration.into());
    let (mut timings, elapsed) = run_inference_loop(&mut engine, &data, &shape, duration)?;

    if timings.is_empty() {
        return Err(FluxError::Config {
            reason: "duration too short — no inferences completed".into(),
            hint: Some("increase --duration to at least 1 second".into()),
        });
    }
    timings.sort_unstable();

    print_report(&info, layout, dtype, timings.len(), elapsed, &timings);
    Ok(())
}

/// Build the synthetic input shape `(shape, total_elements)` for the
/// given model config.  Channel count is derived from `input_color`;
/// Stage 5 will refine the planar (YUY2/NV12) formats.
fn synthetic_input_shape(cfg: &ModelConfig) -> (Vec<usize>, usize) {
    use fluxframe_core::frame::PixelFormat;
    // Rgb / Bgr / Yuy2 / Nv12 — Stage 5 will model planar (YUY2/NV12)
    // formats explicitly; treating them as 3-channel here is a
    // benchmark-only approximation.  Non_exhaustive future variants also
    // default to 3.
    // FIXME(stage-5): refine planar formats.
    let channels = match cfg.input_color {
        PixelFormat::Gray8 => 1usize,
        PixelFormat::Rgba => 4,
        _ => 3,
    };
    let (n, c, h, w) = (
        1_usize,
        channels,
        cfg.input_height as usize,
        cfg.input_width as usize,
    );
    let shape = match cfg.input_layout {
        InputLayout::Nhwc => vec![n, h, w, c],
        InputLayout::Nchw => vec![n, c, h, w],
        _ => {
            tracing::warn!(
                layout = ?cfg.input_layout,
                "unknown InputLayout variant — defaulting to NHWC",
            );
            vec![n, h, w, c]
        }
    };
    let elements = shape.iter().product();
    (shape, elements)
}

/// Run inference repeatedly for at least `duration`, collecting per-call
/// timings in microseconds.  Returns the timings and the total elapsed
/// wall-clock time of the loop.
fn run_inference_loop(
    engine: &mut OnnxEngine,
    data: &[f32],
    shape: &[usize],
    duration: Duration,
) -> Result<(Vec<u128>, Duration), FluxError> {
    let start = Instant::now();
    let mut timings = Vec::with_capacity(estimate_capacity(duration));
    while start.elapsed() < duration {
        let t = Instant::now();
        engine.infer(InferenceInput { data, shape })?;
        timings.push(t.elapsed().as_micros());
    }
    let elapsed = start.elapsed();
    Ok((timings, elapsed))
}

/// Conservative upper bound on inferences per run: 200 fps × duration_secs.
fn estimate_capacity(duration: Duration) -> usize {
    (duration.as_secs_f64() * 200.0).ceil() as usize
}

fn print_report(
    info: &fluxframe_core::traits::ModelInfo,
    layout: InputLayout,
    dtype: TensorDType,
    total: usize,
    elapsed: Duration,
    timings: &[u128],
) {
    let p50 = percentile(timings, 50);
    let p95 = percentile(timings, 95);
    let max = timings.last().copied().unwrap_or(0);

    println!("model: {}", info.name);
    println!(
        "input: {}x{} {} {:?}",
        info.input_width,
        info.input_height,
        dtype_label(dtype),
        layout,
    );
    println!("runs:  {total} (in {elapsed:?})");
    println!("p50: {p50} us   p95: {p95} us   max: {max} us");
}

fn percentile(sorted: &[u128], p: usize) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    // Standard nearest-rank: floor((p/100) * (n-1))
    let idx = (sorted.len().saturating_sub(1) * p) / 100;
    sorted[idx]
}

fn dtype_label(dtype: TensorDType) -> &'static str {
    match dtype {
        TensorDType::F32 => "f32",
        TensorDType::U8 => "u8",
        TensorDType::I8 => "i8",
        // `TensorDType` is `#[non_exhaustive]`.
        _ => {
            tracing::warn!(dtype = ?dtype, "unknown TensorDType variant");
            "unknown"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxframe_core::frame::PixelFormat;
    use fluxframe_effects::ml::{InputLayout, ModelConfig};

    fn cfg_for(layout: InputLayout, color: PixelFormat, w: u32, h: u32) -> ModelConfig {
        let mut c = ModelConfig::new("test", w, h);
        c.input_layout = layout;
        c.input_color = color;
        c
    }

    #[test]
    fn nhwc_rgb_shape_matches_dimensions() {
        let cfg = cfg_for(InputLayout::Nhwc, PixelFormat::Rgb, 4, 3);
        let (shape, elements) = synthetic_input_shape(&cfg);
        assert_eq!(shape, vec![1, 3, 4, 3]);
        assert_eq!(elements, 36);
    }

    #[test]
    fn nchw_gray8_shape_one_channel() {
        let cfg = cfg_for(InputLayout::Nchw, PixelFormat::Gray8, 4, 3);
        let (shape, elements) = synthetic_input_shape(&cfg);
        assert_eq!(shape, vec![1, 1, 3, 4]);
        assert_eq!(elements, 12);
    }

    #[test]
    fn rgba_yields_four_channels() {
        let cfg = cfg_for(InputLayout::Nhwc, PixelFormat::Rgba, 2, 2);
        let (shape, _) = synthetic_input_shape(&cfg);
        assert_eq!(shape, vec![1, 2, 2, 4]);
    }

    #[test]
    fn percentile_returns_zero_on_empty_slice() {
        assert_eq!(percentile(&[], 50), 0);
    }

    #[test]
    fn percentile_picks_correct_indices() {
        // Standard nearest-rank definition: idx = floor((n-1) * p / 100).
        // For a length-100 sorted slice with sorted[i] = i + 1, this gives
        // p50 -> sorted[49] = 50, p95 -> sorted[94] = 95,
        // p100 -> sorted[99] = 100.
        let sorted: Vec<u128> = (1..=100).collect();
        assert_eq!(percentile(&sorted, 50), 50);
        assert_eq!(percentile(&sorted, 95), 95);
        assert_eq!(percentile(&sorted, 100), 100);
    }
}
