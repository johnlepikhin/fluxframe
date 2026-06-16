//! [`SegmentationBase`] — the first stage of the composite pipeline.
//!
//! Owns the inference engine and the model-side scratch buffers,
//! decodes the engine's output tensor into a confidence mask at model
//! resolution, and tracks consecutive failures so a few transient
//! errors do not collapse the whole pipeline.
//!
//! The mask post-processing chain (threshold, dilate, feather, EMA)
//! and the resize-to-frame step run *after* this stage, in the
//! `MaskEffect` chain owned by the composite effect.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use fluxframe_core::Counters;
use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::{EffectError, InferenceError};
use fluxframe_core::frame::{PixelFormat, VideoFrame};
use fluxframe_core::traits::{InferenceEngine, InferenceInput};
use serde::Deserialize;
use tracing::{info, warn};

use crate::backend::{BackendOverrides, build_inference_engine};
use crate::ml::{InputLayout, ModelConfig, OutputLayout, OutputType, load_sidecar_or_placeholder};
use crate::processing::resize_rgb_bilinear;

/// Factory closure used to materialise the inference engine inside
/// [`SegmentationBase::prepare`]. Mirrors the factory the legacy
/// `background_blur` effect used so tests can keep their existing mock
/// engines.
///
/// Kept `pub(crate)` so the public API does not leak `ModelConfig`;
/// external callers go through
/// [`SegmentationBase::with_inference_factory`], which takes a generic
/// `impl FnOnce(...)` and boxes internally.
pub(crate) type InferenceFactory = Box<
    dyn FnOnce(&Path, ModelConfig) -> Result<Box<dyn InferenceEngine + Send>, InferenceError>
        + Send,
>;

/// User-configurable fields for [`SegmentationBase`]. The new
/// composite TOML schema exposes these under `[mask]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentationConfig {
    /// Path to the ONNX model. Required.
    pub model: PathBuf,
    /// Optional path to the model-config sidecar. Defaults to
    /// `<model>.toml` next to the model.
    #[serde(default)]
    pub model_config: Option<PathBuf>,
    /// Consecutive-failure budget before [`SegmentationBase::process`]
    /// returns [`SegmentationOutcome::Fatal`].
    #[serde(default = "default_fallback_threshold")]
    pub fallback_threshold: u32,
}

/// Default value for [`SegmentationConfig::fallback_threshold`]. Exported
/// so external callers (e.g. the composite builder) can reference the
/// same constant the serde `default` uses.
pub const DEFAULT_FALLBACK_THRESHOLD: u32 = 3;

/// Default class index for "person" when the model config does not
/// specify `person_class_index`. Matches the convention of common
/// person-segmentation datasets where background = 0, person = 1.
const DEFAULT_PERSON_CLASS_INDEX: u32 = 1;

/// Tolerance for matching a float-encoded integer label. The
/// CategoryMask convention stores integer labels as `f32`; values
/// within ±0.5 of the target class are accepted as that class.
const CATEGORY_LABEL_TOLERANCE: f32 = 0.5;

fn default_fallback_threshold() -> u32 {
    DEFAULT_FALLBACK_THRESHOLD
}

impl SegmentationConfig {
    /// Validate the configuration against the schema's invariants.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::InvalidConfig`] when the model path is
    /// empty, has a non-`.onnx` extension, or `fallback_threshold` is
    /// zero (zero would never tolerate a transient blip).
    pub fn validate(&self) -> Result<(), EffectError> {
        if self.model.as_os_str().is_empty() {
            return Err(invalid_config("`model` is required"));
        }
        match self.model.extension().and_then(|s| s.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("onnx") => {}
            Some(other) => {
                return Err(invalid_config(format!(
                    "`model` must point to an .onnx file, got extension `{other}`"
                )));
            }
            None => {
                return Err(invalid_config(
                    "`model` must point to an .onnx file (no extension found)",
                ));
            }
        }
        if self.fallback_threshold == 0 {
            return Err(invalid_config(
                "fallback_threshold must be >= 1; 0 means 'never tolerate' which is unintended",
            ));
        }
        Ok(())
    }
}

/// Outcome of a single [`SegmentationBase::process`] call.
///
/// `Ok` carries the dimensions of the freshly written mask so the
/// caller can borrow [`SegmentationBase::mask`] with the right shape.
/// `Fallback` is a soft transient error: the caller should pass the
/// current frame through unchanged. `Fatal` crossed the
/// `fallback_threshold` and demands propagation.
#[derive(Debug)]
pub enum SegmentationOutcome {
    /// Inference + decode succeeded. Mask lives in [`SegmentationBase::mask`].
    Ok {
        /// Mask width in pixels (equals model input width).
        width: u32,
        /// Mask height in pixels (equals model input height).
        height: u32,
    },
    /// Transient backend failure; not yet over threshold.
    Fallback,
    /// Unrecoverable failure — propagate as effect error.
    Fatal(EffectError),
}

/// Segmentation stage: resize → normalise → infer → decode → mask.
///
/// Consumers are typically the composite effect; tests can use it
/// directly with an injected [`InferenceFactory`].
pub struct SegmentationBase {
    config: SegmentationConfig,
    model_config: Option<ModelConfig>,
    engine: Option<Box<dyn InferenceEngine + Send>>,
    inference_factory: Option<InferenceFactory>,

    // Scratch.
    model_input_u8: Vec<u8>,
    /// `f32`-normalised, layout-packed model input. Length is
    /// `model_pixels * 3` and the channel order is dictated by the
    /// `model_config.input_layout`.
    model_input_f32: Vec<f32>,
    /// Raw segmentation mask decoded from the inference output, at
    /// model resolution.
    mask_raw: Vec<f32>,

    // Negotiated.
    frame_w: u32,
    frame_h: u32,
    model_w: u32,
    model_h: u32,

    // Fallback tracking.
    consecutive_failures: u32,
}

impl SegmentationBase {
    /// Construct an unprepared instance. Call [`Self::prepare`] before
    /// [`Self::process`].
    #[must_use]
    pub fn new(config: SegmentationConfig) -> Self {
        Self {
            config,
            model_config: None,
            engine: None,
            inference_factory: None,
            model_input_u8: Vec::new(),
            model_input_f32: Vec::new(),
            mask_raw: Vec::new(),
            frame_w: 0,
            frame_h: 0,
            model_w: 0,
            model_h: 0,
            consecutive_failures: 0,
        }
    }

    /// Install a custom inference-engine factory. Consumed once in
    /// [`Self::prepare`] in place of [`build_inference_engine`]. Use
    /// for tests that inject a mock engine without touching ORT.
    ///
    /// # Notes
    ///
    /// The factory signature unavoidably mentions [`ModelConfig`] —
    /// the engine cannot be built without it. The closure is boxed
    /// internally so the [`InferenceFactory`] alias stays a crate
    /// private implementation detail.
    #[must_use]
    pub fn with_inference_factory<F>(mut self, factory: F) -> Self
    where
        F: FnOnce(&Path, ModelConfig) -> Result<Box<dyn InferenceEngine + Send>, InferenceError>
            + Send
            + 'static,
    {
        self.inference_factory = Some(Box::new(factory));
        self
    }

    /// Dimensions of the inference model's input plane. Valid only
    /// after [`Self::prepare`] succeeded.
    #[must_use]
    pub fn model_dimensions(&self) -> (u32, u32) {
        (self.model_w, self.model_h)
    }

    /// Borrow the most recently produced mask. Length is
    /// `model_w * model_h`. Contents are meaningful only after a
    /// [`SegmentationOutcome::Ok`] return from [`Self::process`].
    #[must_use]
    pub fn mask(&self) -> &[f32] {
        &self.mask_raw
    }

    /// Mutable variant of [`Self::mask`]. Used by the composite to
    /// hand the buffer to the mask chain as a [`fluxframe_core::MaskPlane`].
    #[must_use]
    pub fn mask_mut(&mut self) -> &mut [f32] {
        &mut self.mask_raw
    }

    // ----- Stage 15 engine-lifecycle hooks -------------------------
    //
    // The Stage 15 supervisor wraps `CompositeEffect` in a
    // `ManagedComposite` that drops the engine after a deep-idle
    // window and rebuilds it on consumer resume. These accessors are
    // the minimal API surface that lets the wrapper drive the engine
    // slot without forcing every `VideoEffect` to gain lifecycle
    // methods. The wrapper itself lives in the CLI crate
    // (`cli::idle::managed_composite`).

    /// Move the loaded engine out of `self`. Subsequent
    /// [`Self::process`] calls return [`SegmentationOutcome::Fatal`]
    /// until [`Self::install_engine`] runs. Idempotent: returns
    /// `None` if no engine is currently held.
    #[must_use]
    pub fn take_engine(&mut self) -> Option<Box<dyn InferenceEngine + Send>> {
        self.engine.take()
    }

    /// Install a freshly built engine. Any previously held engine is
    /// dropped on assignment — the caller does not need to call
    /// [`Self::take_engine`] first. Use the explicit pair
    /// (`take_engine` + `install_engine`) when the caller needs to
    /// observe the old engine (e.g. for telemetry or staged teardown).
    pub fn install_engine(&mut self, engine: Box<dyn InferenceEngine + Send>) {
        self.engine = Some(engine);
    }

    /// `true` when an engine is currently loaded; `false` after
    /// [`Self::take_engine`] until the next [`Self::install_engine`].
    #[must_use]
    pub fn engine_loaded(&self) -> bool {
        self.engine.is_some()
    }

    /// Path to the ONNX model file. Comes from the original
    /// [`SegmentationConfig`] and is stable for the lifetime of this
    /// base, including across engine unload/reload.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        &self.config.model
    }

    /// Model-config sidecar that was loaded by the most recent
    /// [`Self::prepare`]. `None` before the first `prepare` call.
    /// The lifecycle wrapper needs this to call
    /// [`build_inference_engine`] when rebuilding the engine after a
    /// deep-idle drop.
    #[must_use]
    pub fn model_config(&self) -> Option<&ModelConfig> {
        self.model_config.as_ref()
    }

    /// Load model + sidecar, build the inference engine and allocate
    /// scratch.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::PrepareFailed`] when the sidecar is
    /// unreadable, [`EffectError::Inference`] when the engine cannot
    /// be constructed, or [`EffectError::InvalidConfig`] for non-RGB
    /// input.
    pub fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError> {
        if context.format != PixelFormat::Rgb {
            return Err(prepare_err(format!(
                "segmentation requires RGB input, got {:?}",
                context.format
            )));
        }
        self.config.validate()?;

        let model_config =
            load_sidecar_or_placeholder(&self.config.model).map_err(|e: InferenceError| {
                EffectError::Inference {
                    name: "segmentation".to_string(),
                    source: e,
                }
            })?;
        self.model_w = model_config.input_width;
        self.model_h = model_config.input_height;

        let engine = build_engine(
            self.inference_factory.take(),
            &self.config.model,
            model_config.clone(),
            context.counters.clone(),
        )?;
        self.engine = Some(engine);
        self.model_config = Some(model_config);

        self.frame_w = context.width;
        self.frame_h = context.height;
        let model_pixels = (self.model_w as usize) * (self.model_h as usize);
        // Use clear()+resize() instead of `vec![...]` so existing capacity
        // is reused on re-prepare; saves an allocation per reconfigure.
        self.model_input_u8.clear();
        self.model_input_u8.resize(model_pixels * 3, 0);
        self.model_input_f32.clear();
        self.model_input_f32.resize(model_pixels * 3, 0.0);
        self.mask_raw.clear();
        self.mask_raw.resize(model_pixels, 0.0);
        self.consecutive_failures = 0;

        // Engine pre-warm: a single synthetic infer with the zero
        // tensor primes ORT's arena allocator and triggers OpenVINO's
        // compile-on-first-infer kernels. Without it the operator's
        // first ~minute of camera frames pays the startup tax.
        let layout = self
            .model_config
            .as_ref()
            .map_or(InputLayout::Nhwc, |c| c.input_layout);
        let warmup_shape: [usize; 4] = match layout {
            InputLayout::Nhwc => [1, self.model_h as usize, self.model_w as usize, 3],
            InputLayout::Nchw => [1, 3, self.model_h as usize, self.model_w as usize],
        };
        let warmup_start = std::time::Instant::now();
        if let Some(engine) = self.engine.as_mut() {
            match engine.infer(InferenceInput {
                data: &self.model_input_f32,
                shape: &warmup_shape,
            }) {
                Ok(_) => info!(
                    warmup_us =
                        u64::try_from(warmup_start.elapsed().as_micros()).unwrap_or(u64::MAX),
                    "inference engine pre-warm complete",
                ),
                Err(e) => warn!(error = %e, "inference engine pre-warm failed; continuing anyway"),
            }
        }

        info!(
            model = %self.config.model.display(),
            frame = ?(context.width, context.height),
            model_input = ?(self.model_w, self.model_h),
            "segmentation prepared",
        );
        Ok(())
    }

    /// Run inference on `frame`. On success the mask at model
    /// resolution lives in [`Self::mask`].
    ///
    /// Sequence:
    /// 1. Bilinear resize frame → `model_input_u8`.
    /// 2. Pack into `model_input_f32` with the model's layout and
    ///    quantisation params.
    /// 3. Invoke the engine.
    /// 4. Decode the output tensor into `mask_raw`.
    ///
    /// On transient failure increments `consecutive_failures` and
    /// returns [`SegmentationOutcome::Fallback`] until the threshold
    /// is crossed.
    pub fn process(&mut self, frame: &VideoFrame, ctx: &mut FrameContext) -> SegmentationOutcome {
        if frame.format != PixelFormat::Rgb {
            return SegmentationOutcome::Fatal(process_err(format!(
                "unexpected pixel format {:?}",
                frame.format
            )));
        }
        if frame.width != self.frame_w || frame.height != self.frame_h {
            return SegmentationOutcome::Fatal(process_err(format!(
                "frame {}x{} differs from prepared {}x{}",
                frame.width, frame.height, self.frame_w, self.frame_h
            )));
        }
        let Some((scale, zero, layout)) = self
            .model_config
            .as_ref()
            .map(|c| (c.input_scale, c.input_zero_point, c.input_layout))
        else {
            return SegmentationOutcome::Fatal(process_err("process called before prepare"));
        };

        // 1. Resize frame → model_input_u8 (bilinear stretch).
        resize_rgb_bilinear(
            frame.data.as_slice(),
            frame.width,
            frame.height,
            &mut self.model_input_u8,
            self.model_w,
            self.model_h,
        );

        // 2. Normalise + pack.
        pack_input(
            &self.model_input_u8,
            &mut self.model_input_f32,
            layout,
            self.model_h,
            self.model_w,
            scale,
            zero,
        );

        // 3. Build input shape.
        let shape: [usize; 4] = match layout {
            InputLayout::Nhwc => [1, self.model_h as usize, self.model_w as usize, 3],
            InputLayout::Nchw => [1, 3, self.model_h as usize, self.model_w as usize],
        };

        // 4. Infer.
        let Some(engine) = self.engine.as_mut() else {
            return SegmentationOutcome::Fatal(process_err("process called before prepare"));
        };
        let t_inf = std::time::Instant::now();
        let output = match engine.infer(InferenceInput {
            data: &self.model_input_f32,
            shape: &shape,
        }) {
            Ok(out) => {
                self.consecutive_failures = 0;
                out
            }
            Err(e) => {
                self.consecutive_failures += 1;
                let fails = self.consecutive_failures;
                if fails.is_power_of_two() {
                    warn!(
                        error = %e,
                        consecutive = fails,
                        threshold = self.config.fallback_threshold,
                        "inference failed; passing frame through"
                    );
                }
                if fails >= self.config.fallback_threshold {
                    return SegmentationOutcome::Fatal(EffectError::Inference {
                        name: "segmentation".to_string(),
                        source: e,
                    });
                }
                return SegmentationOutcome::Fallback;
            }
        };
        ctx.telemetry.record_inference(t_inf.elapsed());

        // 5. Decode output into mask_raw.
        let Some(model_config) = self.model_config.as_ref() else {
            return SegmentationOutcome::Fatal(process_err(
                "model_config missing — process called before prepare",
            ));
        };
        if let Err(err) = decode_mask(
            &output.data,
            &output.shape,
            &mut self.mask_raw,
            model_config,
        ) {
            return SegmentationOutcome::Fatal(err);
        }

        SegmentationOutcome::Ok {
            width: self.model_w,
            height: self.model_h,
        }
    }
}

/// Resolve the inference factory and build the engine; mirrors the
/// path the legacy `background_blur` followed so its existing factory
/// override hooks keep working unchanged.
fn build_engine(
    factory: Option<InferenceFactory>,
    model_path: &Path,
    model_config: ModelConfig,
    counters: Option<Arc<Counters>>,
) -> Result<Box<dyn InferenceEngine + Send>, EffectError> {
    match factory {
        Some(f) => f(model_path, model_config),
        None => build_inference_engine(
            model_path,
            model_config,
            BackendOverrides::current(),
            counters,
        ),
    }
    .map_err(|e| EffectError::Inference {
        name: "segmentation".to_string(),
        source: e,
    })
}

/// Pack an HWC `u8` frame into the layout/dtype expected by an ONNX model.
///
/// * `src_hwc_u8` is `h * w * 3` bytes in RGB HWC order.
/// * `dst_f32` is `h * w * 3` floats, layout dictated by `layout`.
/// * Per-pixel transform: `out = (v - zero) * scale`.
fn pack_input(
    src_hwc_u8: &[u8],
    dst_f32: &mut [f32],
    layout: InputLayout,
    h: u32,
    w: u32,
    scale: f32,
    zero: f32,
) {
    debug_assert_eq!(src_hwc_u8.len(), (h as usize) * (w as usize) * 3);
    debug_assert_eq!(dst_f32.len(), (h as usize) * (w as usize) * 3);
    let h = h as usize;
    let w = w as usize;
    let pixels = h * w;
    match layout {
        InputLayout::Nhwc => {
            for (out, &v) in dst_f32.iter_mut().zip(src_hwc_u8.iter()) {
                *out = (f32::from(v) - zero) * scale;
            }
        }
        InputLayout::Nchw => {
            for c in 0..3 {
                let plane = c * pixels;
                for y in 0..h {
                    let row_src = y * w * 3;
                    let row_dst = plane + y * w;
                    for x in 0..w {
                        let v = src_hwc_u8[row_src + x * 3 + c];
                        dst_f32[row_dst + x] = (f32::from(v) - zero) * scale;
                    }
                }
            }
        }
    }
}

/// Decode an inference output tensor into a `[0, 1]` confidence mask
/// at the model's output resolution.
fn decode_mask(
    data: &[f32],
    shape: &[usize],
    dst: &mut [f32],
    config: &ModelConfig,
) -> Result<(), EffectError> {
    let expected = dst.len();
    match config.output_type {
        OutputType::Mask => {
            if data.len() < expected {
                return Err(process_err(format!(
                    "model output has {} elements, expected at least {expected}",
                    data.len()
                )));
            }
            dst.copy_from_slice(&data[..expected]);
        }
        OutputType::Probabilities | OutputType::Logits => {
            let is_logits = matches!(config.output_type, OutputType::Logits);
            decode_two_class(data, shape, dst, config.output_layout, is_logits)?;
        }
        OutputType::CategoryMask => {
            let class = config
                .person_class_index
                .unwrap_or(DEFAULT_PERSON_CLASS_INDEX);
            if data.len() < expected {
                return Err(process_err("category mask shorter than expected"));
            }
            let class_f = class as f32;
            for (out, &v) in dst.iter_mut().zip(data.iter()) {
                *out = if (v - class_f).abs() < CATEGORY_LABEL_TOLERANCE {
                    1.0
                } else {
                    0.0
                };
            }
        }
    }
    Ok(())
}

/// Decode a 1- or 2-class probability/logit tensor into a foreground
/// mask.
///
/// The tensor layout (HW / NHWC / NCHW) decides where the bg/fg
/// channels live; never derive this from `shape` magic — older code
/// did and broke on planar NCHW outputs.
fn decode_two_class(
    data: &[f32],
    shape: &[usize],
    dst: &mut [f32],
    layout: OutputLayout,
    is_logits: bool,
) -> Result<(), EffectError> {
    let pixels = dst.len();
    match layout {
        OutputLayout::Hw => {
            if data.len() < pixels {
                return Err(process_err(format!(
                    "output {} too small for {} mask pixels (HW)",
                    data.len(),
                    pixels
                )));
            }
            for (out, &v) in dst.iter_mut().zip(data.iter()) {
                *out = if is_logits { sigmoid(v) } else { v };
            }
        }
        OutputLayout::Nhwc => {
            if data.len() == pixels * 2 {
                for i in 0..pixels {
                    let bg = data[i * 2];
                    let fg = data[i * 2 + 1];
                    dst[i] = if is_logits {
                        softmax_foreground(bg, fg)
                    } else {
                        fg
                    };
                }
            } else if data.len() >= pixels && data.len() / pixels == 1 {
                for (out, &v) in dst.iter_mut().zip(data.iter()) {
                    *out = if is_logits { sigmoid(v) } else { v };
                }
            } else {
                return Err(process_err(format!(
                    "NHWC output has {} elements; expected {} (C=1) or {} (C=2) — shape {:?}",
                    data.len(),
                    pixels,
                    pixels * 2,
                    shape
                )));
            }
        }
        OutputLayout::Nchw => {
            if data.len() == pixels * 2 {
                let fg_plane = pixels;
                for i in 0..pixels {
                    let bg = data[i];
                    let fg = data[fg_plane + i];
                    dst[i] = if is_logits {
                        softmax_foreground(bg, fg)
                    } else {
                        fg
                    };
                }
            } else if data.len() == pixels {
                for (out, &v) in dst.iter_mut().zip(data.iter()) {
                    *out = if is_logits { sigmoid(v) } else { v };
                }
            } else {
                return Err(process_err(format!(
                    "NCHW output has {} elements; expected {} (C=1) or {} (C=2) — shape {:?}",
                    data.len(),
                    pixels,
                    pixels * 2,
                    shape
                )));
            }
        }
    }
    Ok(())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically stable softmax for the foreground class of a two-class
/// (background, foreground) logit pair.
fn softmax_foreground(background: f32, foreground: f32) -> f32 {
    let max = background.max(foreground);
    let numerator = (foreground - max).exp();
    let denominator = (background - max).exp() + numerator;
    numerator / denominator
}

fn invalid_config(reason: impl Into<String>) -> EffectError {
    EffectError::InvalidConfig {
        name: "segmentation".to_string(),
        reason: reason.into(),
        hint: None,
    }
}

fn prepare_err(reason: impl Into<String>) -> EffectError {
    EffectError::PrepareFailed {
        name: "segmentation".to_string(),
        reason: reason.into(),
    }
}

fn process_err(reason: impl Into<String>) -> EffectError {
    EffectError::ProcessFailed {
        name: "segmentation".to_string(),
        reason: reason.into(),
    }
}

// Compile-time guard: SegmentationBase must stay `Send` so the
// composite effect can move it across thread boundaries.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<SegmentationBase>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_parses_with_defaults() {
        let raw = r#"
model = "/tmp/dummy.onnx"
"#;
        let parsed: SegmentationConfig = toml::from_str(raw).expect("parse ok");
        assert_eq!(parsed.fallback_threshold, 3);
        assert!(parsed.model_config.is_none());
    }

    #[test]
    fn validate_rejects_missing_model() {
        let cfg = SegmentationConfig {
            model: PathBuf::new(),
            model_config: None,
            fallback_threshold: 3,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_onnx() {
        let cfg = SegmentationConfig {
            model: PathBuf::from("/tmp/model.bin"),
            model_config: None,
            fallback_threshold: 3,
        };
        let err = cfg.validate().expect_err("non-onnx must fail");
        assert!(format!("{err}").contains("onnx"));
    }

    #[test]
    fn validate_rejects_zero_threshold() {
        let cfg = SegmentationConfig {
            model: PathBuf::from("/tmp/m.onnx"),
            model_config: None,
            fallback_threshold: 0,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn decode_mask_passes_through_mask_output() {
        let cfg = ModelConfig::new("t", 2, 2);
        let data = vec![0.1_f32, 0.9, 0.5, 0.0];
        let mut dst = vec![0.0_f32; 4];
        decode_mask(&data, &[2, 2], &mut dst, &cfg).expect("ok");
        assert_eq!(dst, data);
    }

    #[test]
    fn decode_mask_handles_category_mask() {
        let mut cfg = ModelConfig::new("t", 2, 2);
        cfg.output_type = OutputType::CategoryMask;
        cfg.person_class_index = Some(15);
        let data = vec![0.0_f32, 15.0, 7.0, 15.0];
        let mut dst = vec![0.0_f32; 4];
        decode_mask(&data, &[2, 2], &mut dst, &cfg).expect("ok");
        assert_eq!(dst, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn decode_mask_softmaxes_two_channel_logits_nhwc() {
        let mut cfg = ModelConfig::new("t", 1, 1);
        cfg.output_type = OutputType::Logits;
        cfg.output_layout = OutputLayout::Nhwc;
        let data = vec![-10.0_f32, 10.0];
        let mut dst = vec![0.0_f32; 1];
        decode_mask(&data, &[1, 1, 1, 2], &mut dst, &cfg).expect("ok");
        assert!(dst[0] > 0.99, "got {}", dst[0]);
    }

    #[test]
    fn pack_input_nhwc_matches_identity() {
        let src = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let mut dst = vec![0.0f32; 12];
        pack_input(&src, &mut dst, InputLayout::Nhwc, 2, 2, 1.0, 0.0);
        let expected: Vec<f32> = src.iter().map(|&v| f32::from(v)).collect();
        assert_eq!(dst, expected);
    }

    #[test]
    fn pack_input_nchw_transposes_channels() {
        let src = vec![10u8, 20, 30, 40, 50, 60];
        let mut dst = vec![0.0f32; 6];
        pack_input(&src, &mut dst, InputLayout::Nchw, 1, 2, 1.0, 0.0);
        assert_eq!(dst, vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0]);
    }
}
