//! `background_blur` — first production-grade FluxFrame effect.
//!
//! Pipeline (RGB-in, RGB-out, single pass per frame):
//! 1. Resize the frame to the model's input resolution (bilinear stretch;
//!    aspect-ratio-preserving letterboxing is a Stage 5 follow-up).
//! 2. Normalise to f32 using `input_scale` + `input_zero_point` and pack
//!    into the layout (`NHWC`/`NCHW`) the model expects.
//! 3. Run inference via the supplied [`OnnxEngine`].
//! 4. Decode the output tensor into a `[0,1]` confidence mask.
//! 5. Resize the mask back to the frame resolution.
//! 6. Temporal-smooth, threshold, dilate, feather the mask.
//! 7. Box-blur the original frame (background).
//! 8. Alpha-composite original (foreground) over blurred (background)
//!    using the post-processed mask.
//!
//! Configurable parameters (TOML keys mirror spec §13):
//!
//! ```toml
//! [effects.background_blur]
//! model = "./models/person-seg.onnx"
//! # model_config is auto-discovered as <model>.toml unless overridden
//! blur_radius = 21
//! blur_passes = 2
//! mask_threshold = 0.5
//! mask_smoothing = 0.65
//! mask_feather_radius = 7
//! mask_dilate = 1
//! fallback_threshold = 3
//! ```

use std::path::PathBuf;

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::{EffectError, InferenceError};
use fluxframe_core::frame::{PixelFormat, VideoFrame};
use fluxframe_core::traits::{InferenceEngine, InferenceInput, RawEffectParams, VideoEffect};
use serde::Deserialize;
use tracing::{debug, info, warn};

use crate::ml::{
    InputLayout, ModelConfig, OnnxEngine, OutputLayout, OutputType, load_sidecar_or_placeholder,
};
use crate::processing::{
    alpha_composite_rgb_in_place, box_blur_rgb, dilate, feather, resize_mask_bilinear,
    resize_rgb_bilinear, smooth_temporal, threshold,
};

/// Default values for [`BlurConfig`].
///
/// Exposed publicly so downstream tests, documentation and tooling can
/// reference the canonical defaults without duplicating literals.
pub mod defaults {
    /// Half-kernel radius of the box blur (pixels).
    pub const BLUR_RADIUS: u32 = 21;
    /// Number of box-blur passes.
    pub const BLUR_PASSES: u32 = 2;
    /// Mask binarisation threshold in `[0, 1]`.
    pub const MASK_THRESHOLD: f32 = 0.5;
    /// EMA factor applied to the previous mask in `[0, 1]`.
    pub const MASK_SMOOTHING: f32 = 0.65;
    /// Feather (mask-blur) radius in pixels.
    pub const MASK_FEATHER_RADIUS: u32 = 7;
    /// Number of 3×3 dilation iterations applied before feathering.
    pub const MASK_DILATE: u32 = 1;
    /// Consecutive-failure budget before escalating to a hard error.
    pub const FALLBACK_THRESHOLD: u32 = 3;
}

// ---------------------------------------------------------------------------
// Bounds for `BlurConfig::validate` — kept module-level so tests can
// reuse the same constants and stay drift-free.
// ---------------------------------------------------------------------------
const MAX_BLUR_RADIUS: u32 = 256;
const MAX_BLUR_PASSES: u32 = 16;
const MAX_DILATE: u32 = 32;
const MAX_FEATHER_RADIUS: u32 = 64;

/// Background-blur effect.
///
/// See module-level documentation for the per-frame pipeline; see
/// [`BlurConfig`] for the TOML schema.
pub struct BackgroundBlurEffect {
    /// User-supplied configuration; populated by [`VideoEffect::configure`].
    /// `None` until `configure` has been called.
    config: Option<BlurConfig>,
    model_config: Option<ModelConfig>,
    /// Inference engine.  Boxed behind the trait so tests can substitute a
    /// mock implementation without touching ONNX Runtime.
    engine: Option<Box<dyn InferenceEngine + Send>>,
    // Frame-resolution scratch:
    blurred: Vec<u8>,
    blur_scratch: Vec<u8>,
    mask_full: Vec<f32>,
    mask_prev: Vec<f32>,
    mask_dilate_scratch: Vec<f32>,
    mask_feather_scratch: Vec<f32>,
    // Model-resolution scratch:
    model_input_u8: Vec<u8>,
    /// `f32`-normalised, layout-packed model input.  Length is
    /// `model_pixels * 3` and the channel order is dictated by
    /// `model_config.input_layout`.
    model_input_f32: Vec<f32>,
    mask_raw: Vec<f32>,
    // Negotiated:
    frame_w: u32,
    frame_h: u32,
    model_w: u32,
    model_h: u32,
    // Fallback tracking:
    consecutive_failures: u32,
}

/// User-configurable parameters for [`BackgroundBlurEffect`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlurConfig {
    /// Path to the ONNX model.  Required.
    pub model: PathBuf,
    /// Optional path to a model-config TOML.  Defaults to `<model>.toml`.
    #[serde(default)]
    pub model_config: Option<PathBuf>,
    /// Half-kernel radius of the box blur (pixels).
    #[serde(default = "default_blur_radius")]
    pub blur_radius: u32,
    /// Number of box-blur passes.
    #[serde(default = "default_blur_passes")]
    pub blur_passes: u32,
    /// Mask binarisation threshold in `[0, 1]`.
    #[serde(default = "default_mask_threshold")]
    pub mask_threshold: f32,
    /// EMA factor applied to the previous mask in `[0, 1]` (closer to
    /// 1 → smoother, more lag).
    #[serde(default = "default_mask_smoothing")]
    pub mask_smoothing: f32,
    /// Feather (mask-blur) radius in pixels.
    #[serde(default = "default_mask_feather_radius")]
    pub mask_feather_radius: u32,
    /// Number of 3×3 dilation iterations applied before feathering.
    #[serde(default = "default_mask_dilate")]
    pub mask_dilate: u32,
    /// After this many consecutive inference failures the effect
    /// returns an error rather than silently passing frames through.
    #[serde(default = "default_fallback_threshold")]
    pub fallback_threshold: u32,
}

fn default_blur_radius() -> u32 {
    defaults::BLUR_RADIUS
}
fn default_blur_passes() -> u32 {
    defaults::BLUR_PASSES
}
fn default_mask_threshold() -> f32 {
    defaults::MASK_THRESHOLD
}
fn default_mask_smoothing() -> f32 {
    defaults::MASK_SMOOTHING
}
fn default_mask_feather_radius() -> u32 {
    defaults::MASK_FEATHER_RADIUS
}
fn default_mask_dilate() -> u32 {
    defaults::MASK_DILATE
}
fn default_fallback_threshold() -> u32 {
    defaults::FALLBACK_THRESHOLD
}

/// Result of the per-frame inference step inside [`BackgroundBlurEffect`].
enum InferenceOutcome {
    /// Inference succeeded; carry the output tensor onward.
    Ok(fluxframe_core::traits::InferenceOutput),
    /// A transient failure occurred but we have not yet crossed
    /// `fallback_threshold` — the caller should pass the frame through.
    Fallback,
    /// We crossed `fallback_threshold` (or some other unrecoverable
    /// error happened); the caller must surface this error.
    Fatal(EffectError),
}

impl BackgroundBlurEffect {
    /// Effect name as registered in the [`crate::EffectRegistry`].
    pub const NAME: &'static str = "background_blur";

    /// Construct an empty instance; call `configure` and `prepare` before
    /// `process`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: None,
            model_config: None,
            engine: None,
            blurred: Vec::new(),
            blur_scratch: Vec::new(),
            mask_full: Vec::new(),
            mask_prev: Vec::new(),
            mask_dilate_scratch: Vec::new(),
            mask_feather_scratch: Vec::new(),
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

    /// Construct an instance with a pre-built inference engine.  Used by
    /// tests to inject a mock without going through `prepare()` (and thus
    /// without loading an ONNX file).  Production code should prefer
    /// `configure` + `prepare`.
    #[cfg(test)]
    fn with_engine(engine: Box<dyn InferenceEngine + Send>) -> Self {
        let mut effect = Self::new();
        effect.engine = Some(engine);
        effect
    }

    fn validate_frame(&self, frame: &VideoFrame) -> Result<(), EffectError> {
        if frame.format != PixelFormat::Rgb {
            return Err(process_err(format!(
                "unexpected pixel format {:?}",
                frame.format
            )));
        }
        if frame.width != self.frame_w || frame.height != self.frame_h {
            return Err(process_err(format!(
                "frame {}x{} differs from prepared {}x{}",
                frame.width, frame.height, self.frame_w, self.frame_h
            )));
        }
        Ok(())
    }

    /// Resize + normalise + infer.  Updates `self.consecutive_failures`
    /// and returns an outcome describing the appropriate next step.
    fn run_inference(&mut self, frame: &VideoFrame) -> InferenceOutcome {
        // Snapshot the small scalars we need from `model_config` to keep
        // the `&self.engine` mutable borrow disjoint from the immutable
        // read of `self.model_config`.  The hot path does no heap traffic.
        let Some((scale, zero, layout)) = self
            .model_config
            .as_ref()
            .map(|c| (c.input_scale, c.input_zero_point, c.input_layout))
        else {
            return InferenceOutcome::Fatal(process_err("process called before prepare"));
        };
        let fallback_threshold = match self.config.as_ref() {
            Some(c) => c.fallback_threshold,
            None => {
                return InferenceOutcome::Fatal(process_err("process called before configure"));
            }
        };

        // 1. Resize frame → model_input_u8 (bilinear stretch).
        // TODO(stage-5): switch to fit_letterbox + padded resize for non-matching aspect ratios.
        resize_rgb_bilinear(
            frame.data.as_slice(),
            frame.width,
            frame.height,
            &mut self.model_input_u8,
            self.model_w,
            self.model_h,
        );

        // 2. Normalise into f32 according to model config and pack into the
        //    layout the model expects.
        pack_input(
            &self.model_input_u8,
            &mut self.model_input_f32,
            layout,
            self.model_h,
            self.model_w,
            scale,
            zero,
        );

        // 3. Build input tensor shape from layout.  Stack-allocated to
        // avoid a heap alloc per frame on the hot path.
        let shape: [usize; 4] = match layout {
            InputLayout::Nhwc => [1, self.model_h as usize, self.model_w as usize, 3],
            InputLayout::Nchw => [1, 3, self.model_h as usize, self.model_w as usize],
        };

        // 4. Run inference.
        let Some(engine) = self.engine.as_mut() else {
            return InferenceOutcome::Fatal(process_err("process called before prepare"));
        };
        // `&shape` coerces from `&[usize; 4]` to `&[usize]` automatically.
        match engine.infer(InferenceInput {
            data: &self.model_input_f32,
            shape: &shape,
        }) {
            Ok(out) => {
                self.consecutive_failures = 0;
                InferenceOutcome::Ok(out)
            }
            Err(e) => {
                self.consecutive_failures += 1;
                let fails = self.consecutive_failures;
                // First failure of a burst: drop the stale `mask_prev` so
                // when inference recovers, the EMA does not blend a
                // pre-failure mask into the post-recovery one.
                if fails == 1 {
                    self.mask_prev.fill(0.0);
                }
                // Rate-limit warnings: emit the first failure, then only on
                // power-of-two boundaries (1, 2, 4, 8, …).  Avoids flooding
                // logs at 30 fail/s while still surfacing growing trouble.
                if fails.is_power_of_two() {
                    warn!(
                        error = %e,
                        consecutive = fails,
                        threshold = fallback_threshold,
                        "inference failed; passing frame through"
                    );
                }
                if fails >= fallback_threshold {
                    InferenceOutcome::Fatal(EffectError::Inference {
                        name: Self::NAME.to_string(),
                        source: e,
                    })
                } else {
                    InferenceOutcome::Fallback
                }
            }
        }
    }

    /// Decode the raw output into a confidence mask, resize to frame
    /// resolution, then apply temporal smoothing, threshold, dilate and
    /// feather in place on `self.mask_full`.
    fn build_mask(
        &mut self,
        output: &fluxframe_core::traits::InferenceOutput,
    ) -> Result<(), EffectError> {
        let Some(model_config) = self.model_config.as_ref() else {
            return Err(process_err("process called before prepare"));
        };
        let Some(config) = self.config.as_ref() else {
            return Err(process_err("process called before configure"));
        };

        // 5. Convert output into a [0,1] mask at model resolution.
        decode_mask(
            &output.data,
            &output.shape,
            &mut self.mask_raw,
            model_config,
        )?;

        // 6. Resize mask to frame resolution.
        resize_mask_bilinear(
            &self.mask_raw,
            self.model_w,
            self.model_h,
            &mut self.mask_full,
            self.frame_w,
            self.frame_h,
        );

        // 7. Temporal smoothing → threshold → dilate → feather.
        // `smooth_temporal` writes the EMA result into its first argument
        // (`mask_prev`).  Swap the two buffers so `mask_full` ends up with
        // the freshest mask and `mask_prev` retains the previous frame's
        // mask, ready to be overwritten on the next call.  Avoids an 8.3MB
        // memcpy per 1080p frame.
        smooth_temporal(&mut self.mask_prev, &self.mask_full, config.mask_smoothing);
        std::mem::swap(&mut self.mask_full, &mut self.mask_prev);

        threshold(&mut self.mask_full, config.mask_threshold);
        dilate(
            &mut self.mask_full,
            &mut self.mask_dilate_scratch,
            self.frame_w,
            self.frame_h,
            config.mask_dilate,
        );
        feather(
            &mut self.mask_full,
            &mut self.mask_feather_scratch,
            self.frame_w,
            self.frame_h,
            config.mask_feather_radius,
        );
        Ok(())
    }

    /// Box-blur the frame and composite original + blurred via the mask.
    fn blend(&mut self, frame: &mut VideoFrame) -> Result<(), EffectError> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| process_err("process called before configure"))?;

        // 8. Blur the frame (background candidate).
        box_blur_rgb(
            frame.data.as_slice(),
            &mut self.blurred,
            &mut self.blur_scratch,
            self.frame_w,
            self.frame_h,
            config.blur_radius,
            config.blur_passes,
        );

        // 9. Promote the frame to Owned upfront so the in-place composite
        //    below can mutate it without paying a silent CoW.  When the
        //    buffer is already Owned (the common case from
        //    `sample_to_frame` in the GStreamer adapter) this is a no-op.
        frame.data.make_owned();

        // 10. Alpha composite in place: read fg from `frame_bytes`, blend
        //     with `self.blurred` (bg) by `self.mask_full`, write back to
        //     `frame_bytes`.  Saves ~18.6MB/frame at 1080p compared with
        //     a scratch round-trip.
        let frame_bytes = frame
            .data
            .as_mut_owned()
            .expect("VideoFrame::make_owned guarantees as_mut_owned returns Some");
        alpha_composite_rgb_in_place(frame_bytes, &self.blurred, &self.mask_full);
        Ok(())
    }
}

impl BlurConfig {
    fn validate(&self) -> Result<(), EffectError> {
        if self.model.as_os_str().is_empty() {
            return Err(invalid_config("`model` is required"));
        }
        // Reject obvious non-ONNX paths up front: the user almost certainly
        // typoed a sibling file (e.g. the sidecar TOML).  Doing this here —
        // rather than only at engine load time — gives a clear, early
        // diagnostic.
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
        if !(0.0..=1.0).contains(&self.mask_threshold) {
            return Err(invalid_config(format!(
                "mask_threshold must be in [0,1], got {}",
                self.mask_threshold
            )));
        }
        if !(0.0..=1.0).contains(&self.mask_smoothing) {
            return Err(invalid_config(format!(
                "mask_smoothing must be in [0,1], got {}",
                self.mask_smoothing
            )));
        }
        if self.blur_radius > MAX_BLUR_RADIUS {
            return Err(invalid_config(format!(
                "blur_radius must be <= {MAX_BLUR_RADIUS}, got {}",
                self.blur_radius
            )));
        }
        if self.blur_passes > MAX_BLUR_PASSES {
            return Err(invalid_config(format!(
                "blur_passes must be <= {MAX_BLUR_PASSES}, got {}",
                self.blur_passes
            )));
        }
        if self.mask_dilate > MAX_DILATE {
            return Err(invalid_config(format!(
                "mask_dilate must be <= {MAX_DILATE}, got {}",
                self.mask_dilate
            )));
        }
        if self.mask_feather_radius > MAX_FEATHER_RADIUS {
            return Err(invalid_config(format!(
                "mask_feather_radius must be <= {MAX_FEATHER_RADIUS}, got {}",
                self.mask_feather_radius
            )));
        }
        if self.fallback_threshold == 0 {
            return Err(invalid_config(
                "fallback_threshold must be >= 1; 0 means 'never tolerate' which is unintended",
            ));
        }
        Ok(())
    }
}

/// Build an [`EffectError::InvalidConfig`] with the effect name pre-filled.
fn invalid_config(reason: impl Into<String>) -> EffectError {
    EffectError::InvalidConfig {
        name: BackgroundBlurEffect::NAME.to_string(),
        reason: reason.into(),
    }
}

/// Build an [`EffectError::ProcessFailed`] with the effect name pre-filled.
fn process_err(reason: impl Into<String>) -> EffectError {
    EffectError::ProcessFailed {
        name: BackgroundBlurEffect::NAME.to_string(),
        reason: reason.into(),
    }
}

/// Build an [`EffectError::PrepareFailed`] with the effect name pre-filled.
fn prepare_err(reason: impl Into<String>) -> EffectError {
    EffectError::PrepareFailed {
        name: BackgroundBlurEffect::NAME.to_string(),
        reason: reason.into(),
    }
}

impl Default for BackgroundBlurEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoEffect for BackgroundBlurEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: BlurConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| invalid_config(e.to_string()))?;
        cfg.validate()?;
        self.config = Some(cfg);
        Ok(())
    }

    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError> {
        if context.format != PixelFormat::Rgb {
            return Err(prepare_err(format!(
                "background_blur requires RGB input, got {:?}",
                context.format
            )));
        }
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| prepare_err("prepare called before configure"))?;

        // Load model configuration sidecar (or placeholder for unknown models).
        let model_config = load_sidecar_or_placeholder(&config.model)
            .map_err(|e: InferenceError| prepare_err(e.to_string()))?;
        self.model_w = model_config.input_width;
        self.model_h = model_config.input_height;

        // Build engine.  Preserve the structured `InferenceError` so the
        // §27 CLI hint detector can distinguish `ModelNotFound` from
        // `BackendUnavailable` etc. without string-sniffing.
        let engine = OnnxEngine::load(&config.model, model_config.clone()).map_err(
            |e: InferenceError| EffectError::Inference {
                name: Self::NAME.to_string(),
                source: e,
            },
        )?;
        self.engine = Some(Box::new(engine));
        self.model_config = Some(model_config);

        // Allocate scratch.
        self.frame_w = context.width;
        self.frame_h = context.height;
        let frame_pixels = (context.width as usize) * (context.height as usize);
        let frame_bytes = frame_pixels * 3;
        let model_pixels = (self.model_w as usize) * (self.model_h as usize);
        let model_bytes = model_pixels * 3;

        self.blurred = vec![0u8; frame_bytes];
        self.blur_scratch = vec![0u8; frame_bytes];
        self.mask_full = vec![0.0f32; frame_pixels];
        self.mask_prev = vec![0.0f32; frame_pixels];
        self.mask_dilate_scratch = vec![0.0f32; frame_pixels];
        self.mask_feather_scratch = vec![0.0f32; frame_pixels];
        self.model_input_u8 = vec![0u8; model_bytes];
        self.model_input_f32 = vec![0.0f32; model_pixels * 3];
        self.mask_raw = vec![0.0f32; model_pixels];

        self.consecutive_failures = 0;
        info!(
            model = %config.model.display(),
            frame = ?(context.width, context.height),
            model_input = ?(self.model_w, self.model_h),
            "background_blur prepared"
        );
        Ok(())
    }

    /// Apply the background-blur pipeline to `frame` in place.
    ///
    /// # Errors
    ///
    /// * [`EffectError::ProcessFailed`] — frame validation failed
    ///   (non-RGB pixel format, or dimensions differ from those negotiated
    ///   in [`Self::prepare`]).
    /// * [`EffectError::Inference`] — the inference engine returned an
    ///   error for `fallback_threshold` consecutive frames; the underlying
    ///   [`InferenceError`] is preserved in `source`.
    fn process(
        &mut self,
        frame: &mut VideoFrame,
        frame_ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        self.validate_frame(frame)?;

        // Run inference.  On transient failure the frame is passed through
        // unchanged; consecutive failures eventually escalate to an error.
        let output = match self.run_inference(frame) {
            InferenceOutcome::Ok(out) => out,
            InferenceOutcome::Fallback => {
                frame_ctx.fallback_active = true;
                return Ok(());
            }
            InferenceOutcome::Fatal(err) => return Err(err),
        };

        self.build_mask(&output)?;
        self.blend(frame)?;

        debug!(frame_seq = frame.meta.sequence, "background_blur applied");
        Ok(())
    }
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
            // Source is HWC `[h * w * 3]`; output is channel-planar CHW:
            // channel 0 first (pixels floats), then channel 1, then 2.
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
            let class = config.person_class_index.unwrap_or(1);
            if data.len() < expected {
                return Err(process_err("category mask shorter than expected"));
            }
            let class_f = class as f32;
            for (out, &v) in dst.iter_mut().zip(data.iter()) {
                *out = if (v - class_f).abs() < 0.5 { 1.0 } else { 0.0 };
            }
        }
    }
    Ok(())
}

/// Decode a 1- or 2-class probability/logit tensor into a foreground mask.
///
/// The tensor layout (HW / NHWC / NCHW) decides where the bg/fg channels
/// live; never derive this from `shape` magic — older code did and broke
/// on planar NCHW outputs.
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
            // Single-channel mask.
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
            // `[1, H, W, C]` — channels last, interleaved per pixel.
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
            // `[1, C, H, W]` — planar.  Foreground plane starts after the
            // background plane.
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

/// Compute softmax over `(bg, fg)` logits and return the foreground
/// probability.  Numerically stable via the standard `max`-subtraction
/// trick: `e^{x - max}` keeps the intermediate exponents in `(0, 1]` even
/// when the raw logits are large, avoiding overflow.
fn softmax_foreground(background: f32, foreground: f32) -> f32 {
    let max = background.max(foreground);
    let numerator = (foreground - max).exp();
    let denominator = (background - max).exp() + numerator;
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxframe_core::traits::{InferenceInput, InferenceOutput, ModelInfo};
    use std::sync::Arc;

    fn valid_blur_config() -> BlurConfig {
        BlurConfig {
            model: PathBuf::from("/tmp/dummy.onnx"),
            model_config: None,
            blur_radius: defaults::BLUR_RADIUS,
            blur_passes: defaults::BLUR_PASSES,
            mask_threshold: defaults::MASK_THRESHOLD,
            mask_smoothing: defaults::MASK_SMOOTHING,
            mask_feather_radius: defaults::MASK_FEATHER_RADIUS,
            mask_dilate: defaults::MASK_DILATE,
            fallback_threshold: defaults::FALLBACK_THRESHOLD,
        }
    }

    #[test]
    fn config_parses_with_defaults() {
        let raw = r#"
model = "/tmp/dummy.onnx"
"#;
        let parsed: BlurConfig = toml::from_str(raw).expect("parse ok");
        assert_eq!(parsed.blur_radius, defaults::BLUR_RADIUS);
        assert_eq!(parsed.blur_passes, defaults::BLUR_PASSES);
        assert!((parsed.mask_threshold - defaults::MASK_THRESHOLD).abs() < 1e-6);
    }

    #[test]
    fn validate_rejects_missing_model() {
        let mut cfg = valid_blur_config();
        cfg.model = PathBuf::new();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_onnx_extension() {
        let mut cfg = valid_blur_config();
        cfg.model = PathBuf::from("/tmp/model.toml");
        let err = cfg.validate().expect_err("non-onnx must fail");
        assert!(format!("{err}").contains("onnx"));
    }

    #[test]
    fn validate_rejects_no_extension() {
        let mut cfg = valid_blur_config();
        cfg.model = PathBuf::from("/tmp/model");
        let err = cfg.validate().expect_err("no extension must fail");
        assert!(format!("{err}").contains("extension"));
    }

    #[test]
    fn validate_rejects_out_of_range_threshold() {
        let mut cfg = valid_blur_config();
        cfg.mask_threshold = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_excessive_blur_radius() {
        let mut cfg = valid_blur_config();
        cfg.blur_radius = MAX_BLUR_RADIUS + 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_excessive_feather_radius() {
        let mut cfg = valid_blur_config();
        cfg.mask_feather_radius = MAX_FEATHER_RADIUS + 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_fallback_threshold() {
        let mut cfg = valid_blur_config();
        cfg.fallback_threshold = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_passes_with_sane_values() {
        let cfg = valid_blur_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn name_is_stable() {
        let effect = BackgroundBlurEffect::new();
        assert_eq!(effect.name(), "background_blur");
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
        // bg logit much smaller than fg → softmax(fg) close to 1.
        let data = vec![-10.0_f32, 10.0];
        let mut dst = vec![0.0_f32; 1];
        decode_mask(&data, &[1, 1, 1, 2], &mut dst, &cfg).expect("ok");
        assert!(dst[0] > 0.99, "got {}", dst[0]);
    }

    #[test]
    fn decode_mask_handles_nhwc_probabilities() {
        let mut cfg = ModelConfig::new("t", 2, 2);
        cfg.output_type = OutputType::Probabilities;
        cfg.output_layout = OutputLayout::Nhwc;
        // 2×2 pixels, 2 channels (bg, fg), interleaved per pixel:
        // p0: bg=0.2 fg=0.8, p1: bg=0.7 fg=0.3,
        // p2: bg=0.1 fg=0.9, p3: bg=0.5 fg=0.5.
        let data = vec![0.2_f32, 0.8, 0.7, 0.3, 0.1, 0.9, 0.5, 0.5];
        let mut dst = vec![0.0_f32; 4];
        decode_mask(&data, &[1, 2, 2, 2], &mut dst, &cfg).expect("ok");
        // Probabilities path takes the foreground channel verbatim.
        assert!((dst[0] - 0.8).abs() < 1e-6, "p0 got {}", dst[0]);
        assert!((dst[1] - 0.3).abs() < 1e-6, "p1 got {}", dst[1]);
        assert!((dst[2] - 0.9).abs() < 1e-6, "p2 got {}", dst[2]);
        assert!((dst[3] - 0.5).abs() < 1e-6, "p3 got {}", dst[3]);
    }

    #[test]
    fn decode_mask_handles_hw_logits() {
        let mut cfg = ModelConfig::new("t", 2, 2);
        cfg.output_type = OutputType::Logits;
        cfg.output_layout = OutputLayout::Hw;
        // 2×2 single-channel logits.  Sigmoid is monotone, so a higher
        // logit must yield a higher mask value.
        let data = vec![2.0_f32, -2.0, 0.0, 1.0];
        let mut dst = vec![0.0_f32; 4];
        decode_mask(&data, &[2, 2], &mut dst, &cfg).expect("ok");
        // sigmoid(2)  ≈ 0.881, sigmoid(-2) ≈ 0.119,
        // sigmoid(0)  = 0.5,   sigmoid(1)  ≈ 0.731.
        assert!(dst[0] > 0.85, "p0 got {}", dst[0]);
        assert!(dst[1] < 0.15, "p1 got {}", dst[1]);
        assert!((dst[2] - 0.5).abs() < 1e-6, "p2 got {}", dst[2]);
        assert!(dst[3] > dst[2] && dst[3] < dst[0], "p3 got {}", dst[3]);
        // Monotone: larger logit ⇒ larger sigmoid output.
        assert!(dst[1] < dst[2], "monotone: dst[1] !< dst[2]");
        assert!(dst[2] < dst[3], "monotone: dst[2] !< dst[3]");
        assert!(dst[3] < dst[0], "monotone: dst[3] !< dst[0]");
    }

    #[test]
    fn decode_mask_softmaxes_two_channel_logits_nchw() {
        let mut cfg = ModelConfig::new("t", 2, 1);
        cfg.output_type = OutputType::Logits;
        cfg.output_layout = OutputLayout::Nchw;
        // 2 pixels, 2 channels, planar: [bg_p0, bg_p1, fg_p0, fg_p1].
        let data = vec![-10.0_f32, 10.0, 10.0, -10.0];
        let mut dst = vec![0.0_f32; 2];
        decode_mask(&data, &[1, 2, 1, 2], &mut dst, &cfg).expect("ok");
        // Pixel 0: bg=-10, fg=10 → softmax(fg) ≈ 1.
        assert!(dst[0] > 0.99, "p0 got {}", dst[0]);
        // Pixel 1: bg=10, fg=-10 → softmax(fg) ≈ 0.
        assert!(dst[1] < 0.01, "p1 got {}", dst[1]);
    }

    #[test]
    fn pack_input_nhwc_matches_identity() {
        let src = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let mut dst = vec![0.0f32; 12];
        // 2×2 image, scale=1.0, zero=0.0 → identity.
        pack_input(&src, &mut dst, InputLayout::Nhwc, 2, 2, 1.0, 0.0);
        let expected: Vec<f32> = src.iter().map(|&v| f32::from(v)).collect();
        assert_eq!(dst, expected);
    }

    #[test]
    fn pack_input_nchw_transposes_channels() {
        // 2×1 image, RGB pixels: P0=(10, 20, 30), P1=(40, 50, 60).
        let src = vec![10u8, 20, 30, 40, 50, 60];
        let mut dst = vec![0.0f32; 6];
        pack_input(&src, &mut dst, InputLayout::Nchw, 1, 2, 1.0, 0.0);
        // CHW planar: [R0, R1, G0, G1, B0, B1].
        assert_eq!(dst, vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0]);
    }

    #[test]
    fn pack_input_applies_scale_and_zero() {
        let src = vec![10u8, 110, 210];
        let mut dst = vec![0.0f32; 3];
        // 1×1×3, scale=0.01, zero=10 → (v - 10) * 0.01.
        pack_input(&src, &mut dst, InputLayout::Nhwc, 1, 1, 0.01, 10.0);
        assert!((dst[0] - 0.0).abs() < 1e-6);
        assert!((dst[1] - 1.0).abs() < 1e-6);
        assert!((dst[2] - 2.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------
    // Mock-engine happy-path test.
    // -----------------------------------------------------------------

    /// Returns a fixed `InferenceOutput` regardless of input.  Used to
    /// drive [`BackgroundBlurEffect::run_inference`] from a unit test
    /// without loading an ONNX model.
    struct MockEngine {
        output: InferenceOutput,
    }

    impl InferenceEngine for MockEngine {
        fn model_info(&self) -> ModelInfo {
            ModelInfo {
                name: Arc::from("mock"),
                input_width: 2,
                input_height: 2,
                input_format: PixelFormat::Rgb,
            }
        }

        fn infer(&mut self, _input: InferenceInput<'_>) -> Result<InferenceOutput, InferenceError> {
            Ok(self.output.clone())
        }
    }

    #[test]
    fn process_with_mock_engine_blends_known_mask() {
        // Build effect manually (no on-disk model load).
        let mut effect = BackgroundBlurEffect::with_engine(Box::new(MockEngine {
            // 2×2 mask: foreground top-left, background elsewhere.
            output: InferenceOutput {
                data: vec![1.0, 0.0, 0.0, 0.0],
                shape: vec![2, 2],
            },
        }));
        effect.config = Some(valid_blur_config());
        effect.model_config = Some(ModelConfig::new("mock", 2, 2));
        effect.frame_w = 2;
        effect.frame_h = 2;
        effect.model_w = 2;
        effect.model_h = 2;
        let frame_pixels = 4;
        let frame_bytes = frame_pixels * 3;
        let model_pixels = 4;
        effect.blurred = vec![0u8; frame_bytes];
        effect.blur_scratch = vec![0u8; frame_bytes];
        effect.mask_full = vec![0.0; frame_pixels];
        effect.mask_prev = vec![0.0; frame_pixels];
        effect.mask_dilate_scratch = vec![0.0; frame_pixels];
        effect.mask_feather_scratch = vec![0.0; frame_pixels];
        effect.model_input_u8 = vec![0u8; model_pixels * 3];
        effect.model_input_f32 = vec![0.0; model_pixels * 3];
        effect.mask_raw = vec![0.0; model_pixels];

        // Non-uniform 2×2 frame so the box blur actually changes pixel
        // values (uniform input would leave the blurred buffer identical
        // to the original and defeat the `assert_ne!` below).
        let pixels: Vec<u8> = vec![
            200, 50, 50, // top-left   — red
            50, 200, 50, // top-right  — green
            50, 50, 200, // bottom-left — blue
            200, 200, 50, // bottom-right — yellow
        ];
        let meta = fluxframe_core::frame::FrameMeta::default();
        let mut frame = VideoFrame::new_packed(
            fluxframe_core::frame::FrameBuffer::Owned(pixels),
            2,
            2,
            PixelFormat::Rgb,
            meta,
        )
        .expect("packed frame ok");
        let mut frame_ctx = FrameContext::default();

        // Snapshot the input bytes before processing so we can assert
        // the composite actually mutated the frame.
        let original = frame.data.as_slice().to_vec();

        effect
            .process(&mut frame, &mut frame_ctx)
            .expect("process ok");

        // Sanity: process produced *some* output; precise blending depends
        // on the box-blur kernel and mask post-processing, both of which
        // are exercised by their own unit tests.  We only check that the
        // pipeline actually mutated the frame (composite ran end-to-end).
        let out = frame.data.as_slice();
        assert_eq!(out.len(), frame_bytes);
        assert_ne!(out, &original[..], "frame was not modified by composite");
    }
}
