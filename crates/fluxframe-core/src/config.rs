//! Top-level configuration types.
//!
//! These mirror the TOML structure in §13 of the spec.  Defaults are
//! chosen so that a fully empty `[section]` produces a runnable pipeline
//! at the documented 1280x720@30 reference resolution.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize};

use crate::frame::PixelFormat;

// ---------------------------------------------------------------------------
// Defaults & bounds
// ---------------------------------------------------------------------------

/// Default capture device path used when `input.device` is omitted.
pub const DEFAULT_INPUT_DEVICE: &str = "/dev/video0";
/// Default sink device path used when `output.device` is omitted.
pub const DEFAULT_OUTPUT_DEVICE: &str = "/dev/video10";
/// Default frame width in pixels.
pub const DEFAULT_WIDTH: u32 = 1280;
/// Default frame height in pixels.
pub const DEFAULT_HEIGHT: u32 = 720;
/// Default frames-per-second target.
pub const DEFAULT_FPS: u32 = 30;
/// Default maximum end-to-end latency budget in milliseconds.
pub const DEFAULT_MAX_LATENCY_MS: u32 = 120;
/// Default cap on frames in flight through the pipeline.
pub const DEFAULT_MAX_INFLIGHT_FRAMES: u32 = 1;
/// Default tracing log level.
pub const DEFAULT_LOG_LEVEL: &str = "info";

/// Upper bound on frame width accepted by [`FluxConfig::validate`].
pub const MAX_WIDTH: u32 = 16384;
/// Upper bound on frame height accepted by [`FluxConfig::validate`].
pub const MAX_HEIGHT: u32 = 16384;
/// Upper bound on frame rate accepted by [`FluxConfig::validate`].
pub const MAX_FPS: u32 = 240;
/// Upper bound on `max_inflight_frames` — protects §22 bounded-queue invariant.
pub const MAX_INFLIGHT_FRAMES: u32 = 64;
/// Upper bound on `max_latency_ms` accepted by [`FluxConfig::validate`].
pub const MAX_LATENCY_MS: u32 = 10_000;
/// Default cadence of the metrics reporter (`[realtime] metrics_interval_secs`).
/// `0` disables periodic reporting (the teardown summary still runs).
pub const DEFAULT_METRICS_INTERVAL_SECS: u32 = 5;
/// Default downscale factor applied to the output relative to input.
/// `1.0` means "publish at exactly the input resolution"; `0.5` halves
/// each dimension.  Values outside `[MIN_OUTPUT_SCALE, MAX_OUTPUT_SCALE]`
/// are rejected — output is downscale-or-passthrough only, never
/// upscale or stretch.
pub const DEFAULT_OUTPUT_SCALE: f32 = 1.0;
/// Lower inclusive bound on `output.scale`.  Sub-normal and absurdly
/// tiny values (e.g. `1e-30`) would collapse to a 2×2 image while
/// silently consuming a pipeline; `0.05` keeps the smallest sensible
/// output above 64×36 from the reference 1280×720 input.
pub const MIN_OUTPUT_SCALE: f32 = 0.05;
/// Upper inclusive bound on `output.scale`.  Upscaling past the input
/// resolution is not supported: it would inflate bandwidth without
/// adding image information (the ML model has already discarded
/// everything past its own input size).
pub const MAX_OUTPUT_SCALE: f32 = 1.0;

/// Upper bound on positive `metrics_interval_secs` accepted by
/// [`FluxConfig::validate`].  `0` is always valid — it is the
/// documented sentinel that disables the periodic reporter.
/// 1 h is well past any realistic operator setting; the cap exists
/// only to catch typos like `metrics_interval_secs = 50000` that
/// would effectively silence the reporter.
pub const MAX_METRICS_INTERVAL_SECS: u32 = 3600;

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

/// Backend selection for input/output.
///
/// `Auto` lets the pipeline pick the right backend from device path/string.
/// Stage 0 only declares the enum; Stages 1-2 wire the selection logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum BackendKind {
    /// Pick the backend automatically based on the device string.
    #[default]
    Auto,
    /// Linux V4L2 capture/loopback backend.
    V4l2,
    /// Synthetic test-pattern source (for development and CI).
    Testsrc,
    /// Discarding sink (for development and CI).
    Fakesink,
}

// ---------------------------------------------------------------------------
// PixelFormat <-> TOML string adapter
// ---------------------------------------------------------------------------

/// Serde adapter for [`PixelFormat`] using the uppercase tag strings
/// documented in §13 (`"RGB"`, `"RGBA"`, `"BGR"`, `"YUY2"`, `"NV12"`,
/// `"GRAY8"`).
///
/// Kept local to `config.rs` so that the format tag vocabulary is owned by
/// the config layer and `frame::PixelFormat` stays serde-free.
mod pixel_format_serde {
    use super::PixelFormat;
    use serde::{Deserialize, Deserializer, Serializer};

    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "serde calls expect &T signature"
    )]
    pub(super) fn serialize<S: Serializer>(value: &PixelFormat, ser: S) -> Result<S::Ok, S::Error> {
        let tag = match value {
            PixelFormat::Rgb => "RGB",
            PixelFormat::Rgba => "RGBA",
            PixelFormat::Bgr => "BGR",
            PixelFormat::Yuy2 => "YUY2",
            PixelFormat::Nv12 => "NV12",
            PixelFormat::Gray8 => "GRAY8",
        };
        ser.serialize_str(tag)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<PixelFormat, D::Error> {
        let raw = String::deserialize(de)?;
        match raw.to_ascii_uppercase().as_str() {
            "RGB" => Ok(PixelFormat::Rgb),
            "RGBA" => Ok(PixelFormat::Rgba),
            "BGR" => Ok(PixelFormat::Bgr),
            "YUY2" => Ok(PixelFormat::Yuy2),
            "NV12" => Ok(PixelFormat::Nv12),
            "GRAY8" => Ok(PixelFormat::Gray8),
            other => Err(serde::de::Error::custom(format!(
                "unknown pixel format `{other}`; expected one of RGB, RGBA, BGR, YUY2, NV12, GRAY8"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Section: input
// ---------------------------------------------------------------------------

/// Capture-side configuration (`[input]` table).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputConfig {
    /// Capture backend selection.
    #[serde(default)]
    pub backend: BackendKind,
    /// Device path or backend-specific identifier (e.g. `/dev/video0`).
    #[serde(default = "default_input_device")]
    pub device: String,
    /// Requested capture width in pixels.
    #[serde(default = "default_width")]
    pub width: u32,
    /// Requested capture height in pixels.
    #[serde(default = "default_height")]
    pub height: u32,
    /// Requested capture frame rate (frames per second).
    #[serde(default = "default_fps")]
    pub fps: u32,
    /// Requested pixel format negotiated with the capture backend.
    #[serde(
        default = "default_input_format",
        serialize_with = "pixel_format_serde::serialize",
        deserialize_with = "pixel_format_serde::deserialize"
    )]
    pub format: PixelFormat,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            backend: BackendKind::default(),
            device: default_input_device(),
            width: default_width(),
            height: default_height(),
            fps: default_fps(),
            format: default_input_format(),
        }
    }
}

// ---------------------------------------------------------------------------
// Section: output
// ---------------------------------------------------------------------------

/// Output/sink configuration (`[output]` table).
///
/// Output dimensions and frame rate are NOT independent knobs: the
/// supervisor publishes at `input.width × scale × input.height × scale`
/// and at exactly `input.fps`.  This makes aspect-ratio drift and
/// fps drift (videorate frame duplication/drop) impossible by
/// construction — the operator cannot accidentally configure a
/// stretched 1024×768 output from a 1280×720 camera, and cannot
/// silently introduce a 30↔60 fps converter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    /// Output backend selection.  Defaults to `V4l2` (loopback device).
    #[serde(default = "default_output_backend")]
    pub backend: BackendKind,
    /// Sink device path or backend-specific identifier (e.g. `/dev/video10`).
    #[serde(default = "default_output_device")]
    pub device: String,
    /// Downscale factor relative to the input resolution.  Constructed
    /// through [`OutputScale`] (validated at TOML parse / try-from time)
    /// so the runtime never sees an out-of-range value.
    #[serde(default)]
    pub scale: OutputScale,
    /// Pixel format pushed to the sink.  Defaults to `YUY2` for V4L2 loopback.
    #[serde(
        default = "default_output_format",
        serialize_with = "pixel_format_serde::serialize",
        deserialize_with = "pixel_format_serde::deserialize"
    )]
    pub format: PixelFormat,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            backend: default_output_backend(),
            device: default_output_device(),
            scale: OutputScale::default(),
            format: default_output_format(),
        }
    }
}

impl OutputConfig {
    /// Resolve the effective output dimensions for the given input
    /// frame size.  Result is rounded to the nearest even pixel
    /// because chroma-subsampled formats (NV12, YUY2) require even
    /// width/height per plane.  Guaranteed `≥ 2` for any positive
    /// input.
    #[must_use]
    pub fn effective_dimensions(&self, input_w: u32, input_h: u32) -> (u32, u32) {
        let scale = self.scale.value();
        (
            scale_dimension(input_w, scale),
            scale_dimension(input_h, scale),
        )
    }
}

/// Validated downscale factor in `[MIN_OUTPUT_SCALE, MAX_OUTPUT_SCALE]`.
///
/// Constructed via [`OutputScale::new`] / `TryFrom<f32>` / serde — every
/// path runs the same range check, so consumers see only values that
/// have already been screened for `NaN`, `±∞`, `≤ 0`, sub-normal
/// noise, and upscaling.  No `From<f32>` / `Deref` to discourage
/// silent unwrap; reach for [`OutputScale::value`] when the raw `f32`
/// is genuinely needed (e.g. arithmetic).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct OutputScale(f32);

impl OutputScale {
    /// The passthrough factor (`1.0`).  Sink publishes at the input
    /// resolution; `videoscale` short-circuits.
    pub const IDENTITY: Self = Self(DEFAULT_OUTPUT_SCALE);

    /// Build from a raw `f32`, returning a structured config error
    /// when the value falls outside the supported range.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::FluxError::Config`] when the value is
    /// `NaN`, non-finite, `≤ 0`, below [`MIN_OUTPUT_SCALE`], or above
    /// [`MAX_OUTPUT_SCALE`].
    pub fn new(value: f32) -> Result<Self, crate::error::FluxError> {
        if value.is_nan() {
            return Err(config_err(
                "output.scale must be a finite number, got NaN".into(),
            ));
        }
        if !value.is_finite() {
            return Err(config_err(format!(
                "output.scale must be finite, got {value}"
            )));
        }
        if value <= 0.0 {
            return Err(config_err(format!(
                "output.scale must be positive, got {value} (use 1.0 for passthrough, 0.5 for half-size, …)"
            )));
        }
        if value < MIN_OUTPUT_SCALE {
            return Err(config_err(format!(
                "output.scale must be >= {MIN_OUTPUT_SCALE}, got {value} (smaller values collapse to a sub-pixel output)"
            )));
        }
        if value > MAX_OUTPUT_SCALE {
            return Err(config_err(format!(
                "output.scale must be <= {MAX_OUTPUT_SCALE} (upscaling is unsupported), got {value}"
            )));
        }
        Ok(Self(value))
    }

    /// Raw factor — already validated to be in `[MIN_OUTPUT_SCALE,
    /// MAX_OUTPUT_SCALE]`.  `const` so [`OutputScale::IDENTITY`] is
    /// usable in const contexts.
    #[inline]
    #[must_use]
    pub const fn value(self) -> f32 {
        self.0
    }
}

impl Default for OutputScale {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl TryFrom<f32> for OutputScale {
    type Error = crate::error::FluxError;
    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<'de> Deserialize<'de> for OutputScale {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = f32::deserialize(deserializer)?;
        Self::new(raw).map_err(serde::de::Error::custom)
    }
}

/// Apply `scale` to `input` and round to the nearest even pixel.
/// Minimum of 2 so a downscale of a tiny test input still yields a
/// valid chroma-subsampled frame size.
///
/// Caller must pass a `scale` already validated through [`OutputScale`].
fn scale_dimension(input: u32, scale: f32) -> u32 {
    let scaled = (f64::from(input) * f64::from(scale)).round() as u32;
    // Round down to even (chroma subsampling); guarantee ≥ 2.
    ((scaled / 2) * 2).max(2)
}

// ---------------------------------------------------------------------------
// Section: realtime
// ---------------------------------------------------------------------------

/// Realtime / scheduling configuration (`[realtime]` table).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealtimeConfig {
    /// Maximum acceptable end-to-end latency in milliseconds before the
    /// pipeline considers a frame late.
    #[serde(default = "default_max_latency_ms")]
    pub max_latency_ms: u32,
    /// Whether the pipeline is allowed to drop late frames.
    #[serde(default = "default_drop_late")]
    pub drop_late_frames: bool,
    /// Maximum number of frames in flight through the pipeline at once
    /// (§22 bounded-queue invariant).
    #[serde(default = "default_max_inflight")]
    pub max_inflight_frames: u32,
    /// How often the metrics reporter emits a per-window summary, in
    /// seconds.  `0` disables periodic reporting (the teardown summary
    /// is unconditional).
    #[serde(default = "default_metrics_interval_secs")]
    pub metrics_interval_secs: u32,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: default_max_latency_ms(),
            drop_late_frames: default_drop_late(),
            max_inflight_frames: default_max_inflight(),
            metrics_interval_secs: default_metrics_interval_secs(),
        }
    }
}

// ---------------------------------------------------------------------------
// Section: effects
// ---------------------------------------------------------------------------

/// Effect chain plus per-effect parameter tables.
///
/// Per-effect tables are kept as `toml::Value` to avoid baking a closed
/// schema into core; each effect parses its own slice in `configure`.
///
/// `deny_unknown_fields` is intentionally NOT applied here — the flattened
/// `per_effect` map collects every non-`chain` key.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EffectsConfig {
    /// Ordered list of effect names to apply (snake_case internal form).
    #[serde(default)]
    pub chain: Vec<String>,
    /// Per-effect parameter tables, keyed by effect name.
    #[serde(flatten, default)]
    pub per_effect: BTreeMap<String, toml::Value>,
}

// ---------------------------------------------------------------------------
// Section: mask / background / foreground (composite pipeline)
// ---------------------------------------------------------------------------

/// One sub-pipeline section of the composite effect.
///
/// Used for `[mask]`, `[background]`, and `[foreground]` top-level
/// TOML sections. Each section holds a `chain` of effect names plus
/// one sub-table per effect with its parameters; the mask section
/// additionally carries the segmentation model path.
///
/// `deny_unknown_fields` is intentionally NOT applied — the flattened
/// `per_effect` map collects every key that is not one of the
/// reserved control fields. The composite builder rejects unknown
/// effect names against the active registry.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PipelineSection {
    /// Ordered list of effect names to apply within this sub-pipeline.
    #[serde(default)]
    pub chain: Vec<String>,
    /// ONNX model path. Only consumed for the `[mask]` section;
    /// ignored on background/foreground.
    #[serde(default)]
    pub model: Option<PathBuf>,
    /// Optional sidecar `<model>.toml`. Mask section only.
    #[serde(default)]
    pub model_config: Option<PathBuf>,
    /// Consecutive-failure tolerance for the segmentation engine.
    /// Mask section only.
    #[serde(default)]
    pub fallback_threshold: Option<u32>,
    /// Per-effect parameter tables, keyed by effect name. The
    /// composite builder hands each table to the corresponding
    /// effect's `configure`.
    #[serde(flatten, default)]
    pub per_effect: BTreeMap<String, toml::Value>,
}

// ---------------------------------------------------------------------------
// Section: logging
// ---------------------------------------------------------------------------

/// Logging configuration (`[logging]` table).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoggingConfig {
    /// `tracing` log level filter (e.g. `info`, `debug`, `module=trace`).
    #[serde(default = "default_log_level")]
    pub level: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
        }
    }
}

// ---------------------------------------------------------------------------
// Root config
// ---------------------------------------------------------------------------

/// Root configuration object — mirrors the top-level TOML document.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct FluxConfig {
    /// `[input]` section.
    #[serde(default)]
    pub input: InputConfig,
    /// `[output]` section.
    #[serde(default)]
    pub output: OutputConfig,
    /// `[realtime]` section.
    #[serde(default)]
    pub realtime: RealtimeConfig,
    /// `[effects]` section.
    #[serde(default)]
    pub effects: EffectsConfig,
    /// `[mask]` section. Presence activates the composite pipeline:
    /// segmentation → mask chain → background/foreground chains →
    /// alpha composite.
    #[serde(default)]
    pub mask: Option<PipelineSection>,
    /// `[background]` section. Active only when `[mask]` is set; an
    /// absent section means "passthrough" (background equals the
    /// original frame).
    #[serde(default)]
    pub background: Option<PipelineSection>,
    /// `[foreground]` section. Same semantics as `[background]`.
    #[serde(default)]
    pub foreground: Option<PipelineSection>,
    /// `[logging]` section.
    #[serde(default)]
    pub logging: LoggingConfig,
}

impl FluxConfig {
    /// Parse a TOML document into a [`FluxConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::FluxError::Config`] if `text` is not valid
    /// TOML or contains fields that the schema rejects
    /// (`deny_unknown_fields`, unknown pixel format, …).
    pub fn from_toml_str(text: &str) -> Result<Self, crate::error::FluxError> {
        let cfg: Self = toml::from_str(text)?;
        Ok(cfg)
    }

    /// Minimal structural validation.  Deeper checks (model file exists,
    /// device is writable, …) live in `fluxframe check`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::FluxError::Config`] if any field violates the
    /// declared upper/lower bound (zero dimensions, oversized resolution,
    /// excessive fps, unbounded inflight queue, out-of-range latency).
    pub fn validate(&self) -> Result<(), crate::error::FluxError> {
        // §13 `chain` is a reserved key for the chain list, not an effect name.
        // serde's `flatten` cannot enforce this — do it explicitly.
        const RESERVED_EFFECTS_KEYS: &[&str] = &["chain"];

        check_dimensions("input", self.input.width, self.input.height)?;
        check_fps("input", self.input.fps)?;
        // `OutputConfig.scale: OutputScale` is validated at construction
        // (TOML deserialize / TryFrom); no runtime re-check needed.

        if self.realtime.max_inflight_frames == 0
            || self.realtime.max_inflight_frames > MAX_INFLIGHT_FRAMES
        {
            return Err(config_err(format!(
                "realtime.max_inflight_frames must be in 1..={MAX_INFLIGHT_FRAMES}, got {}",
                self.realtime.max_inflight_frames
            )));
        }
        if self.realtime.max_latency_ms == 0 || self.realtime.max_latency_ms > MAX_LATENCY_MS {
            return Err(config_err(format!(
                "realtime.max_latency_ms must be in 1..={MAX_LATENCY_MS}, got {}",
                self.realtime.max_latency_ms
            )));
        }

        // `0` is the documented "disable periodic reporting" sentinel;
        // any positive value must stay within the sane upper bound so
        // `metrics_interval_secs = 50000` does not silently turn into
        // a ~14 h cadence the operator never sees.
        if self.realtime.metrics_interval_secs > MAX_METRICS_INTERVAL_SECS {
            return Err(config_err(format!(
                "realtime.metrics_interval_secs must be 0 (disabled) or in 1..={MAX_METRICS_INTERVAL_SECS}, got {}",
                self.realtime.metrics_interval_secs
            )));
        }

        for key in self.effects.per_effect.keys() {
            if RESERVED_EFFECTS_KEYS.contains(&key.as_str()) {
                return Err(crate::error::FluxError::Config {
                    reason: format!(
                        "effects.{key} is reserved and cannot be used as an effect name"
                    ),
                    hint: Some("rename the effect or use a different key".into()),
                });
            }
        }
        Ok(())
    }
}

/// Build a [`crate::error::FluxError::Config`] from a reason string with
/// no hint attached.
fn config_err(reason: String) -> crate::error::FluxError {
    crate::error::FluxError::Config { reason, hint: None }
}

fn check_dimensions(section: &str, width: u32, height: u32) -> Result<(), crate::error::FluxError> {
    if width == 0 || height == 0 {
        return Err(config_err(format!(
            "{section} width/height must be > 0 (got {width}x{height})"
        )));
    }
    if width > MAX_WIDTH || height > MAX_HEIGHT {
        return Err(config_err(format!(
            "{section} width/height must be <= {MAX_WIDTH}x{MAX_HEIGHT} (got {width}x{height})"
        )));
    }
    Ok(())
}

fn check_fps(section: &str, fps: u32) -> Result<(), crate::error::FluxError> {
    if fps == 0 {
        return Err(config_err(format!("{section}.fps must be > 0")));
    }
    if fps > MAX_FPS {
        return Err(config_err(format!(
            "{section}.fps must be <= {MAX_FPS} (got {fps})"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Default helpers (thin wrappers over module constants for serde `default = `)
// ---------------------------------------------------------------------------

fn default_input_device() -> String {
    DEFAULT_INPUT_DEVICE.to_string()
}
fn default_output_device() -> String {
    DEFAULT_OUTPUT_DEVICE.to_string()
}
fn default_width() -> u32 {
    DEFAULT_WIDTH
}
fn default_height() -> u32 {
    DEFAULT_HEIGHT
}
fn default_fps() -> u32 {
    DEFAULT_FPS
}
fn default_input_format() -> PixelFormat {
    PixelFormat::Rgb
}
fn default_output_format() -> PixelFormat {
    PixelFormat::Yuy2
}
fn default_output_backend() -> BackendKind {
    BackendKind::V4l2
}
fn default_max_latency_ms() -> u32 {
    DEFAULT_MAX_LATENCY_MS
}
fn default_drop_late() -> bool {
    true
}
fn default_max_inflight() -> u32 {
    DEFAULT_MAX_INFLIGHT_FRAMES
}
fn default_metrics_interval_secs() -> u32 {
    DEFAULT_METRICS_INTERVAL_SECS
}
fn default_log_level() -> String {
    DEFAULT_LOG_LEVEL.to_string()
}
