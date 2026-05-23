//! Top-level configuration types.
//!
//! These mirror the TOML structure in §13 of the spec.  Defaults are
//! chosen so that a fully empty `[section]` produces a runnable pipeline
//! at the documented 1280x720@30 reference resolution.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    /// Output backend selection.  Defaults to `V4l2` (loopback device).
    #[serde(default = "default_output_backend")]
    pub backend: BackendKind,
    /// Sink device path or backend-specific identifier (e.g. `/dev/video10`).
    #[serde(default = "default_output_device")]
    pub device: String,
    /// Output frame width in pixels.
    #[serde(default = "default_width")]
    pub width: u32,
    /// Output frame height in pixels.
    #[serde(default = "default_height")]
    pub height: u32,
    /// Output frame rate (frames per second).
    #[serde(default = "default_fps")]
    pub fps: u32,
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
            width: default_width(),
            height: default_height(),
            fps: default_fps(),
            format: default_output_format(),
        }
    }
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
        check_dimensions("output", self.output.width, self.output.height)?;
        check_fps("input", self.input.fps)?;
        check_fps("output", self.output.fps)?;

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
