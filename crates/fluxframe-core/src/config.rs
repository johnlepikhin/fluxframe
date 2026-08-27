//! Top-level configuration types.
//!
//! These mirror the TOML structure in §13 of the spec.  Defaults are
//! chosen so that a fully empty `[section]` produces a runnable pipeline
//! at the documented 1280x720@30 reference resolution.

use std::collections::BTreeMap;
use std::fmt;
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
///
/// The tick is emitted at `info` so the operator sees it without
/// touching `RUST_LOG`, which makes the cadence the only lever on log
/// volume: at the previous 5 s default the line accounted for ~99 % of
/// a long-running daemon's log (335 MB over a few weeks, no rotation).
/// 30 s keeps the fps/percentile trend readable while cutting that by
/// ~83 %.  Operators who want the tick out of the way entirely can
/// filter it by target: `level = "fluxframe=info,fluxframe::metrics=debug"`.
///
/// Invariant to preserve when tuning this: the per-stage
/// [`crate::metrics::LatencyHistogram`] rings must hold at least
/// `fps * metrics_interval_secs` samples, otherwise a tick's percentiles
/// describe only the tail of the interval instead of the whole of it.
/// The ring capacity is not a core constant — it is chosen by the
/// runtime-metrics module in the `fluxframe-cli` crate, which owns the
/// histograms; bump it there in step with any large increase here.
pub const DEFAULT_METRICS_INTERVAL_SECS: u32 = 30;
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

/// Upper bound on `input.auto.poll_interval_secs` accepted by
/// [`FluxConfig::validate`].  Larger settings effectively wedge the
/// auto-pick loop (a 1 h cap is well past any reasonable operator
/// value — the cap exists only to catch typos).
pub const MAX_POLL_INTERVAL_SECS: u32 = 3600;

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
// InputDevice <-> TOML string adapter
// ---------------------------------------------------------------------------

/// Typed input-device selector used in [`InputConfig::device`].
///
/// Parsed from the `input.device` TOML string via a thin serde adapter:
///
/// * `"auto"` (case-insensitive) → [`InputDevice::Auto`] — let the
///   supervisor pick the first available V4L2 capture device at
///   startup, polling per `[input.auto]`.
/// * `"testsrc"` → [`InputDevice::Testsrc`] — synthetic test source
///   used for development and CI.
/// * anything else → [`InputDevice::Path`] holding the literal path
///   (e.g. `/dev/video0`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputDevice {
    /// Auto-pick the first available V4L2 capture device at startup;
    /// `[input.auto]` polling knobs apply.
    Auto,
    /// Synthetic test source.
    Testsrc,
    /// Explicit device path (e.g. `/dev/video0`).
    Path(PathBuf),
}

impl fmt::Display for InputDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputDevice::Auto => f.write_str("auto"),
            InputDevice::Testsrc => f.write_str("testsrc"),
            InputDevice::Path(p) => write!(f, "{}", p.display()),
        }
    }
}

/// Serde adapter for [`InputDevice`] — string in/out, matching the
/// existing TOML shape (`device = "auto" | "testsrc" | "/dev/videoN"`).
///
/// Kept local to `config.rs` for the same reason as
/// [`pixel_format_serde`]: the tag vocabulary belongs to the config
/// layer, not to the type itself.
mod input_device_serde {
    use super::InputDevice;
    use serde::{Deserialize, Deserializer, Serializer};
    use std::path::PathBuf;

    pub(super) fn serialize<S: Serializer>(value: &InputDevice, ser: S) -> Result<S::Ok, S::Error> {
        // Reuse Display so the string form is one source of truth.
        ser.collect_str(value)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<InputDevice, D::Error> {
        let raw = String::deserialize(de)?;
        Ok(match raw.to_ascii_lowercase().as_str() {
            "auto" => InputDevice::Auto,
            "testsrc" => InputDevice::Testsrc,
            _ => InputDevice::Path(PathBuf::from(raw)),
        })
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
    /// Capture-device selector — see [`InputDevice`] for the parsed
    /// form. The TOML string `"auto"` (case-insensitive) maps to
    /// [`InputDevice::Auto`] and activates the `[input.auto]` polling
    /// loop; `"testsrc"` selects the synthetic source; anything else
    /// is taken as a literal device path (e.g. `/dev/video0`).
    #[serde(
        default = "default_input_device",
        serialize_with = "input_device_serde::serialize",
        deserialize_with = "input_device_serde::deserialize"
    )]
    pub device: InputDevice,
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
    /// Knobs for `device = "auto"` mode. Ignored when `device` is an
    /// explicit path or backend name.
    #[serde(default)]
    pub auto: AutoInputConfig,
    /// Base delay (milliseconds) for the input-acquire exponential
    /// backoff. When `idle.enabled`, the supervisor keeps the
    /// v4l2loopback output streaming a placeholder and retries acquiring
    /// the camera with backoff starting at this value (doubling up to
    /// `acquire_backoff_max_ms`). Only device-contention errors
    /// (busy/absent) are retried; permanent errors fail fast.
    #[serde(default = "default_acquire_backoff_base_ms")]
    pub acquire_backoff_base_ms: u32,
    /// Ceiling (milliseconds) for the input-acquire backoff. Must be
    /// `>= acquire_backoff_base_ms`.
    #[serde(default = "default_acquire_backoff_max_ms")]
    pub acquire_backoff_max_ms: u32,
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
            auto: AutoInputConfig::default(),
            acquire_backoff_base_ms: default_acquire_backoff_base_ms(),
            acquire_backoff_max_ms: default_acquire_backoff_max_ms(),
        }
    }
}

/// `[input.auto]` — runtime parameters for auto-picking the input
/// V4L2 device. All fields have defaults so an empty / missing
/// `[input.auto]` table behaves like a sensible default. The
/// operator opts out simply by setting `input.device` to anything
/// other than `"auto"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutoInputConfig {
    /// Polling cadence in seconds while waiting for any V4L2 capture
    /// device to appear. Validated by [`FluxConfig::validate`] to be
    /// in `1..=MAX_POLL_INTERVAL_SECS`.
    #[serde(default = "default_auto_poll_interval_secs")]
    pub poll_interval_secs: u32,
    /// Extra device paths to skip when scanning, on top of the
    /// supervisor's automatic exclusion of the output device.
    #[serde(default)]
    pub exclude_devices: Vec<PathBuf>,
}

impl Default for AutoInputConfig {
    fn default() -> Self {
        Self {
            poll_interval_secs: default_auto_poll_interval_secs(),
            exclude_devices: Vec::new(),
        }
    }
}

fn default_auto_poll_interval_secs() -> u32 {
    2
}
fn default_acquire_backoff_base_ms() -> u32 {
    500
}
fn default_acquire_backoff_max_ms() -> u32 {
    5000
}

/// Upper inclusive bound on `input.acquire_backoff_max_ms` — a busy
/// camera should be re-tried at least every 30 s; beyond that the
/// recovery latency feels broken.
pub const MAX_ACQUIRE_BACKOFF_MS: u32 = 30_000;

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
// Section: pipeline sub-section (composite building block)
// ---------------------------------------------------------------------------

/// One sub-pipeline section of the composite effect.
///
/// Used for the `mask`, `background`, and `foreground` sub-sections
/// of a named preset (`[presets.NAME.mask]`, …). Each section holds a
/// `chain` of effect names plus one sub-table per effect with its
/// parameters; the mask section additionally carries the segmentation
/// model path.
///
/// `deny_unknown_fields` is intentionally NOT applied — the flattened
/// `per_effect` map collects every key that is not one of the
/// reserved control fields. The composite builder rejects unknown
/// effect names against the active registry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
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
// Section: presets (named composite pipelines)
// ---------------------------------------------------------------------------

/// One named composite pipeline. Each preset describes up to three
/// sub-pipelines (`mask`, `background`, `foreground`), all optional.
///
/// Selection rules (resolved by the CLI, not by serde):
///
/// * `mask` absent — no composite is built; the frame passes through
///   the pipeline unchanged.
/// * `mask` present, `background`/`foreground` absent — the
///   corresponding plane chain is empty (the un-composited plane is
///   used as-is in the alpha composite).
///
/// `deny_unknown_fields` keeps typos in the section name surfaced
/// loudly. The inner [`PipelineSection`] cannot use it because of
/// `#[serde(flatten)]` on `per_effect`, but the wrapper has no flatten.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Preset {
    /// Mask sub-pipeline: segmentation model plus the chain of
    /// `MaskEffect`s post-processing the confidence mask.
    #[serde(default)]
    pub mask: Option<PipelineSection>,
    /// Background plane sub-pipeline (applied to the background half
    /// of the alpha composite).
    #[serde(default)]
    pub background: Option<PipelineSection>,
    /// Foreground plane sub-pipeline (applied to the foreground half
    /// of the alpha composite).
    #[serde(default)]
    pub foreground: Option<PipelineSection>,
    /// Post-composite sub-pipeline of mask-aware frame-level effects.
    /// Runs inside `CompositeEffect` immediately after
    /// `alpha_composite_rgb_in_place`. Requires `mask` to be present
    /// — otherwise validation fails (the post chain has nothing to
    /// look at).
    #[serde(default)]
    pub post: Option<PipelineSection>,
}

// ---------------------------------------------------------------------------
// Section: control (live-reconfig socket)
// ---------------------------------------------------------------------------

/// Control-socket configuration (`[control]` table). Drives the UNIX
/// socket used for live reconfiguration (Stage 13). Disabled by
/// default — operators must opt in explicitly because the socket is a
/// new (local-only) surface.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlConfig {
    /// Enable the listener. Default `false`.
    #[serde(default)]
    pub enabled: bool,
    /// Override the socket path. `None` (default) derives the path
    /// from `$XDG_RUNTIME_DIR/fluxframe.sock`, falling back to
    /// `/tmp/fluxframe.sock` when `$XDG_RUNTIME_DIR` is unset. Mode
    /// `0600` is applied race-free (bind into a sibling temp path,
    /// chmod, then atomic rename), so other local users on a shared
    /// host cannot connect. Multi-user hosts that share `/tmp`
    /// should set this explicitly — e.g.
    /// `socket_path = "/run/user/$UID/fluxframe.sock"`.
    #[serde(default)]
    pub socket_path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Section: idle (consumer-aware lifecycle)
// ---------------------------------------------------------------------------

/// Placeholder kind for the idle frame. `"color"` paints a solid RGB
/// fill, `"image"` loads a static image file from `placeholder_path`.
///
/// Lowercase serde tag keeps the TOML form ergonomic and matches the
/// surrounding conventions (`pixel_format`, `backend`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum IdlePlaceholderKind {
    /// Solid RGB fill (`placeholder_rgb`).
    #[default]
    Color,
    /// Static image loaded from `placeholder_path`.
    Image,
}

/// Idle-mode configuration (`[idle]` table).
///
/// Stage 15 introduces consumer-aware lifecycle management for the
/// v4l2loopback sink: when no reader is attached to the output device,
/// the supervisor tears down the input pipeline and publishes a cheap
/// placeholder at `fps` instead of running the full effect chain. After
/// `deep_idle_secs` of continued idleness the ONNX session is also
/// dropped from memory, reclaiming ~150 MB.
///
/// Stage 15 ships with `enabled = false` so existing deployments are
/// unaffected by the upgrade — operators opt in explicitly. A future
/// stage may flip the default once the feature has time in the field.
///
/// `deny_unknown_fields` keeps typos loud.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdleConfig {
    /// Master switch. `false` (default in Stage 15) keeps Stage 14
    /// behaviour exactly: no detector thread, no state machine. The
    /// detector additionally refuses to spawn when the output sink is
    /// not a `/dev/videoN` (V4L2) device.
    #[serde(default)]
    pub enabled: bool,

    /// Placeholder kind — `Color` paints a solid fill, `Image` loads
    /// a static file. Defaults to `Color` because a config without
    /// `placeholder_path` should still produce a working placeholder.
    #[serde(default)]
    pub placeholder: IdlePlaceholderKind,

    /// RGB triplet for [`IdlePlaceholderKind::Color`]. Each component is
    /// `0..=255`. Defaults to a neutral dark grey that is unambiguously
    /// "not live video" without being attention-grabbing.
    #[serde(default = "default_placeholder_rgb")]
    pub placeholder_rgb: [u8; 3],

    /// Path to a static image file for [`IdlePlaceholderKind::Image`].
    /// Ignored when `placeholder = "color"`. Currently passed verbatim
    /// to [`image::ImageReader::open`] — operators should use an
    /// absolute path. Stage 15 Step 4 (supervisor wiring) will resolve
    /// relative paths against the config-file directory.
    #[serde(default)]
    pub placeholder_path: Option<PathBuf>,

    /// Frame rate (frames per second) during idle. 1 Hz is enough to
    /// keep v4l2loopback's ring buffer fresh; higher values cost CPU
    /// without observable consumer benefit.
    #[serde(default = "default_idle_fps")]
    pub fps: u32,

    /// Minimum placeholder cadence (fps) used while idle so the
    /// v4l2loopback device keeps advertising CAPTURE caps and stays
    /// enumerable by capability-filtering consumers (Chrome/WebRTC).
    /// This is a heartbeat for device VISIBILITY, independent of the
    /// camera: it is NOT bounded by the input fps (no real camera
    /// frames flow while idle). The effective idle cadence is
    /// `max(idle.fps, idle.min_visibility_fps)`. A bare 1 Hz placeholder
    /// is too sparse for Chrome to reliably enumerate/retain the node.
    #[serde(default = "default_idle_min_visibility_fps")]
    pub min_visibility_fps: u32,

    /// Seconds of "no consumer" observed before the supervisor flips
    /// from Active to Idle. The 5 s default absorbs the usual reopen
    /// storm from Zoom/OBS startup without flapping the camera LED.
    #[serde(default = "default_idle_teardown_secs")]
    pub teardown_secs: u32,

    /// Deprecated and ignored since Stage 16: the `DeepIdle` state was
    /// removed (it was a no-op that never unloaded ONNX and could wedge
    /// the daemon). The field is still accepted so existing TOML keeps
    /// loading under `deny_unknown_fields`; its value has no effect.
    #[serde(default = "default_idle_deep_secs")]
    pub deep_idle_secs: u32,

    /// Fallback-path poll cadence in milliseconds. Lower = faster wake
    /// on consumer reconnect; higher = cheaper steady-state. The 250 ms
    /// default hits a ≤ 500 ms wake budget from Idle.
    ///
    /// Only the `/proc`-polling fallback uses this. The kernel-event
    /// source is edge-driven and the inotify source blocks on its own
    /// fd, so neither has a poll cadence to tune.
    #[serde(default = "default_idle_poll_interval_ms")]
    pub poll_interval_ms: u32,

    /// How often the detector re-reads the driver's capture-usage value
    /// instead of waiting for the next event, in seconds. `0` disables
    /// the re-read entirely.
    ///
    /// The kernel client-usage subscription is edge-triggered: a single
    /// event the driver never queues — or never delivers — latches the
    /// verdict for the rest of the run, and the camera stays down until
    /// the daemon is restarted. This interval bounds that failure to one
    /// period.
    ///
    /// Only the kernel-event source honours it; the `inotify` and
    /// `/proc` paths cannot use the re-read (its `open`/`close` would be
    /// mistaken for a consumer by their own open-balance heuristic).
    ///
    /// Read once, when the detector starts: changing it needs a daemon
    /// restart, not a `reload`.
    #[serde(default = "default_idle_resync_interval_secs")]
    pub resync_interval_secs: u32,

    /// Which mechanism decides whether a consumer is attached.
    ///
    /// The operator-facing kill switch for the detector. Unlike
    /// `enabled = false` — which also removes the placeholder and with
    /// it the loopback's CAPTURE-caps heartbeat — this only changes
    /// *how* presence is determined, so a misbehaving detector can be
    /// swapped out without giving up device visibility.
    #[serde(default)]
    pub presence_source: IdlePresenceSource,
}

/// Mechanism used to detect whether anything is consuming the loopback.
///
/// Defaults to [`IdlePresenceSource::Auto`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdlePresenceSource {
    /// Kernel `V4L2_EVENT_PRI_CLIENT_USAGE` events, falling back to the
    /// `inotify` + `/proc` heuristic when the driver does not support
    /// them (v4l2loopback < 0.13, or a non-loopback sink).
    #[default]
    Auto,
    /// Kernel events only. If the subscription fails the detector
    /// reports `Unknown` — which the state machine treats as "consumer
    /// present", so idle never engages — rather than silently degrading
    /// to the heuristic. For diagnosing whether a fallback is masking a
    /// problem.
    KernelEvent,
    /// Force the pre-0.5 `inotify` + `/proc` heuristic. Cannot produce
    /// an authoritative "nobody is attached" when consumers run as
    /// another user (see the detector's module docs), so idle may fail
    /// to engage. Kept as an escape hatch.
    Inotify,
    /// Never report "absent": idle is never entered, the camera is held
    /// for the whole run. The placeholder machinery and the Stage-16
    /// loopback-visibility guarantees stay intact — this disables only
    /// the power saving, not the output.
    Disabled,
}

impl Default for IdleConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            placeholder: IdlePlaceholderKind::default(),
            placeholder_rgb: default_placeholder_rgb(),
            placeholder_path: None,
            fps: default_idle_fps(),
            min_visibility_fps: default_idle_min_visibility_fps(),
            teardown_secs: default_idle_teardown_secs(),
            deep_idle_secs: default_idle_deep_secs(),
            poll_interval_ms: default_idle_poll_interval_ms(),
            resync_interval_secs: default_idle_resync_interval_secs(),
            presence_source: IdlePresenceSource::default(),
        }
    }
}

impl IdleConfig {
    /// `true` when the configuration is equivalent to "idle disabled".
    ///
    /// Used by the serde adapter on [`FluxConfig`] to skip emitting an
    /// `[idle]` table when the user has not opted in. This keeps
    /// `current_config` JSON byte-identical to the Stage 14 shape for
    /// configs that never touch idle mode.
    #[must_use]
    pub fn is_off(&self) -> bool {
        !self.enabled
    }
}

/// Upper bound on `idle.teardown_secs` — 1 h is well past any
/// realistic operator setting; the cap exists to catch typos.
pub const MAX_IDLE_TEARDOWN_SECS: u32 = 3600;

/// Upper bound on `idle.deep_idle_secs` — same rationale as
/// [`MAX_IDLE_TEARDOWN_SECS`]; deep-idle thresholds beyond an hour
/// negate the memory-reclaim payoff.
pub const MAX_IDLE_DEEP_SECS: u32 = 3600;

/// Lower inclusive bound on `idle.poll_interval_ms`. Below ~100 ms
/// the sysfs poller starts to show on `top`; the wake budget gains
/// nothing because the kernel only flips `state` on reader STREAMON.
pub const MIN_IDLE_POLL_INTERVAL_MS: u32 = 100;

/// Upper inclusive bound on `idle.poll_interval_ms`. Past 5 s the
/// wake latency dominates the 5 s cooldown and the user-visible
/// reconnect feels broken.
pub const MAX_IDLE_POLL_INTERVAL_MS: u32 = 5000;

/// Upper inclusive bound on `idle.fps`. v4l2loopback does not benefit
/// from placeholder fps above the input fps; capping at the input fps
/// upper bound matches operator expectations.
pub const MAX_IDLE_FPS: u32 = 60;

/// Lower inclusive bound on a non-zero `idle.resync_interval_secs`.
///
/// Each resync opens the loopback node for the duration of one ioctl.
/// v4l2loopback caps concurrent openers (`max_openers`, 10 by default)
/// and a real consumer that opens the device inside that window gets
/// `EBUSY` — so a mistyped `1` would turn the diagnostic into the very
/// failure it is meant to detect. Five seconds is well below any wake
/// budget an operator cares about and still leaves the node free
/// essentially all of the time.
pub const MIN_IDLE_RESYNC_INTERVAL_SECS: u32 = 5;

/// Upper inclusive bound on `idle.resync_interval_secs` — same
/// typo-catching rationale as [`MAX_IDLE_TEARDOWN_SECS`]. Past an hour
/// the safety net is slower than an operator noticing the black frame.
pub const MAX_IDLE_RESYNC_INTERVAL_SECS: u32 = 3600;

fn default_placeholder_rgb() -> [u8; 3] {
    [16, 16, 16]
}
fn default_idle_fps() -> u32 {
    1
}
fn default_idle_min_visibility_fps() -> u32 {
    10
}
fn default_idle_teardown_secs() -> u32 {
    5
}
fn default_idle_deep_secs() -> u32 {
    30
}
fn default_idle_poll_interval_ms() -> u32 {
    250
}
fn default_idle_resync_interval_secs() -> u32 {
    30
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
    /// Named composite pipelines. The CLI selects one by name via
    /// `--preset NAME`; falling back to the preset named `"default"`
    /// when no flag is given. Empty map is valid at parse time but
    /// causes a runtime error at preset-resolution if `--preset` is
    /// invoked (or if no `default` exists for the implicit selection).
    #[serde(default)]
    pub presets: BTreeMap<String, Preset>,
    /// `[logging]` section.
    #[serde(default)]
    pub logging: LoggingConfig,
    /// `[control]` section. Enables the UNIX-socket live-reconfig
    /// surface (Stage 13). Disabled by default — opt-in feature.
    #[serde(default)]
    pub control: ControlConfig,
    /// `[idle]` section. Stage 15 consumer-aware lifecycle. Disabled
    /// by default — operators opt in. The `skip_serializing_if` arm
    /// keeps `current_config` JSON byte-identical to Stage 14 when
    /// idle is off, so existing GUI clients are unaffected.
    #[serde(default, skip_serializing_if = "IdleConfig::is_off")]
    pub idle: IdleConfig,
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

        // Auto-pick poll cadence: `0` would busy-loop on the device
        // scanner; absurdly large values silently disable the polling
        // loop the operator opted into.
        if self.input.auto.poll_interval_secs == 0
            || self.input.auto.poll_interval_secs > MAX_POLL_INTERVAL_SECS
        {
            return Err(config_err(format!(
                "input.auto.poll_interval_secs must be in 1..={MAX_POLL_INTERVAL_SECS}, got {}",
                self.input.auto.poll_interval_secs
            )));
        }

        // Input-acquire backoff bounds. `base == 0` would busy-spin on a
        // contended camera; `max < base` is incoherent; an oversized cap
        // makes recovery feel broken.
        if self.input.acquire_backoff_base_ms == 0 {
            return Err(config_err(
                "input.acquire_backoff_base_ms must be > 0, got 0".into(),
            ));
        }
        if self.input.acquire_backoff_max_ms > MAX_ACQUIRE_BACKOFF_MS {
            return Err(config_err(format!(
                "input.acquire_backoff_max_ms must be <= {MAX_ACQUIRE_BACKOFF_MS}, got {}",
                self.input.acquire_backoff_max_ms
            )));
        }
        if self.input.acquire_backoff_max_ms < self.input.acquire_backoff_base_ms {
            return Err(config_err(format!(
                "input.acquire_backoff_max_ms ({}) must be >= input.acquire_backoff_base_ms ({})",
                self.input.acquire_backoff_max_ms, self.input.acquire_backoff_base_ms
            )));
        }

        // Idle-mode bounds. Validated even when `enabled = false` so a
        // future toggle does not surface a stale invalid value at
        // runtime — fail fast at config-load.
        if self.idle.fps == 0 || self.idle.fps > MAX_IDLE_FPS {
            return Err(config_err(format!(
                "idle.fps must be in 1..={MAX_IDLE_FPS}, got {}",
                self.idle.fps
            )));
        }
        // Visibility heartbeat. NOT bounded by input.fps — it streams a
        // static placeholder, not camera frames, so the loopback stays
        // enumerable by Chrome/WebRTC even with no/low-fps camera.
        if self.idle.min_visibility_fps == 0 || self.idle.min_visibility_fps > MAX_IDLE_FPS {
            return Err(config_err(format!(
                "idle.min_visibility_fps must be in 1..={MAX_IDLE_FPS}, got {}",
                self.idle.min_visibility_fps
            )));
        }
        if self.idle.teardown_secs == 0 || self.idle.teardown_secs > MAX_IDLE_TEARDOWN_SECS {
            return Err(config_err(format!(
                "idle.teardown_secs must be in 1..={MAX_IDLE_TEARDOWN_SECS}, got {}",
                self.idle.teardown_secs
            )));
        }
        if self.idle.deep_idle_secs == 0 {
            return Err(config_err("idle.deep_idle_secs must be > 0, got 0".into()));
        }
        if self.idle.deep_idle_secs > MAX_IDLE_DEEP_SECS {
            return Err(config_err(format!(
                "idle.deep_idle_secs must be <= {MAX_IDLE_DEEP_SECS}, got {}",
                self.idle.deep_idle_secs
            )));
        }
        // DeepIdle state removed (Stage 16); deep_idle_secs kept for TOML
        // back-compat, the deep > teardown cross-check is dropped.
        if self.idle.poll_interval_ms < MIN_IDLE_POLL_INTERVAL_MS
            || self.idle.poll_interval_ms > MAX_IDLE_POLL_INTERVAL_MS
        {
            return Err(config_err(format!(
                "idle.poll_interval_ms must be in {MIN_IDLE_POLL_INTERVAL_MS}..={MAX_IDLE_POLL_INTERVAL_MS}, got {}",
                self.idle.poll_interval_ms
            )));
        }
        // `0` is a deliberate value here — "never re-read" — so it is
        // excluded from the range rather than rejected with it.
        if self.idle.resync_interval_secs != 0
            && (self.idle.resync_interval_secs < MIN_IDLE_RESYNC_INTERVAL_SECS
                || self.idle.resync_interval_secs > MAX_IDLE_RESYNC_INTERVAL_SECS)
        {
            return Err(config_err(format!(
                "idle.resync_interval_secs must be 0 (disabled) or in \
                 {MIN_IDLE_RESYNC_INTERVAL_SECS}..={MAX_IDLE_RESYNC_INTERVAL_SECS}, got {}",
                self.idle.resync_interval_secs
            )));
        }
        if self.idle.placeholder == IdlePlaceholderKind::Image
            && self.idle.placeholder_path.is_none()
        {
            return Err(config_err(
                "idle.placeholder = \"image\" requires idle.placeholder_path".into(),
            ));
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

fn default_input_device() -> InputDevice {
    InputDevice::Path(PathBuf::from(DEFAULT_INPUT_DEVICE))
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

#[cfg(test)]
mod tests {
    use super::{
        FluxConfig, MAX_IDLE_RESYNC_INTERVAL_SECS, MIN_IDLE_RESYNC_INTERVAL_SECS,
        default_idle_resync_interval_secs,
    };

    fn cfg_with_resync(value: &str) -> Result<FluxConfig, crate::error::FluxError> {
        let text = format!("[idle]\nenabled = true\nresync_interval_secs = {value}\n");
        FluxConfig::from_toml_str(&text)
    }

    #[test]
    fn resync_interval_defaults_when_absent() {
        let cfg = FluxConfig::from_toml_str("[idle]\nenabled = true\n").expect("parse");
        assert_eq!(
            cfg.idle.resync_interval_secs,
            default_idle_resync_interval_secs()
        );
        cfg.validate().expect("default must validate");
    }

    #[test]
    fn resync_interval_zero_is_accepted_as_disabled() {
        let cfg = cfg_with_resync("0").expect("parse");
        cfg.validate()
            .expect("0 means 'never re-read', not an out-of-range value");
    }

    #[test]
    fn resync_interval_below_the_floor_is_rejected() {
        // A mistyped `1` would open the loopback every second and starve
        // `max_openers`; the floor is what stops that reaching a run.
        let cfg = cfg_with_resync(&(MIN_IDLE_RESYNC_INTERVAL_SECS - 1).to_string()).expect("parse");
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn resync_interval_above_the_ceiling_is_rejected() {
        let cfg = cfg_with_resync(&(MAX_IDLE_RESYNC_INTERVAL_SECS + 1).to_string()).expect("parse");
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn resync_interval_at_the_bounds_is_accepted() {
        for value in [MIN_IDLE_RESYNC_INTERVAL_SECS, MAX_IDLE_RESYNC_INTERVAL_SECS] {
            let cfg = cfg_with_resync(&value.to_string()).expect("parse");
            cfg.validate().unwrap_or_else(|e| panic!("{value}: {e}"));
        }
    }
}
