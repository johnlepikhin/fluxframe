//! Error model for FluxFrame.
//!
//! Errors live in three families, matching the architectural layers:
//!
//! * `EffectError`    — anything raised from `VideoEffect::process`.
//! * `InferenceError` — anything raised from `InferenceEngine::infer`.
//! * `PipelineError`  — capture/output/transport problems.
//!
//! `FluxError` is the top-level enum used at the CLI boundary and groups
//! all of the above plus configuration and I/O.  Concrete diagnostics
//! (device path, plugin name, hint text) belong in the variant payloads
//! so the §27 user-facing error format ("Error / Reason / Hint") can be
//! produced without string parsing.

use std::path::PathBuf;

use thiserror::Error;

/// Errors raised from a `VideoEffect` implementation.
///
/// These cover the full lifecycle of an effect: configuration validation,
/// one-time preparation (model loading, GPU buffer allocation, etc.) and
/// per-frame processing.  Inference failures inside an effect are wrapped
/// in [`EffectError::Inference`] so callers can still inspect the
/// underlying [`InferenceError`] via `source()`.
#[derive(Debug, Error)]
#[non_exhaustive]
#[expect(
    missing_docs,
    reason = "variant fields are exposed via #[error(...)] Display formatting"
)]
pub enum EffectError {
    /// The effect's configuration payload (TOML section) failed validation.
    ///
    /// Raised before any frame is processed, typically from
    /// `VideoEffect::configure`.
    #[error("effect '{name}' configuration is invalid: {reason}")]
    InvalidConfig { name: String, reason: String },

    /// The effect failed during its one-time preparation step.
    ///
    /// Typical causes: model file missing, GPU context not available,
    /// shader compilation failure.
    #[error("effect '{name}' failed to prepare: {reason}")]
    PrepareFailed { name: String, reason: String },

    /// The effect failed while processing a frame.
    ///
    /// Indicates a runtime error: unexpected pixel format, allocation
    /// failure, or a numerical error inside the algorithm.
    #[error("effect '{name}' failed while processing frame: {reason}")]
    ProcessFailed { name: String, reason: String },

    /// The effect delegated to an inference engine, which then failed.
    ///
    /// The underlying [`InferenceError`] is preserved as `#[source]` so
    /// the full causal chain is visible in error reports.
    #[error("inference error inside effect '{name}': {source}")]
    Inference {
        name: String,
        #[source]
        source: InferenceError,
    },
}

/// Errors raised from an `InferenceEngine` implementation.
///
/// Covers model lifecycle (load, configure) and per-call execution.
/// These can be triggered standalone or wrapped inside
/// [`EffectError::Inference`] when an effect drives the engine.
#[derive(Debug, Error)]
#[non_exhaustive]
#[expect(
    missing_docs,
    reason = "variant fields are exposed via #[error(...)] Display formatting"
)]
pub enum InferenceError {
    /// The model file path does not exist or is not readable.
    #[error("model file not found: {path}")]
    ModelNotFound { path: String },

    /// The model file exists but could not be parsed/loaded by the backend.
    #[error("model could not be loaded: {reason}")]
    ModelLoadFailed { reason: String },

    /// The model's configuration (input/output shapes, providers) is invalid.
    #[error("model config invalid: {reason}")]
    InvalidModelConfig { reason: String },

    /// The chosen backend (e.g. CUDA, CoreML, ONNX Runtime) is not available.
    ///
    /// Carries a user-facing hint suggesting how to install or enable it.
    #[error("inference backend unavailable: {reason}")]
    BackendUnavailable { reason: String, hint: String },

    /// The model loaded successfully but a specific `infer` call failed.
    #[error("inference failed: {reason}")]
    InferenceFailed { reason: String },
}

/// Errors raised from the capture/output transport (GStreamer) layer.
///
/// These surface from pipeline construction, negotiation and runtime.
/// Each variant carries enough structured context (device name, element
/// name, hint) to render the §27 "Error / Reason / Hint" format.
#[derive(Debug, Error)]
#[non_exhaustive]
#[expect(
    missing_docs,
    reason = "variant fields are exposed via #[error(...)] Display formatting"
)]
pub enum PipelineError {
    /// A required GStreamer element is not registered.
    ///
    /// `hint` names the package that typically ships it
    /// (e.g. `gstreamer1.0-plugins-good`).
    #[error("required GStreamer element '{element}' is missing")]
    MissingElement { element: String, hint: String },

    /// The input device (camera, file, virtual source) could not be opened.
    #[error("input device '{device}' could not be opened: {reason}")]
    InputDeviceUnavailable {
        device: String,
        reason: String,
        hint: String,
    },

    /// The output device (v4l2loopback, file, network sink) is not usable.
    #[error("output device '{device}' is not usable: {reason}")]
    OutputDeviceUnavailable {
        device: String,
        reason: String,
        hint: String,
    },

    /// A stage in the pipeline received a pixel format it cannot handle.
    ///
    /// At least one of `format`/`raw_label` is `Some`.  Use `format` when
    /// the source backend produced a recognised FluxFrame variant; use
    /// `raw_label` for backend-specific identifiers that don't map to
    /// `PixelFormat` (e.g. raw GStreamer caps strings).
    #[error("pixel format not supported (format={format:?}, raw_label={raw_label:?})")]
    UnsupportedPixelFormat {
        /// Recognised FluxFrame variant when the backend produced one.
        format: Option<crate::frame::PixelFormat>,
        /// Backend-specific identifier when the variant could not be mapped.
        raw_label: Option<String>,
    },

    /// Upstream and downstream caps could not be reconciled.
    #[error("caps negotiation failed: {reason}")]
    CapsNegotiationFailed { reason: String },

    /// A GStreamer state change (NULL → READY → PAUSED → PLAYING) failed.
    #[error("pipeline state change failed: {reason}")]
    StateChangeFailed { reason: String },

    /// A GStreamer bus reported a fatal error from a pipeline element.
    ///
    /// Carries the GStreamer-level message string and optional debug string
    /// instead of the raw `glib::Error`, so `fluxframe-core` stays free of
    /// the `glib` dependency.  The string content is whatever GStreamer
    /// produced — sufficient for §27 diagnostics and logs.
    #[error("pipeline bus error from {element}: {message}")]
    BusError {
        /// Pipeline element name that originated the error (best-effort).
        element: String,
        /// `glib::Error` `Display` representation.
        message: String,
        /// Optional GStreamer debug payload.
        debug: Option<String>,
    },

    /// A runtime error reported by the pipeline bus (EOS, fatal warning, etc.).
    #[error("pipeline runtime error: {reason}")]
    Runtime { reason: String },
}

/// Top-level error type used at the CLI and library boundary.
///
/// Groups the per-layer error families ([`EffectError`],
/// [`InferenceError`], [`PipelineError`]) together with configuration
/// and I/O failures.  All conversions from lower layers happen via
/// `#[from]`, so application code can use `?` freely while still
/// retaining structured diagnostics for the §27 error renderer.
#[derive(Debug, Error)]
#[non_exhaustive]
#[expect(
    missing_docs,
    reason = "variant fields are exposed via #[error(...)] Display formatting"
)]
pub enum FluxError {
    /// An effect failed (configuration, preparation, or per-frame).
    #[error(transparent)]
    Effect(#[from] EffectError),

    /// An inference engine failed (model lifecycle or per-call).
    #[error(transparent)]
    Inference(#[from] InferenceError),

    /// The capture/output pipeline failed.
    #[error(transparent)]
    Pipeline(#[from] PipelineError),

    /// A configuration error reported by manual validation.
    ///
    /// `reason` is the machine-readable cause; `hint` is an optional
    /// user-facing suggestion (e.g. "check that the file path exists").
    /// Use [`FluxError::ConfigParse`] for TOML parser failures.
    #[error("config error: {reason}")]
    Config {
        reason: String,
        /// Optional hint for the user (e.g. "check that the file path exists").
        hint: Option<String>,
    },

    /// The configuration file could not be parsed as TOML.
    ///
    /// Wraps the upstream [`toml::de::Error`] so `?` from `toml::from_str`
    /// works directly.  The TOML error already contains line/column info.
    #[error("config parse failed: {source}")]
    ConfigParse {
        #[from]
        source: toml::de::Error,
    },

    /// An I/O operation failed, with the offending path attached.
    ///
    /// Construct via [`FluxError::io`].  The `path` is included in the
    /// `Display` output so users see *which* file failed without having
    /// to consult logs.
    #[error("I/O error at {path}: {source}", path = path.display())]
    Io {
        /// Filesystem path that the failed operation was targeting.
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// An aggregate of multiple failures, e.g. from `fluxframe check`.
    ///
    /// `primary` is the first failure (preserved so consumers can still
    /// `matches!()` on the typed variant); `count` is the total number of
    /// failures that the producer already rendered to the user.  Renderers
    /// (e.g. `main.rs`) should detect this variant and **skip** their own
    /// `Error:`/`Hint:` print to avoid duplicating output.
    #[error("{count} failures (primary: {primary})")]
    Aggregated {
        /// First failure in the aggregate; preserved so `matches!()` on the
        /// typed variant still works after aggregation.
        primary: Box<FluxError>,
        /// Total number of failures the producer already rendered.
        count: usize,
    },
}

impl FluxError {
    /// Construct an [`FluxError::Io`] from a path and the underlying
    /// [`std::io::Error`].
    ///
    /// A dedicated constructor (rather than `#[from]`) is required
    /// because the path is not recoverable from the I/O error itself —
    /// callers must supply it explicitly so the §27 diagnostic can
    /// render the failing file.
    #[must_use]
    pub fn io(path: PathBuf, source: std::io::Error) -> Self {
        FluxError::Io { path, source }
    }

    /// Return `true` if this is an [`FluxError::Aggregated`] variant.
    ///
    /// Convenience for renderers (e.g. `main.rs`) that need to skip their
    /// own diagnostic print when the producer already rendered each
    /// underlying failure.
    #[must_use]
    pub fn is_aggregated(&self) -> bool {
        matches!(self, FluxError::Aggregated { .. })
    }
}

/// User-facing diagnostic rendering surface for §27 "Error / Reason / Hint".
///
/// CLI binaries call [`Diagnostic::render`] to emit the canonical three-line
/// format on stderr.  Errors that do not carry a separate hint return
/// `None` from [`Diagnostic::hint`].
pub trait Diagnostic {
    /// User-facing reason string.  **Walks the `source()` chain** so
    /// wrappers like `#[error(transparent)]` don't hide the underlying
    /// cause; consecutive duplicates are deduplicated.
    fn reason(&self) -> String;

    /// Optional user-facing remediation hint.
    fn hint(&self) -> Option<&str> {
        None
    }
}

impl Diagnostic for FluxError {
    fn reason(&self) -> String {
        // walk the source chain so structured wrappers (transparent) don't
        // hide the original message; concatenate with ": " between layers.
        let mut parts = vec![self.to_string()];
        let mut src: Option<&dyn std::error::Error> = std::error::Error::source(self);
        while let Some(s) = src {
            parts.push(s.to_string());
            src = s.source();
        }
        // dedupe consecutive equal entries (transparent variants make
        // top-level and immediate source identical).
        parts.dedup();
        parts.join(": ")
    }

    fn hint(&self) -> Option<&str> {
        match self {
            FluxError::Config { hint, .. } => hint.as_deref(),
            FluxError::Pipeline(
                PipelineError::MissingElement { hint, .. }
                | PipelineError::InputDeviceUnavailable { hint, .. }
                | PipelineError::OutputDeviceUnavailable { hint, .. },
            )
            | FluxError::Inference(InferenceError::BackendUnavailable { hint, .. }) => Some(hint),
            _ => None,
        }
    }
}
