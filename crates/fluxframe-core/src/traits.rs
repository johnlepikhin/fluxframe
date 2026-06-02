//! Public trait surface of FluxFrame.
//!
//! Effects, inference engines, sources and sinks all implement these
//! traits.  The contract is: GStreamer never leaks across this boundary
//! and ONNX Runtime never leaks across the inference boundary.

use std::sync::Arc;

use crate::context::{FrameContext, ProcessingContext};
use crate::error::{EffectError, InferenceError, PipelineError};
use crate::frame::{PixelFormat, VideoFrame};
use crate::plane::SubchainKind;

/// Raw, format-coupled parameters passed to an effect at startup.
///
/// The choice of `toml::Value` is provisional: it keeps the registry
/// free of a closed schema while still allowing each effect to parse
/// only what it needs. If the config format changes (YAML, JSON), this
/// alias becomes the single migration point.
pub type RawEffectParams = toml::Value;

/// Implemented by every effect in the chain.  Mirrors §15 of the spec.
///
/// Lifecycle ordering: `configure` → `prepare` → repeated `process` → `shutdown`.
///
/// # Threading
///
/// Effects are `Send` but not `Sync`. The realtime pipeline owns each
/// effect on a single worker thread; if shared inspection is needed
/// (e.g. metrics), wrap the effect in `Arc<Mutex<dyn VideoEffect>>` —
/// `Mutex` provides `Sync` even when its contents are not.
/// Object-safe helper enabling downcasting from `&mut dyn VideoEffect`
/// to a concrete effect type (`CompositeEffect` in particular). All
/// `'static` types automatically implement this via the blanket impl
/// below; no per-effect boilerplate is needed.
///
/// End users **should not** implement this manually — the blanket impl
/// below covers every `'static` type. Implementing it manually risks
/// shadowing the blanket impl and confusing the downcast machinery.
pub trait AsAnyMut {
    /// Coerce `self` to `&mut dyn Any` for `downcast_mut`.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl<T: std::any::Any> AsAnyMut for T {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Top-level video effect: operates on a whole [`VideoFrame`] at a
/// time. The composite effect, passthrough effect, and any future
/// frame-level effects implement this trait. `AsAnyMut` supertrait
/// enables Stage 13 downcast from `&mut dyn VideoEffect` to a
/// concrete type when the runtime needs effect-specific methods.
pub trait VideoEffect: Send + AsAnyMut {
    /// Stable identifier used by the registry and the config (`snake_case`).
    fn name(&self) -> &'static str;

    /// Apply user-supplied configuration.  May be a no-op for stateless
    /// effects.
    ///
    /// **MAY be called after `prepare()`** for live reconfiguration via
    /// the control socket (Stage 13). Implementations either update
    /// `self.config` in place or re-allocate scratch lazily on the
    /// next `process()`. MUST NOT panic on a re-call; return
    /// `EffectError::InvalidConfig` on invalid input without
    /// mutating self.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError`] if the supplied parameters cannot be parsed
    /// or are semantically invalid.  Errors here abort startup.
    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError>;

    /// Negotiate runtime resources (allocate buffers, load assets) once the
    /// `ProcessingContext` is known.  Runs after pipeline caps are settled
    /// but before the first frame.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError`] if resource acquisition fails (e.g. model
    /// load, buffer allocation, GPU init).
    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError>;

    /// Process a single frame in place.  Must not block on I/O.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError`] if processing fails for the current frame.
    /// Whether the pipeline drops the frame or aborts is decided by the
    /// runtime, not by the effect.
    fn process(
        &mut self,
        frame: &mut VideoFrame,
        context: &mut FrameContext,
    ) -> Result<(), EffectError>;

    /// Release runtime resources.  Default is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError`] if teardown fails.  The runtime will log
    /// the error but continue shutdown of remaining effects.
    fn shutdown(&mut self) -> Result<(), EffectError> {
        Ok(())
    }

    /// Live-reconfigure a sub-effect identified by `(section, name)`.
    /// Used by Stage 13's control socket. The default impl returns
    /// [`EffectError::ProcessFailed`] indicating the effect does not
    /// host a named sub-chain — only the composite effect's override
    /// has a real implementation.
    ///
    /// `section` is the typed [`SubchainKind`] for the composite;
    /// future effects may host their own namespaces.
    ///
    /// # Errors
    ///
    /// Implementations return [`EffectError::InvalidConfig`] for
    /// unknown sub-chain names (the chain machinery exists but no
    /// effect matches `name`) and [`EffectError::ProcessFailed`] only
    /// when no sub-chain machinery exists at all (i.e. the default
    /// implementation on a plain effect). [`EffectError::InvalidConfig`]
    /// is also returned when the new params do not validate for the
    /// named sub-effect.
    fn reconfigure_named_effect(
        &mut self,
        section: SubchainKind,
        name: &str,
        _params: RawEffectParams,
    ) -> Result<(), EffectError> {
        Err(EffectError::ProcessFailed {
            name: self.name().to_string(),
            reason: format!(
                "effect '{}' does not host a named sub-chain (asked for {section}.{name})",
                self.name()
            ),
        })
    }
}

/// Static description of an inference model exposed by an [`InferenceEngine`].
///
/// Effects use this to validate compatibility (expected input layout) before
/// the first inference call, so the hot path only sees pre-validated tensors.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Human-readable model identifier (typically the file stem).
    ///
    /// Held as `Arc<str>` so engines can return cheap clones from
    /// [`InferenceEngine::model_info`] without a per-call allocation
    /// on the hot path.
    pub name: Arc<str>,
    /// Expected input tensor width in pixels.
    pub input_width: u32,
    /// Expected input tensor height in pixels.
    pub input_height: u32,
    /// Pixel format the engine expects for raw image input.
    pub input_format: PixelFormat,
}

/// Tensor input for inference.  Layout/dtype is described by the model
/// config consumed by the backend, not by this struct, so the trait remains
/// runtime-agnostic.
#[derive(Debug, Clone)]
pub struct InferenceInput<'a> {
    /// Flattened tensor data in the model's expected dtype (currently `f32`).
    pub data: &'a [f32],
    /// Tensor shape in row-major order.  Interpretation (NCHW/NHWC/…) is
    /// the engine's responsibility per its model config.
    pub shape: &'a [usize],
}

/// Tensor output from inference.  Owned so the caller can outlive the
/// engine's internal buffers without an extra copy on the hot path.
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    /// Flattened result tensor.
    pub data: Vec<f32>,
    /// Result tensor shape in row-major order.
    pub shape: Vec<usize>,
}

/// ML inference abstraction.  Effects depend on this trait, not on
/// `ort::Session`.  Mirrors §17 of the spec.
///
/// # Threading
///
/// Inference engines are `Send` but not `Sync`. Backends often hold
/// non-thread-safe session handles (e.g. `ort::Session`); the pipeline
/// pins each engine to one worker thread. For shared access wrap in
/// `Arc<Mutex<dyn InferenceEngine>>`.
pub trait InferenceEngine: Send {
    /// Return static information about the loaded model.
    fn model_info(&self) -> ModelInfo;

    /// Run a single forward pass.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] on shape mismatch, backend failure, or
    /// resource exhaustion.
    fn infer(&mut self, input: InferenceInput<'_>) -> Result<InferenceOutput, InferenceError>;
}

/// Abstraction over video capture backends (V4L2, testsrc, …).
///
/// `start` returns once frames are flowing; the actual delivery mechanism
/// (channel, latest-frame slot) belongs to the runtime, not this trait,
/// to keep backends agnostic of the threading model chosen by the CLI.
///
/// # Threading
///
/// Sources are `Send` but not `Sync`. The pipeline has a single owner per
/// source; concurrent control (e.g. external pause) should go through a
/// runtime-level command channel, not direct shared access.
pub trait VideoSource: Send {
    /// Begin capturing.  Must return only once frames are actually flowing
    /// to the runtime's delivery channel.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] if the backend cannot be initialised
    /// (device busy, missing permissions, unsupported caps, …).
    fn start(&mut self) -> Result<(), PipelineError>;

    /// Stop capture and release device resources.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] if the backend reports a teardown failure;
    /// the runtime logs it but continues shutdown.
    fn stop(&mut self) -> Result<(), PipelineError>;
}

/// Abstraction over video output backends (v4l2sink, fakesink, …).
///
/// # Threading
///
/// Sinks are `Send` but not `Sync`.  Like sources, the pipeline owns each
/// sink on a single thread; concurrent access should be mediated by the
/// runtime, not by the trait.
pub trait VideoSink: Send {
    /// Open the output device / initialise any downstream pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] on device-open failure or caps mismatch.
    fn start(&mut self) -> Result<(), PipelineError>;

    /// Deliver a processed frame to the sink.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] if the backend rejects the frame
    /// (format/size mismatch) or the underlying device fails.
    fn push(&mut self, frame: &VideoFrame) -> Result<(), PipelineError>;

    /// Tear down the sink and release device resources.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] on teardown failure; logged but
    /// non-fatal for the rest of the pipeline.
    fn stop(&mut self) -> Result<(), PipelineError>;
}
