//! Processing-time context structs passed to effects.
//!
//! The fields here grow as new effects need them.  Keeping the structs
//! distinct (`ProcessingContext` is per-pipeline, `FrameContext` is
//! per-frame) prevents effects from accidentally holding references to
//! per-frame data across frames.

use std::sync::Arc;

use crate::frame::{PixelFormat, Timestamp};
use crate::geometry::Region;
use crate::metrics::{Counters, EffectTelemetry};

/// Pipeline-wide context handed to every effect once during `prepare`.
///
/// `counters` is the supervisor's `Arc<Counters>` shared with effects
/// that need to publish runtime events directly (for example, a
/// sticky-fallback decorator inside an effect's backend chain
/// incrementing `blur_runtime_fallback_gpu_to_cpu`).  `None` for
/// stand-alone effect-chain tests and any caller that has no
/// supervisor — effects MUST treat the absence as "no counter
/// publishing" rather than as an error.
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    /// Negotiated frame width in pixels.
    pub width: u32,
    /// Negotiated frame height in pixels.
    pub height: u32,
    /// Negotiated pixel format flowing through the chain.
    pub format: PixelFormat,
    /// Nominal frame rate (frames per second).
    pub fps: u32,
    /// Process-wide event counters shared with the supervisor.  See
    /// the struct-level docstring for the contract; `None` is legal.
    pub counters: Option<Arc<Counters>>,
}

/// Per-frame context.  Effects may mutate this to record diagnostics or
/// signal downstream behaviour (e.g. "fallback was applied to this frame").
#[derive(Debug, Clone, Default)]
pub struct FrameContext {
    /// Monotonically increasing frame index assigned at capture.
    pub frame_sequence: u64,
    /// Pipeline-monotonic timestamp at which the frame entered processing.
    pub frame_timestamp: Timestamp,
    /// Set by an effect (or the runtime) when it had to fall back for this frame.
    pub fallback_active: bool,
    /// Sink for per-stage timings.  The supervisor populates this before
    /// invoking the effect chain; defaults to a no-op so test/stand-alone
    /// constructions of [`FrameContext`] work without wiring metrics.
    pub telemetry: EffectTelemetry,
    /// Crop-before-processing handshake with the runtime's framing
    /// stage (see [`Framing`]).
    pub framing: Framing,
    /// Set by an effect that holds frames back (a pipelined composite
    /// delays its output by one frame) when it has nothing to emit for
    /// this call: the first frame after a start or reset.  The runtime
    /// then neither publishes the frame nor counts it as dropped, and
    /// later effects in the chain do not run.  Reset by the runtime
    /// before each frame.
    pub output_withheld: bool,
}

/// Handshake between the runtime's framing stage, which cuts each
/// processed frame out of a (usually larger) captured frame, and an
/// effect that decides where to cut (`auto_frame`).
///
/// The cut for frame *N + 1* is decided while processing frame *N*.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Framing {
    /// Set by the runtime before the chain runs: where the frame being
    /// processed came from.  `None` when the chain runs without a
    /// framing stage (tests, stand-alone use) — framing effects then
    /// have nothing to steer.
    pub source: Option<FramingSource>,
    /// Set by a framing effect: the region of the next captured frame
    /// to process.  Cleared by the runtime before each frame; when no
    /// effect sets it, the runtime returns to the full view.
    pub request: Option<Region>,
}

/// Where a processed frame was cut from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FramingSource {
    /// Captured frame width in pixels.
    pub width: u32,
    /// Captured frame height in pixels.
    pub height: u32,
    /// Region of the captured frame that was scaled to the processed
    /// frame, in captured-frame pixels.
    pub region: Region,
}

/// Coarse runtime state used by the metrics layer and shutdown logic.
///
/// Grows as Stage 5 (realtime hardening) lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum RuntimeState {
    /// No worker thread is active; the pipeline is idle.
    #[default]
    Stopped,
    /// Pipeline construction in progress (caps negotiation, model load).
    Starting,
    /// Frames are flowing end-to-end.
    Running,
    /// Shutdown initiated; workers are draining or releasing resources.
    Stopping,
}
