//! Processing-time context structs passed to effects.
//!
//! The fields here grow as new effects need them.  Keeping the structs
//! distinct (`ProcessingContext` is per-pipeline, `FrameContext` is
//! per-frame) prevents effects from accidentally holding references to
//! per-frame data across frames.

use crate::frame::{PixelFormat, Timestamp};

/// Pipeline-wide context handed to every effect once during `prepare`.
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
