//! Stage 15 — Idle mode (consumer-aware lifecycle).
//!
//! When the v4l2loopback output device has no readers the supervisor
//! tears down the input pipeline, drops the ONNX session after a
//! grace period, and publishes a cheap placeholder frame at low fps.
//! On consumer reconnect it resumes the full processing pipeline.
//!
//! Submodules:
//!
//! * [`state`] — pure state machine driving [`IdleEdge`]/[`IdleLevel`]
//!   transitions; no I/O. Wired into the worker loop in
//!   `runtime::run_process_loop`.
//! * `detector` — inotify-driven consumer presence detector with a
//!   `/proc/*/fd/` walk on every open/close event; polling fallback
//!   for environments where inotify is unavailable. Linux-only —
//!   `/proc/*/fd/` and `inotify` are both Linux interfaces; the
//!   module is `cfg`-gated and the supervisor consults the same
//!   gate when wiring it.
//! * `placeholder` (Step 2) — pre-rendered frame cache.
//! * `managed_composite` (Step 2) — lifecycle wrapper around
//!   `CompositeEffect` that owns the engine slot.
//! * `reload` (Step 4) — off-worker reload thread.

#[cfg(target_os = "linux")]
pub(crate) mod detector;
// `managed_composite` wraps `CompositeEffect` + ONNX engine
// lifecycle, both of which are gated behind the `ml` feature in
// fluxframe-effects. Without `ml` there is no engine to manage and
// `CompositeEffect` is not exposed by the effects crate, so the
// module is gated to keep slim builds compiling. The supervisor
// (Step 4) consults the same cfg when wiring the wrapper.
#[cfg(feature = "ml")]
pub(crate) mod managed_composite;
pub(crate) mod placeholder;
pub(crate) mod reload;
pub(crate) mod state;

// Re-exports forwarded for downstream Step 4 wiring. Marked
// `dead_code` until the supervisor consumes them; without the
// allow Rust 2024 surfaces them as unused-imports.
#[cfg(target_os = "linux")]
#[allow(
    unused_imports,
    reason = "Stage 15 Step 4 consumes these from runtime::run_process_loop"
)]
pub(crate) use detector::{ConsumerDetector, device_path};
#[cfg(feature = "ml")]
#[allow(
    unused_imports,
    reason = "Stage 15 Step 4 consumes these from runtime::run_process_loop"
)]
pub(crate) use managed_composite::ManagedComposite;
#[allow(
    unused_imports,
    reason = "Stage 15 Step 4 consumes these from runtime::run_process_loop"
)]
pub(crate) use placeholder::{Placeholder, build as build_placeholder};
#[allow(
    unused_imports,
    reason = "Stage 15 Step 4 consumes these from runtime::run_process_loop"
)]
pub(crate) use state::{
    ConsumerStatus, IdleEdge, IdleLevel, IdleState, IdleStateMachine, IdleTick,
};
