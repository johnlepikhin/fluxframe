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
//! * `detector` — consumer-presence detector. Primary source is the
//!   kernel's `V4L2_EVENT_PRI_CLIENT_USAGE`; an inotify + `/proc/*/fd/`
//!   heuristic (itself falling back to `/proc` polling) covers drivers
//!   that do not implement it. Selectable via `idle.presence_source`;
//!   see the module docs for why the heuristic cannot be authoritative
//!   on its own. Linux-only, `cfg`-gated, and the supervisor consults
//!   the same gate when wiring it.
//! * `resync` — the detector's safety net: re-reads the driver's
//!   absolute capture-usage value on a timer, because the kernel event
//!   it otherwise relies on is edge-triggered and a single lost event
//!   used to latch the verdict for the rest of the run.
//! * `placeholder` (Step 2) — pre-rendered frame cache.
//! * `managed_composite` (Step 2) — lifecycle wrapper around
//!   `CompositeEffect` that owns the engine slot.
//! * `reload` (Step 4) — off-worker reload thread.

#[cfg(target_os = "linux")]
pub(crate) mod detector;
// Level-triggered repair for the detector's verdict. Same `cfg` as
// `detector`: it re-opens the same v4l2loopback node and is meaningless
// without it.
#[cfg(target_os = "linux")]
pub(crate) mod resync;
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
pub(crate) use detector::{ConsumerDetector, DetectorParams, device_path};
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
