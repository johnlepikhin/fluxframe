//! GStreamer + V4L2 layer for FluxFrame.
//!
//! Stage 0 surface: only `init` exists, so that workspace builds prove
//! `pkg-config` linkage works.  Pipeline construction (`input`, `output`),
//! V4L2 device enumeration (`v4l2`) and caps negotiation land in Stages 1
//! and 2.

#![warn(missing_docs)]
// The crate opts out of the workspace's `unsafe_code = "forbid"` (see
// `Cargo.toml`) because the V4L2 event ioctls have no safe wrapper
// anywhere in the ecosystem. `forbid` cannot be relaxed per-module, so
// the exemption is re-narrowed here: `deny` at the crate root plus a
// single `allow` on the one module that needs it. Anything else that
// wants `unsafe` has to add its own `allow` and justify it in review,
// rather than inheriting a crate-wide permission.
#![deny(unsafe_code)]
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(not(target_os = "linux"))]
compile_error!("fluxframe-gst is Linux-only (see §2 of doc/ideas/001-mvp.md)");

pub mod bus;
pub mod frame_conv;
pub mod input;
pub mod output;
pub mod slot;
pub mod util;
pub mod v4l2;
pub mod v4l2_caps;
#[allow(
    unsafe_code,
    reason = "raw VIDIOC_SUBSCRIBE_EVENT / VIDIOC_DQEVENT ioctls and poll(2); \
              see the module docs for the safety argument"
)]
pub mod v4l2_events;

pub use bus::{BusEvent, BusListener, BusSource, WatchedPipeline, translate_fatal};
pub use slot::LatestFrameSlot;
pub use util::{check_v4l2_input_access, check_v4l2_output_access};
pub use v4l2::{
    EnumerationStatus, LoopbackState, V4l2Device, V4l2DeviceKind, enumerate_devices,
    enumerate_devices_in, enumerate_devices_status, enumerate_devices_status_in,
    read_loopback_state, read_loopback_state_in,
};
pub use v4l2_events::{ClientUsage, ProbeFailure, classify_probe_error, probe_client_usage};

use fluxframe_core::error::PipelineError;

/// Initialise GStreamer.  Must be called once before any other pipeline
/// operation.  Safe to call multiple times — `gstreamer::init` is idempotent.
pub fn init() -> Result<(), PipelineError> {
    gstreamer::init().map_err(|e| PipelineError::Runtime {
        reason: format!("gstreamer::init failed: {e}"),
    })
}
