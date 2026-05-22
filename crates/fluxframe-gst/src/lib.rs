//! GStreamer + V4L2 layer for FluxFrame.
//!
//! Stage 0 surface: only `init` exists, so that workspace builds prove
//! `pkg-config` linkage works.  Pipeline construction (`input`, `output`),
//! V4L2 device enumeration (`v4l2`) and caps negotiation land in Stages 1
//! and 2.

#![warn(missing_docs)]
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(not(target_os = "linux"))]
compile_error!("fluxframe-gst is Linux-only (see §2 of doc/ideas/001-mvp.md)");

pub mod bus;
pub mod frame_conv;
pub mod input;
pub mod output;
pub mod slot;
mod util;
pub mod v4l2;

pub use bus::{BusEvent, BusListener, WatchedPipeline};
pub use slot::LatestFrameSlot;
pub use v4l2::{V4l2Device, V4l2DeviceKind, enumerate_devices};

use fluxframe_core::error::PipelineError;

/// Initialise GStreamer.  Must be called once before any other pipeline
/// operation.  Safe to call multiple times — `gstreamer::init` is idempotent.
pub fn init() -> Result<(), PipelineError> {
    gstreamer::init().map_err(|e| PipelineError::Runtime {
        reason: format!("gstreamer::init failed: {e}"),
    })
}
