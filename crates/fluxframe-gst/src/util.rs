//! Internal helpers shared between the input and output pipelines.
//!
//! Kept `pub(crate)` so GStreamer types do not leak through the crate's
//! public surface.  Both helpers are deliberately tiny — the alternative
//! was to duplicate them across `input.rs` and `output.rs`, which led to
//! drift (one site silently truncated oversized dimensions, the other
//! used `unwrap_or(i32::MAX)`).

use std::fs::OpenOptions;
use std::io;
use std::path::Path;

use fluxframe_core::error::PipelineError;
use fluxframe_core::frame::PixelFormat;

use crate::frame_conv::pixel_format_to_gst;

/// Construct a GStreamer element by factory name, surfacing a structured
/// [`PipelineError::MissingElement`] when the plugin is not registered.
pub(crate) fn make_element(factory: &str, name: &str) -> Result<gstreamer::Element, PipelineError> {
    gstreamer::ElementFactory::make(factory)
        .name(name)
        .build()
        .map_err(|_| PipelineError::MissingElement {
            element: factory.into(),
            hint: format!("GStreamer plugin providing `{factory}` is not installed"),
        })
}

/// Build a `video/x-raw` caps description with the given geometry, framerate
/// and pixel format.
///
/// GStreamer requires `i32` for width/height/framerate; values that do not
/// fit are rejected with [`PipelineError::CapsNegotiationFailed`] instead of
/// being silently truncated to `i32::MAX` (which would have negotiated an
/// arbitrary smaller resolution downstream).
pub(crate) fn build_caps(
    width: u32,
    height: u32,
    fps: u32,
    format: PixelFormat,
) -> Result<gstreamer::Caps, PipelineError> {
    let width_i32 = i32::try_from(width).map_err(|_| PipelineError::CapsNegotiationFailed {
        reason: format!("value too large for GStreamer: width={width}"),
    })?;
    let height_i32 = i32::try_from(height).map_err(|_| PipelineError::CapsNegotiationFailed {
        reason: format!("value too large for GStreamer: height={height}"),
    })?;
    let fps_i32 = i32::try_from(fps).map_err(|_| PipelineError::CapsNegotiationFailed {
        reason: format!("value too large for GStreamer: fps={fps}"),
    })?;

    let gst_fmt = pixel_format_to_gst(format);
    Ok(gstreamer::Caps::builder("video/x-raw")
        .field("format", gst_fmt.to_str())
        .field("width", width_i32)
        .field("height", height_i32)
        .field("framerate", gstreamer::Fraction::new(fps_i32, 1))
        .build())
}

/// Map an `io::Error` from a v4l2 device open into a structured pipeline
/// hint.  Stays private to this module; the public surface is the two
/// `check_v4l2_*_access` wrappers, which differ only in the open flags
/// they pass.
fn map_v4l2_open_error(err: &io::Error) -> &'static str {
    match err.kind() {
        io::ErrorKind::NotFound => {
            "check that the device exists; run 'fluxframe list' for the available devices"
        }
        io::ErrorKind::PermissionDenied => "add user to the 'video' group or check udev rules",
        io::ErrorKind::ResourceBusy => {
            "another application is holding the device; close it (e.g. browser tab, OBS)"
        }
        _ => "see dmesg for kernel-level diagnostics",
    }
}

/// Verify the calling process can open `device` for reading.
///
/// Surfaces a [`PipelineError::InputDeviceUnavailable`] with an actionable
/// hint *before* GStreamer's `v4l2src` tries to open the same path — the
/// raw GStreamer state-change error for `EACCES`/`EBUSY`/`ENOENT` is opaque
/// ("Internal data stream error" or similar), which defeats the §27
/// Error/Reason/Hint diagnostic contract without this pre-check.
///
/// There is a small TOCTOU window between this check and the actual
/// `v4l2src` open; the Stage 2 plan documents that as an accepted
/// trade-off (still strictly better than no hint at all).
pub(crate) fn check_v4l2_input_access(device: &Path) -> Result<(), PipelineError> {
    OpenOptions::new()
        .read(true)
        .open(device)
        .map(|_| ())
        .map_err(|e| PipelineError::InputDeviceUnavailable {
            device: device.display().to_string(),
            reason: e.to_string(),
            hint: map_v4l2_open_error(&e).into(),
        })
}

/// Verify the calling process can open `device` for writing.
///
/// Surfaces a [`PipelineError::OutputDeviceUnavailable`] with an actionable
/// hint *before* GStreamer's `v4l2sink` tries to open the same path — the
/// raw GStreamer error for `EACCES`/`EBUSY`/`ENOENT` is opaque ("Device
/// '/dev/video10' cannot be opened for writing"), which makes Stage 2's
/// "diagnostic over silence" trade-off impossible without this pre-check.
///
/// There is a small TOCTOU window between this check and the actual sink
/// open; that's documented in the Stage 2 plan as an accepted trade-off.
pub(crate) fn check_v4l2_output_access(device: &Path) -> Result<(), PipelineError> {
    OpenOptions::new()
        .write(true)
        .open(device)
        .map(|_| ())
        .map_err(|e| PipelineError::OutputDeviceUnavailable {
            device: device.display().to_string(),
            reason: e.to_string(),
            hint: map_v4l2_open_error(&e).into(),
        })
}
