//! Internal helpers shared between the input and output pipelines.
//!
//! Kept `pub(crate)` so GStreamer types do not leak through the crate's
//! public surface.  Both helpers are deliberately tiny — the alternative
//! was to duplicate them across `input.rs` and `output.rs`, which led to
//! drift (one site silently truncated oversized dimensions, the other
//! used `unwrap_or(i32::MAX)`).

use fluxframe_core::error::PipelineError;
use fluxframe_core::frame::PixelFormat;

use crate::frame_conv::pixel_format_to_gst;

/// Construct a GStreamer element by factory name, surfacing a structured
/// [`PipelineError::MissingElement`] when the plugin is not registered.
pub(crate) fn make_element(
    factory: &str,
    name: &str,
) -> Result<gstreamer::Element, PipelineError> {
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
