//! Conversion between `gst::Sample`/`gst::Buffer` and FluxFrame's
//! [`VideoFrame`].
//!
//! The contract is one-way at the boundary of `fluxframe-gst`:
//! GStreamer types stay inside this crate; the rest of the workspace
//! sees only `VideoFrame`.  Stage 1 implements an owned-copy path
//! (`Vec<u8>` allocation per frame); zero-copy via `Buffer::map` plus a
//! lifetime-anchored `FrameBuffer` variant lands in Stage 5.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use fluxframe_core::error::PipelineError;
use fluxframe_core::frame::{FrameBuffer, FrameMeta, PixelFormat, Stride, Timestamp, VideoFrame};
use gstreamer_video::VideoInfo;
use tracing::warn;

/// Translate a GStreamer pixel format into FluxFrame's enum.
///
/// Returns [`PipelineError::UnsupportedGstFormat`] for variants the MVP
/// processing chain does not handle, carrying the raw GStreamer label so
/// diagnostics do not have to invent a placeholder [`PixelFormat`].
pub fn pixel_format_from_gst(
    fmt: gstreamer_video::VideoFormat,
) -> Result<PixelFormat, PipelineError> {
    use gstreamer_video::VideoFormat as G;
    Ok(match fmt {
        G::Rgb => PixelFormat::Rgb,
        G::Rgba => PixelFormat::Rgba,
        G::Bgr => PixelFormat::Bgr,
        G::Yuy2 => PixelFormat::Yuy2,
        G::Nv12 => PixelFormat::Nv12,
        G::Gray8 => PixelFormat::Gray8,
        other => {
            return Err(PipelineError::UnsupportedGstFormat {
                gst_label: other.to_str().to_string(),
            });
        }
    })
}

/// Translate FluxFrame's pixel format into the corresponding GStreamer enum.
#[must_use]
pub fn pixel_format_to_gst(fmt: PixelFormat) -> gstreamer_video::VideoFormat {
    use gstreamer_video::VideoFormat as G;
    match fmt {
        PixelFormat::Rgb => G::Rgb,
        PixelFormat::Rgba => G::Rgba,
        PixelFormat::Bgr => G::Bgr,
        PixelFormat::Yuy2 => G::Yuy2,
        PixelFormat::Nv12 => G::Nv12,
        PixelFormat::Gray8 => G::Gray8,
    }
}

/// Convert a `gst::Sample` (one frame's worth of buffer + caps) into a
/// [`VideoFrame`].
///
/// Performs an owned copy of the buffer's bytes — Stage 1's hot path is
/// not yet zero-copy.
///
/// `sequence` is the per-pipeline monotonic counter.  Lives outside this
/// function so each [`crate::input::InputPipeline`] instance owns its own
/// numbering instead of sharing a process-global atomic (which entangled
/// concurrent test pipelines and any future multi-input deployment).
///
/// # Errors
///
/// Returns [`PipelineError`] if caps are missing/unparseable, the buffer
/// cannot be mapped, the pixel format is not supported, or the buffer
/// does not match the negotiated dimensions.
pub fn sample_to_frame(
    sample: &gstreamer::Sample,
    sequence: &AtomicU64,
) -> Result<VideoFrame, PipelineError> {
    let caps = sample.caps().ok_or_else(|| PipelineError::Runtime {
        reason: "sample has no caps".into(),
    })?;
    let info = VideoInfo::from_caps(caps).map_err(|e| PipelineError::CapsNegotiationFailed {
        reason: format!("VideoInfo::from_caps failed: {e}"),
    })?;
    let buffer = sample.buffer().ok_or_else(|| PipelineError::Runtime {
        reason: "sample has no buffer".into(),
    })?;
    let map = buffer.map_readable().map_err(|_| PipelineError::Runtime {
        reason: "failed to map buffer for reading".into(),
    })?;

    let format = pixel_format_from_gst(info.format())?;
    let width = info.width();
    let height = info.height();

    let bytes_per_pixel = format
        .bytes_per_pixel()
        .ok_or(PipelineError::UnsupportedPixelFormat { format })?;
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|p| p.checked_mul(bytes_per_pixel))
        .ok_or_else(|| PipelineError::Runtime {
            reason: "frame size overflows usize".into(),
        })?;

    if map.size() < expected {
        return Err(PipelineError::Runtime {
            reason: format!(
                "buffer too small: got {} bytes, expected at least {expected}",
                map.size()
            ),
        });
    }

    let data = FrameBuffer::Owned(map.as_slice()[..expected].to_vec());
    let stride = Stride::Packed(width as usize * bytes_per_pixel);

    let seq = sequence.fetch_add(1, Ordering::Relaxed);
    let source_timestamp = buffer.pts().map(|ts| Timestamp::from_nanos(ts.nseconds()));
    let duration = buffer
        .duration()
        .map(|d| Duration::from_nanos(d.nseconds()));

    Ok(VideoFrame {
        data,
        width,
        height,
        format,
        stride,
        meta: FrameMeta {
            sequence: seq,
            timestamp: source_timestamp.unwrap_or_default(),
            source_timestamp,
            duration,
        },
    })
}

/// Wrap a [`VideoFrame`] into a fresh `gst::Buffer` that can be pushed
/// into an `appsrc`.  Copies the pixel data.
///
/// Takes the frame by value as a forward-compatibility hook: when Stage 5
/// switches to zero-copy via `FrameBuffer::Mapped`, ownership of the frame
/// will need to transfer into the buffer so the `gst::Memory` mapping can
/// outlive the call.  Today the body still copies regardless.
///
/// # Errors
///
/// Returns [`PipelineError::Runtime`] if buffer allocation fails.
#[allow(
    clippy::needless_pass_by_value,
    reason = "Stage 5 will move ownership of the frame's backing memory into the gst::Buffer; \
              keeping the by-value signature now avoids a churning API break later."
)]
pub fn frame_to_buffer(frame: VideoFrame) -> Result<gstreamer::Buffer, PipelineError> {
    let bytes = frame.data.as_slice();
    let mut buffer =
        gstreamer::Buffer::with_size(bytes.len()).map_err(|e| PipelineError::Runtime {
            reason: format!("Buffer::with_size failed: {e}"),
        })?;
    {
        // `Buffer::with_size` returns a freshly allocated buffer whose
        // refcount is exactly one, so `get_mut` is guaranteed to succeed
        // by the GStreamer contract.  We treat a failure here as a
        // programming error rather than a runtime condition.
        let buffer_ref = buffer
            .get_mut()
            .expect("buffer is uniquely owned immediately after with_size allocation");
        let mut map = buffer_ref
            .map_writable()
            .map_err(|_| PipelineError::Runtime {
                reason: "failed to map buffer for writing".into(),
            })?;
        map.as_mut_slice().copy_from_slice(bytes);
    }
    {
        let buffer_ref = buffer
            .get_mut()
            .expect("buffer is uniquely owned immediately after with_size allocation");
        buffer_ref.set_pts(gstreamer::ClockTime::from_nseconds(
            frame.meta.timestamp.as_nanos(),
        ));
        if let Some(d) = frame.meta.duration {
            match u64::try_from(d.as_nanos()) {
                Ok(nanos) => {
                    buffer_ref.set_duration(gstreamer::ClockTime::from_nseconds(nanos));
                }
                Err(_) => {
                    warn!(
                        duration_ns = ?d.as_nanos(),
                        "frame duration overflows u64 nanoseconds; skipping set_duration"
                    );
                }
            }
        }
    }
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_format_mapping_roundtrips() {
        for fmt in [
            PixelFormat::Rgb,
            PixelFormat::Rgba,
            PixelFormat::Bgr,
            PixelFormat::Yuy2,
            PixelFormat::Nv12,
            PixelFormat::Gray8,
        ] {
            let gst_fmt = pixel_format_to_gst(fmt);
            let back = pixel_format_from_gst(gst_fmt).expect("mapping reversible");
            assert_eq!(back, fmt);
        }
    }
}
