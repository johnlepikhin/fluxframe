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
/// Returns [`PipelineError::UnsupportedPixelFormat`] (with `raw_label`
/// populated) for variants the MVP processing chain does not handle, so
/// diagnostics carry the actual GStreamer label rather than a placeholder
/// [`PixelFormat`].
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
            return Err(PipelineError::UnsupportedPixelFormat {
                format: None,
                raw_label: Some(format!("{other:?}")),
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

    let bytes_per_pixel =
        format
            .bytes_per_pixel()
            .ok_or(PipelineError::UnsupportedPixelFormat {
                format: Some(format),
                raw_label: None,
            })?;
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

    let packed_row = (width as usize) * bytes_per_pixel;
    let layout = inspect_row_layout(buffer, &info, packed_row);
    if layout.needs_row_copy {
        warn!(
            seq = sequence.load(Ordering::Relaxed),
            width,
            height,
            ?format,
            packed_row,
            row_stride = layout.row_stride,
            offset = layout.offset,
            buffer_size = map.size(),
            "input buffer has non-packed stride/offset; falling back to row-by-row copy",
        );
    }
    if map.size() != expected {
        warn!(
            seq = sequence.load(Ordering::Relaxed),
            buffer_size = map.size(),
            expected,
            "input buffer size differs from packed W*H*bpp; trailing bytes ignored",
        );
    }

    let data = copy_packed_rgb(
        map.as_slice(),
        &layout,
        packed_row,
        height as usize,
        expected,
    )?;
    let stride = Stride::Packed(packed_row);

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

/// Per-buffer row layout descriptor: actual byte stride and plane
/// offset of the first plane, plus whether either differs from the
/// packed `width * bpp` layout that downstream code assumes.
struct RowLayout {
    /// Actual bytes per row in the buffer.
    row_stride: usize,
    /// Byte offset of the plane within the buffer (usually `0`; nonzero
    /// when a multi-plane buffer puts the active plane after metadata).
    offset: usize,
    /// `true` when either `row_stride != packed_row` or `offset != 0`.
    needs_row_copy: bool,
}

/// Inspect a buffer's row layout, preferring `VideoMeta` (per-buffer
/// overlay) over `VideoInfo::stride` (caps-derived default).  Picks the
/// maximum reasonable row_stride so a misreported value never under-counts
/// bytes and walks off the end of the buffer.
fn inspect_row_layout(
    buffer: &gstreamer::BufferRef,
    info: &VideoInfo,
    packed_row: usize,
) -> RowLayout {
    let packed_row_i32 = i32::try_from(packed_row).unwrap_or(i32::MAX);
    let info_stride = info.stride().first().copied().unwrap_or(packed_row_i32);
    let info_stride_usize = usize::try_from(info_stride).unwrap_or(packed_row);
    let (vm_stride_usize, vm_offset) =
        buffer
            .meta::<gstreamer_video::VideoMeta>()
            .map_or((packed_row, 0_usize), |vm| {
                let s = vm.stride().first().copied().unwrap_or(packed_row_i32);
                let s_usize = usize::try_from(s).unwrap_or(packed_row);
                let o = vm.offset().first().copied().unwrap_or(0);
                (s_usize, o)
            });
    let row_stride = vm_stride_usize.max(info_stride_usize).max(packed_row);
    RowLayout {
        row_stride,
        offset: vm_offset,
        needs_row_copy: row_stride != packed_row || vm_offset != 0,
    }
}

/// Copy `raw` (a mapped GStreamer buffer slice) into a freshly-owned
/// packed RGB `Vec`, honouring `layout`.  When the buffer is already
/// packed the fast bulk-copy path is used; otherwise we copy row-by-row.
fn copy_packed_rgb(
    raw: &[u8],
    layout: &RowLayout,
    packed_row: usize,
    height: usize,
    expected: usize,
) -> Result<FrameBuffer, PipelineError> {
    if !layout.needs_row_copy {
        return Ok(FrameBuffer::Owned(raw[..expected].to_vec()));
    }
    let needed = layout
        .offset
        .saturating_add(layout.row_stride.saturating_mul(height));
    if raw.len() < needed {
        return Err(PipelineError::Runtime {
            reason: format!(
                "buffer too small for row-aware copy: have {}, need {needed} (stride {}, offset {})",
                raw.len(),
                layout.row_stride,
                layout.offset,
            ),
        });
    }
    let mut packed = Vec::with_capacity(expected);
    for row in 0..height {
        let row_start = layout.offset + row * layout.row_stride;
        packed.extend_from_slice(&raw[row_start..row_start + packed_row]);
    }
    Ok(FrameBuffer::Owned(packed))
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
    // `Buffer::with_size` returns a freshly allocated buffer whose refcount
    // is exactly one, so `get_mut` is guaranteed to succeed by the GStreamer
    // contract.  We treat a failure here as a programming error rather than
    // a runtime condition.
    const UNIQUE_BUFFER_MSG: &str =
        "buffer is uniquely owned immediately after with_size allocation";

    let bytes = frame.data.as_slice();
    // Catch effect-chain bugs at the output boundary: if `bytes.len()`
    // disagrees with `width × height × bpp`, downstream `videoconvert`
    // reads the buffer through the caps-declared geometry and either
    // truncates or runs off the end → flicker / row-shift in the sink.
    let bpp = frame.format.bytes_per_pixel().unwrap_or(0);
    let expected_packed = (frame.width as usize) * (frame.height as usize) * bpp;
    if bpp > 0 && bytes.len() != expected_packed {
        warn!(
            seq = frame.meta.sequence,
            width = frame.width,
            height = frame.height,
            format = ?frame.format,
            bytes_len = bytes.len(),
            expected_packed,
            "output frame buffer size does not match packed W*H*bpp; sink will see a stride/length-shifted frame"
        );
    }
    let mut buffer =
        gstreamer::Buffer::with_size(bytes.len()).map_err(|e| PipelineError::Runtime {
            reason: format!("Buffer::with_size failed: {e}"),
        })?;
    {
        let buffer_ref = buffer.get_mut().expect(UNIQUE_BUFFER_MSG);
        let mut map = buffer_ref
            .map_writable()
            .map_err(|_| PipelineError::Runtime {
                reason: "failed to map buffer for writing".into(),
            })?;
        map.as_mut_slice().copy_from_slice(bytes);
    }
    {
        let buffer_ref = buffer.get_mut().expect(UNIQUE_BUFFER_MSG);
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
