//! Video frame model.
//!
//! `VideoFrame` is the universal currency between capture, effect chain
//! and output.  It is deliberately decoupled from GStreamer types so that
//! effects and the inference layer never reach across that boundary.

use std::sync::Arc;
use std::time::Duration;

/// Pixel layout of a `VideoFrame`.
///
/// MVP processing path operates on `Rgb`/`Rgba`; the other variants are
/// reserved for capture/output negotiation and future zero-copy paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 24-bit packed RGB, 8 bits per channel, R-G-B byte order.
    Rgb,
    /// 32-bit packed RGBA, 8 bits per channel, R-G-B-A byte order.
    Rgba,
    /// 24-bit packed BGR, 8 bits per channel, B-G-R byte order.
    /// Common with V4L2 / OpenCV-derived sources.
    Bgr,
    /// 16-bit packed YUV 4:2:2 (`YUYV`).  Two pixels per 4-byte group:
    /// `Y0 U Y1 V`.
    Yuy2,
    /// 12-bit semi-planar YUV 4:2:0.  Plane 0 carries `Y` (full res),
    /// plane 1 carries interleaved `UV` at half resolution in each
    /// dimension.  Most common camera capture format.
    Nv12,
    /// 8-bit single-channel luminance.
    Gray8,
}

impl PixelFormat {
    /// Number of bytes per pixel for packed formats.
    ///
    /// Returns `None` for planar/subsampled formats (`Nv12`) where a single
    /// scalar does not describe the layout.
    #[must_use]
    pub fn bytes_per_pixel(self) -> Option<usize> {
        match self {
            Self::Rgb | Self::Bgr => Some(3),
            Self::Rgba => Some(4),
            Self::Yuy2 => Some(2),
            Self::Gray8 => Some(1),
            Self::Nv12 => None,
        }
    }
}

/// Monotonic timestamp expressed in nanoseconds since pipeline start.
///
/// Kept as a plain newtype so `fluxframe-core` does not pull in
/// `gstreamer::ClockTime`.  Conversion helpers live in `fluxframe-gst`.
///
/// The inner field is private; use [`Timestamp::from_nanos`] /
/// [`Timestamp::as_nanos`] to convert.  This guards against accidental
/// construction with foreign clock domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Timestamp(u64);

impl Timestamp {
    /// Construct a timestamp from a nanosecond count.  The caller is
    /// responsible for ensuring the value is in the pipeline's monotonic
    /// clock domain.
    #[must_use]
    pub fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Returns the underlying nanosecond count.
    #[must_use]
    pub fn as_nanos(self) -> u64 {
        self.0
    }

    /// Returns the timestamp as a [`Duration`] since pipeline start.
    #[must_use]
    pub fn as_duration(self) -> Duration {
        Duration::from_nanos(self.0)
    }
}

/// Frame payload storage.
///
/// `Owned` is the default MVP path.  `Shared` is reserved for the future
/// zero-copy/refcounted paths mentioned in §18 of the spec; introducing it
/// now keeps downstream APIs stable when that optimisation lands.
#[derive(Debug, Clone)]
pub enum FrameBuffer {
    /// Heap-owned buffer.  Default MVP variant — mutation is in-place and
    /// free of refcount traffic.
    Owned(Vec<u8>),
    /// Refcounted shared buffer.  Reserved for zero-copy paths; mutation
    /// requires explicit promotion via [`FrameBuffer::make_owned`].
    Shared(Arc<[u8]>),
}

impl FrameBuffer {
    /// Read-only access to the underlying bytes, regardless of variant.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v.as_slice(),
            Self::Shared(s) => s.as_ref(),
        }
    }

    /// Total byte length of the payload.
    #[must_use]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns `true` if the payload is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Mutable slice **only if buffer is `Owned`**. Returns `None` for
    /// `Shared` without copying — the caller must promote via
    /// [`FrameBuffer::make_owned`] explicitly.
    ///
    /// The realtime hot path should use this method: it guarantees no
    /// hidden allocation, and surfaces accidental `Shared` arrivals as
    /// a `None` that the caller is forced to handle.
    pub fn as_mut_owned(&mut self) -> Option<&mut [u8]> {
        match self {
            Self::Owned(v) => Some(v.as_mut_slice()),
            Self::Shared(_) => None,
        }
    }

    /// Promote a `Shared` buffer to `Owned` by copying. **Materialises a
    /// full-buffer allocation** — only call when copy-on-write is
    /// intentional.  No-op for buffers already `Owned`.
    pub fn make_owned(&mut self) {
        if let Self::Shared(s) = self {
            *self = Self::Owned(s.to_vec());
        }
    }

    /// Mutable slice; if the variant is `Shared`, this performs a full
    /// copy via [`FrameBuffer::make_owned`].
    ///
    /// # Warning
    ///
    /// **Calling this on a `Shared` buffer materialises a full-buffer
    /// copy.** This is fine for offline/test code but unsuitable for the
    /// realtime hot path.  Prefer [`FrameBuffer::as_mut_owned`] paired
    /// with an explicit [`FrameBuffer::make_owned`] when the CoW is
    /// genuinely required — that makes the allocation visible at the
    /// call site instead of hiding it behind a slice access.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.make_owned();
        let Self::Owned(v) = self else {
            unreachable!("just promoted to Owned via make_owned")
        };
        v.as_mut_slice()
    }
}

/// Per-frame metadata distinct from pixel data.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameMeta {
    /// Sequence number assigned by the capture stage.
    pub sequence: u64,
    /// Pipeline-monotonic timestamp at which the frame entered processing.
    pub timestamp: Timestamp,
    /// Capture timestamp from the source, if the backend exposes one.
    /// Distinct from `timestamp` to preserve source-clock information for
    /// latency accounting.
    pub source_timestamp: Option<Timestamp>,
    /// Nominal frame duration, if known from negotiated framerate.
    pub duration: Option<Duration>,
}

/// Row strides in bytes.
///
/// Packed formats carry a single byte-count; planar formats carry one
/// stride per plane (e.g. `Nv12` → 2 entries: `Y` and `UV`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stride {
    /// Single packed plane (`Rgb`/`Rgba`/`Bgr`/`Yuy2`/`Gray8`).
    Packed(usize),
    /// Multiple planes (e.g. `Nv12` → 2, future `I420` → 3).
    Planar(Vec<usize>),
}

impl Stride {
    /// Stride of the first (or only) plane, in bytes.
    ///
    /// For `Packed` this is the single stored value; for `Planar` it is
    /// the first plane's stride, or `0` if the planar list is empty
    /// (which would itself be an invariant violation — see
    /// [`VideoFrame::validate`]).
    #[must_use]
    pub fn primary(&self) -> usize {
        match self {
            Self::Packed(s) => *s,
            Self::Planar(strides) => strides.first().copied().unwrap_or(0),
        }
    }

    /// Number of planes (1 for `Packed`, `strides.len()` for `Planar`).
    #[must_use]
    pub fn plane_count(&self) -> usize {
        match self {
            Self::Packed(_) => 1,
            Self::Planar(strides) => strides.len(),
        }
    }
}

/// A single video frame in the processing pipeline.
///
/// Holds pixel data, geometry, format, per-plane strides and metadata.
/// Fields are public for ergonomic effect authoring; this is intentional
/// for MVP and may tighten once the public surface stabilises.
///
/// # Invariants
///
/// Constructors in this module uphold the following invariants; if you
/// build a `VideoFrame` from untrusted input (deserialisation, FFI),
/// call [`VideoFrame::validate`] before using it:
///
/// - `width > 0` and `height > 0`.
/// - `stride.plane_count()` matches `format` (1 for packed formats,
///   2 for `Nv12`).
/// - `data.len() >= stride.primary() * height as usize` so the
///   primary plane is fully addressable.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Pixel payload (see [`FrameBuffer`]).
    pub data: FrameBuffer,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel layout (see [`PixelFormat`]).
    pub format: PixelFormat,
    /// Row strides per plane (see [`Stride`]).
    pub stride: Stride,
    /// Per-frame metadata.
    pub meta: FrameMeta,
}

impl VideoFrame {
    /// Construct a frame for a packed pixel format with tight (no-padding)
    /// strides.
    ///
    /// Returns `None` if:
    /// - the format is planar (`Nv12`) — callers must build such frames
    ///   explicitly with per-plane strides; or
    /// - `width * bytes_per_pixel` overflows `usize` on the host.
    #[must_use]
    pub fn new_packed(
        data: FrameBuffer,
        width: u32,
        height: u32,
        format: PixelFormat,
        meta: FrameMeta,
    ) -> Option<Self> {
        let bpp = format.bytes_per_pixel()?;
        let stride = (width as usize).checked_mul(bpp)?;
        Some(Self {
            data,
            width,
            height,
            format,
            stride: Stride::Packed(stride),
            meta,
        })
    }

    /// Verify documented invariants. Use this after constructing a frame
    /// from untrusted sources (deserialisation, FFI). Returns `Err` with a
    /// human-readable reason on failure.
    ///
    /// The check is intentionally cheap — it does not validate every
    /// plane of `Planar` strides against the format, only the primary
    /// plane and the plane count.  Deeper validation belongs in
    /// format-specific helpers.
    #[allow(clippy::result_unit_err)]
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.width == 0 || self.height == 0 {
            return Err("width and height must be > 0");
        }
        let expected_planes = match self.format {
            PixelFormat::Nv12 => 2,
            _ => 1,
        };
        if self.stride.plane_count() != expected_planes {
            return Err("stride plane count does not match format");
        }
        if self.data.len() < self.stride.primary() * self.height as usize {
            return Err("data buffer shorter than primary stride * height");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn shared_buffer_promotes_on_make_owned() {
        let arc: Arc<[u8]> = Arc::from(vec![1, 2, 3].into_boxed_slice());
        let mut buf = FrameBuffer::Shared(arc);
        assert!(buf.as_mut_owned().is_none());
        buf.make_owned();
        assert!(buf.as_mut_owned().is_some());
        assert!(matches!(buf, FrameBuffer::Owned(_)));
    }

    #[test]
    fn new_packed_rejects_planar_format() {
        let buf = FrameBuffer::Owned(vec![0; 100]);
        let frame = VideoFrame::new_packed(
            buf,
            10,
            10,
            PixelFormat::Nv12,
            FrameMeta::default(),
        );
        assert!(frame.is_none(), "Nv12 is planar; new_packed must return None");
    }

    #[test]
    fn validate_catches_dimension_mismatch() {
        let buf = FrameBuffer::Owned(vec![0; 5]);
        let frame = VideoFrame::new_packed(
            buf,
            100,
            100,
            PixelFormat::Rgb,
            FrameMeta::default(),
        )
        .expect("constructor ok");
        assert!(frame.validate().is_err(), "data too small for w*h*bpp");
    }
}
