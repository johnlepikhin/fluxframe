//! Framing stage: cuts the processed frame out of the captured one.
//!
//! The runtime captures at `capture_width × capture_height` and runs the
//! effect chain at the (usually smaller) processing size.  Each frame
//! the stage takes a region of the captured frame — the whole view, or
//! the window `auto_frame` asked for — and scales it to the processing
//! size in one bilinear pass.  Effects therefore cost what the
//! processing size costs, while a zoomed-in window still draws on the
//! full capture detail.

use fluxframe_core::frame::{FrameBuffer, PixelFormat, VideoFrame};
use fluxframe_core::{EffectError, Region};

use super::resize::resize_rgb_bilinear_region;

/// Tolerance, in captured pixels, under which a region counts as the
/// full view (so no resampling is needed when sizes also match).
const FULL_VIEW_TOLERANCE: f32 = 0.01;

/// Converts captured frames to processed frames.
#[derive(Debug, Clone, Copy)]
pub struct Framer {
    capture_w: u32,
    capture_h: u32,
    out_w: u32,
    out_h: u32,
    full: Region,
}

impl Framer {
    /// Stage for `capture_w × capture_h` frames processed at
    /// `out_w × out_h`.
    #[must_use]
    pub fn new(capture_w: u32, capture_h: u32, out_w: u32, out_h: u32) -> Self {
        let aspect = out_w as f32 / out_h.max(1) as f32;
        Self {
            capture_w,
            capture_h,
            out_w,
            out_h,
            full: Region::centered_with_aspect(capture_w, capture_h, aspect),
        }
    }

    /// The whole view: the largest region of the captured frame with the
    /// processed frame's aspect ratio, centred.  Shown when nothing asks
    /// for a crop.
    #[must_use]
    pub fn full_view(&self) -> Region {
        self.full
    }

    /// Captured frame size.
    #[must_use]
    pub fn capture_size(&self) -> (u32, u32) {
        (self.capture_w, self.capture_h)
    }

    /// Whether showing `region` needs no resampling: it is the full
    /// view and the sizes match, so the captured frame is used as is.
    #[must_use]
    pub fn is_identity(&self, region: &Region) -> bool {
        (self.capture_w, self.capture_h) == (self.out_w, self.out_h)
            && region.approx_eq(&Region::full(self.out_w, self.out_h), FULL_VIEW_TOLERANCE)
    }

    /// Cut `region` out of `frame` and scale it to the processing size.
    /// Frame metadata is carried over.  Returns `frame` untouched when
    /// [`Self::is_identity`] holds.
    ///
    /// # Errors
    ///
    /// [`EffectError::ProcessFailed`] when `frame` is not a packed RGB
    /// frame of the configured capture size.
    pub fn apply(&self, frame: VideoFrame, region: &Region) -> Result<VideoFrame, EffectError> {
        if (frame.width, frame.height) != (self.capture_w, self.capture_h) {
            return Err(framing_err(format!(
                "captured frame {}x{} differs from the configured capture size {}x{}",
                frame.width, frame.height, self.capture_w, self.capture_h
            )));
        }
        if self.is_identity(region) {
            return Ok(frame);
        }
        if frame.format != PixelFormat::Rgb {
            return Err(framing_err(format!(
                "cropping needs RGB frames, got {:?}",
                frame.format
            )));
        }
        let row = self.capture_w as usize * 3;
        if frame.stride.primary() != row {
            return Err(framing_err(format!(
                "captured frame has padded rows (stride {} for {} bytes of pixels)",
                frame.stride.primary(),
                row
            )));
        }
        let src = &frame.data.as_slice()[..row * self.capture_h as usize];
        let mut out = vec![0u8; self.out_w as usize * self.out_h as usize * 3];
        resize_rgb_bilinear_region(
            src,
            self.capture_w,
            self.capture_h,
            region,
            &mut out,
            self.out_w,
            self.out_h,
        );
        VideoFrame::new_packed(
            FrameBuffer::Owned(out),
            self.out_w,
            self.out_h,
            PixelFormat::Rgb,
            frame.meta,
        )
        .ok_or_else(|| framing_err("processed frame size overflows".into()))
    }
}

fn framing_err(reason: String) -> EffectError {
    EffectError::ProcessFailed {
        name: "framing".into(),
        reason,
    }
}

#[cfg(test)]
mod tests {
    use fluxframe_core::frame::FrameMeta;

    use super::*;

    fn frame(w: u32, h: u32) -> VideoFrame {
        // Horizontal ramp: value = x (mod 256) in every channel.
        let data: Vec<u8> = (0..h)
            .flat_map(|_| (0..w).flat_map(|x| [x as u8; 3]))
            .collect();
        VideoFrame::new_packed(
            FrameBuffer::Owned(data),
            w,
            h,
            PixelFormat::Rgb,
            FrameMeta::default(),
        )
        .expect("frame")
    }

    #[test]
    fn same_size_full_view_is_passed_through() {
        let framer = Framer::new(64, 36, 64, 36);
        let input = frame(64, 36);
        let expected = input.data.as_slice().to_vec();
        let out = framer.apply(input, &framer.full_view()).expect("ok");
        assert_eq!(out.data.as_slice(), expected.as_slice());
    }

    #[test]
    fn full_view_downscales_to_processing_size() {
        let framer = Framer::new(128, 72, 64, 36);
        let out = framer
            .apply(frame(128, 72), &framer.full_view())
            .expect("ok");
        assert_eq!((out.width, out.height), (64, 36));
        assert_eq!(out.data.len(), 64 * 36 * 3);
        // 2× box average of the ramp: column x ← (2x + 2x+1) / 2.
        assert!((i32::from(out.data.as_slice()[3 * 10]) - 20).abs() <= 1);
    }

    #[test]
    fn full_view_crops_mismatched_aspect() {
        // 4:3 capture, 16:9 processing → centred band, no letterbox.
        let framer = Framer::new(128, 96, 64, 36);
        let full = framer.full_view();
        assert!((full.width - 128.0).abs() < 1e-3);
        assert!((full.height - 72.0).abs() < 1e-3);
        assert!((full.y - 12.0).abs() < 1e-3);
    }

    #[test]
    fn region_is_cut_out_and_scaled() {
        let framer = Framer::new(128, 72, 64, 36);
        let region = Region {
            x: 32.0,
            y: 18.0,
            width: 64.0,
            height: 36.0,
        };
        let out = framer.apply(frame(128, 72), &region).expect("ok");
        // 1:1 crop starting at column 32.
        assert_eq!(out.data.as_slice()[0], 32);
    }

    #[test]
    fn wrong_capture_size_is_rejected() {
        let framer = Framer::new(128, 72, 64, 36);
        assert!(framer.apply(frame(64, 36), &framer.full_view()).is_err());
    }
}
