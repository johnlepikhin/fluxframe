//! GStreamer output pipeline construction.
//!
//! Stage 1 supported `fakesink` (CI/tests, no display) and `autovideosink`
//! (manual glance verification).  Stage 2 adds the `v4l2sink` branch for
//! `v4l2loopback` virtual cameras.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use fluxframe_core::error::PipelineError;
use fluxframe_core::frame::{PixelFormat, VideoFrame};
use gstreamer::prelude::*;
use gstreamer_app::AppSrc;
use tracing::trace;

use crate::frame_conv::frame_to_buffer;
use crate::util::{build_caps, check_v4l2_output_access, make_element};

/// Output sink selection.
///
/// Intentionally *not* `#[non_exhaustive]`: this crate is workspace-internal
/// with a single version, so adding a variant should produce a compile-time
/// prompt at every `match` site rather than a silent wildcard fall-through.
///
/// No longer `Copy` after Stage 2: `V4l2Loopback` carries an owned
/// [`PathBuf`].  Use [`Clone`] explicitly where needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputSink {
    /// Discards buffers immediately — for tests and CI.
    Fake,
    /// Picks a platform display sink (Wayland/X11) — for manual glance
    /// verification.
    Auto,
    /// Writes to a `v4l2loopback` virtual camera device.
    V4l2Loopback {
        /// Loopback device path (e.g. `/dev/video10`).
        device: PathBuf,
    },
}

/// Negotiated output parameters.
///
/// `#[non_exhaustive]` so adding optional fields (e.g. v4l2loopback device
/// path) is not a breaking change for downstream crates.
///
/// No longer `Copy` after Stage 2 because [`OutputSink`] now carries an
/// owned [`PathBuf`] for the `V4l2Loopback` variant; only [`Clone`] is
/// derived.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OutputParams {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate (frames per second).
    pub fps: u32,
    /// Pixel format produced by the processing chain.
    pub format: PixelFormat,
    /// Choice of terminal sink element.
    pub sink: OutputSink,
}

impl OutputParams {
    /// Construct a fully-specified [`OutputParams`].
    ///
    /// Provided because the struct is `#[non_exhaustive]`, so cross-crate
    /// callers cannot use the record literal syntax.  Inside this crate
    /// the literal still works.
    #[must_use]
    pub fn new(width: u32, height: u32, fps: u32, format: PixelFormat, sink: OutputSink) -> Self {
        Self {
            width,
            height,
            fps,
            format,
            sink,
        }
    }
}

/// Output pipeline owning the GStreamer elements.
///
/// The processing worker calls [`OutputPipeline::push_frame`] for every
/// processed [`VideoFrame`]; this crate hides the `appsrc` plumbing.
pub struct OutputPipeline {
    pipeline: gstreamer::Pipeline,
    appsrc: AppSrc,
    started: Arc<AtomicBool>,
}

impl OutputPipeline {
    /// Build the pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] if any element cannot be instantiated
    /// or linking fails.
    pub fn build(params: OutputParams) -> Result<Self, PipelineError> {
        // Destructure up-front so the `sink` variant (which owns a PathBuf
        // for V4l2Loopback) can move into the match arm without forcing
        // clippy's `needless_pass_by_value` lint on the public signature.
        let OutputParams {
            width,
            height,
            fps,
            format,
            sink,
            ..
        } = params;

        let pipeline = gstreamer::Pipeline::with_name("fluxframe-output");

        let appsrc_elem = make_element("appsrc", "output_src")?;
        appsrc_elem.set_property("is-live", true);
        appsrc_elem.set_property("do-timestamp", false);
        appsrc_elem.set_property("block", false);
        appsrc_elem.set_property_from_str("format", "time");

        let queue = make_element("queue", "output_queue")?;
        queue.set_property("max-size-buffers", 1u32);
        queue.set_property_from_str("leaky", "downstream");
        queue.set_property("max-size-bytes", 0u32);
        queue.set_property("max-size-time", 0u64);

        let videoconvert = make_element("videoconvert", "output_videoconvert")?;
        let videoscale = make_element("videoscale", "output_videoscale")?;

        let sink_elem = match sink {
            OutputSink::Fake => make_element("fakesink", "output_sink")?,
            OutputSink::Auto => make_element("autovideosink", "output_sink")?,
            OutputSink::V4l2Loopback { device } => {
                // Pre-open write check so EACCES/EBUSY/ENOENT surface with a
                // hint *before* `v4l2sink` returns an opaque GStreamer error.
                // The returned path is canonicalised — feed that to v4l2sink
                // rather than the user-supplied original to close the symlink
                // race window between the pre-check and the kernel open.
                let canon = check_v4l2_output_access(&device)?;
                let elem = make_element("v4l2sink", "output_sink")?;
                elem.set_property_from_str("device", canon.to_string_lossy().as_ref());
                elem
            }
        };
        sink_elem.set_property("sync", false);

        pipeline
            .add_many([&appsrc_elem, &queue, &videoconvert, &videoscale, &sink_elem])
            .map_err(|e| PipelineError::Runtime {
                reason: format!("pipeline.add_many failed: {e}"),
            })?;

        gstreamer::Element::link_many([
            &appsrc_elem,
            &queue,
            &videoconvert,
            &videoscale,
            &sink_elem,
        ])
        .map_err(|e| PipelineError::Runtime {
            reason: format!("element link failed: {e}"),
        })?;

        let appsrc = appsrc_elem
            .dynamic_cast::<AppSrc>()
            .map_err(|_| PipelineError::Runtime {
                reason: "output_src is not an AppSrc".into(),
            })?;

        // Negotiate caps on the appsrc so downstream knows what to expect.
        let caps = build_caps(width, height, fps, format)?;
        appsrc.set_caps(Some(&caps));
        appsrc.set_format(gstreamer::Format::Time);

        Ok(Self {
            pipeline,
            appsrc,
            started: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Transition the pipeline to PLAYING.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::StateChangeFailed`] if the state change
    /// is rejected.
    pub fn start(&self) -> Result<(), PipelineError> {
        self.pipeline
            .set_state(gstreamer::State::Playing)
            .map(|_| ())
            .map_err(|e| PipelineError::StateChangeFailed {
                reason: format!("set_state(Playing) failed: {e}"),
            })?;
        self.started.store(true, Ordering::Release);
        Ok(())
    }

    /// Push one processed frame into the pipeline.
    ///
    /// Takes the frame by value as a forward-compatibility hook for the
    /// Stage 5 zero-copy path: the frame's backing memory will need to
    /// transfer into the `gst::Buffer` so the mapping outlives this call.
    /// Today the body still copies the pixel data regardless.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] if buffer construction fails or the
    /// `appsrc` rejects the push (e.g. pipeline closed).
    pub fn push_frame(&self, frame: VideoFrame) -> Result<(), PipelineError> {
        if !self.started.load(Ordering::Acquire) {
            return Err(PipelineError::Runtime {
                reason: "output pipeline not started".into(),
            });
        }
        let seq = frame.meta.sequence;
        let buffer = frame_to_buffer(frame)?;
        trace!(seq, "pushing frame to appsrc");
        self.appsrc
            .push_buffer(buffer)
            .map(|_| ())
            .map_err(|e| PipelineError::Runtime {
                reason: format!("appsrc.push_buffer failed: {e}"),
            })
    }

    /// Stop the pipeline; signals EOS to the sink and tears down.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::StateChangeFailed`] on teardown failure.
    pub fn stop(&self) -> Result<(), PipelineError> {
        if self.started.swap(false, Ordering::AcqRel) {
            // End-of-stream so downstream drains cleanly.
            let _ = self.appsrc.end_of_stream();
        }
        self.pipeline
            .set_state(gstreamer::State::Null)
            .map(|_| ())
            .map_err(|e| PipelineError::StateChangeFailed {
                reason: format!("set_state(Null) failed: {e}"),
            })
    }

    /// Borrow the pipeline's bus for use with [`crate::bus::BusListener`].
    ///
    /// The bus is the only piece of the underlying GStreamer pipeline that
    /// the supervisor needs visibility into; exposing it instead of the
    /// whole `gstreamer::Pipeline` keeps the GStreamer surface area at this
    /// crate's boundary as small as possible.
    #[must_use]
    pub fn bus(&self) -> gstreamer::Bus {
        self.pipeline.bus().expect("pipelines always have a bus")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_sink_v4l2loopback_is_clone() {
        let sink = OutputSink::V4l2Loopback {
            device: PathBuf::from("/dev/video10"),
        };
        let cloned = sink.clone();
        assert!(matches!(cloned, OutputSink::V4l2Loopback { .. }));
    }

    #[test]
    fn output_params_carry_v4l2_device() {
        let params = OutputParams::new(
            640,
            480,
            30,
            PixelFormat::Yuy2,
            OutputSink::V4l2Loopback {
                device: PathBuf::from("/dev/video10"),
            },
        );
        match &params.sink {
            OutputSink::V4l2Loopback { device } => {
                assert_eq!(device, &PathBuf::from("/dev/video10"));
            }
            other => panic!("expected V4l2Loopback, got {other:?}"),
        }
        // `OutputParams` no longer implements `Copy`; ensure it is still
        // `Clone` for the runtime layer that may dup-and-pass it.
        let _cloned = params.clone();
    }
}
