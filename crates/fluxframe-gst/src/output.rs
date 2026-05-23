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
use tracing::{trace, warn};

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
    /// Pixel format **as produced by the effect chain** — what
    /// `appsrc.set_caps` advertises and what [`OutputPipeline::push_frame`]
    /// expects in the byte buffer.  Lying here (declaring `Yuy2` while
    /// pushing `Rgb` bytes) makes downstream `videoconvert` skip the
    /// conversion and the sink interprets the raw bytes through the wrong
    /// pixel layout — manifests as horizontally-doubled green frames in
    /// v4l2loopback consumers.
    pub format: PixelFormat,
    /// Optional sink-side format pinned via a downstream `capsfilter`.
    /// When `Some`, `videoconvert` is forced to convert `format` →
    /// `sink_format` before the bytes reach the sink (useful when the
    /// effect chain runs in `Rgb` but the v4l2loopback consumer expects
    /// `Yuy2`/`Nv12`).  When `None`, the sink negotiates with whatever
    /// `format` declares.
    pub sink_format: Option<PixelFormat>,
    /// Choice of terminal sink element.
    pub sink: OutputSink,
}

impl OutputParams {
    /// Construct a fully-specified [`OutputParams`].
    ///
    /// `sink_format` defaults to `None` (passthrough — sink consumes
    /// whatever `format` declares).  Use
    /// [`OutputParams::with_sink_format`] to force a colour conversion
    /// between the effect chain and the sink.
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
            sink_format: None,
            sink,
        }
    }

    /// Pin a sink-side format that differs from the effect-chain format.
    /// Inserts a `capsfilter` after `videoconvert`/`videoscale` so the
    /// conversion actually happens.
    #[must_use]
    pub fn with_sink_format(mut self, sink_format: PixelFormat) -> Self {
        self.sink_format = Some(sink_format);
        self
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
    /// Last buffer PTS pushed into `appsrc` (nanoseconds).  Used only by
    /// `push_frame` to warn on non-monotonic timestamps — v4l2sink and
    /// v4l2loopback are picky about clock continuity, and a single
    /// out-of-order / zero PTS shows up as a one-frame visual glitch
    /// ("flicker") in downstream consumers.
    last_pts_ns: std::sync::Mutex<Option<u64>>,
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
            sink_format,
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
                // NOTE: do NOT set `io-mode=rw`.  v4l2loopback with
                // `exclusive_caps=1` only exposes the device's Video
                // Capture capability to consumers when the writer uses
                // mmap+QBUF.  `rw` puts v4l2sink on `write(2)` which
                // keeps the device in Video Output mode only — cheese,
                // Chrome and gst-launch consumers stop seeing it as a
                // camera entirely.
                elem
            }
        };
        sink_elem.set_property("sync", false);

        // When the operator requested a sink-side format different from
        // what the effect chain produces, pin it via a `capsfilter` so
        // `videoconvert` actually performs the conversion.  Only the
        // pixel format gets pinned — width/height/framerate stay
        // flexible so v4l2sink negotiates them with the device.
        // Over-pinning was the previous failure mode (`not-negotiated`
        // at PLAYING).
        let sink_capsfilter = sink_format
            .filter(|sf| *sf != format)
            .map(|sf| -> Result<gstreamer::Element, PipelineError> {
                let cf = make_element("capsfilter", "output_sink_caps")?;
                let gst_fmt = crate::frame_conv::pixel_format_to_gst(sf);
                let caps = gstreamer::Caps::builder("video/x-raw")
                    .field("format", gst_fmt.to_str())
                    .build();
                cf.set_property("caps", caps);
                Ok(cf)
            })
            .transpose()?;

        let mut elements: Vec<&gstreamer::Element> =
            vec![&appsrc_elem, &queue, &videoconvert, &videoscale];
        if let Some(cf) = &sink_capsfilter {
            elements.push(cf);
        }
        elements.push(&sink_elem);

        pipeline
            .add_many(elements.iter().copied())
            .map_err(|e| PipelineError::Runtime {
                reason: format!("pipeline.add_many failed: {e}"),
            })?;

        gstreamer::Element::link_many(elements.iter().copied()).map_err(|e| {
            PipelineError::Runtime {
                reason: format!("element link failed: {e}"),
            }
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

        // Probe sink-pad CAPS events.  The first one is the initial
        // negotiation (logged at debug); any subsequent CAPS event means
        // a mid-stream renegotiation, which historically correlates with
        // single-frame stride shifts in the rendered output and is worth
        // a warn so it's tied to a concrete moment in the operator's
        // log.  Segment/flush events are pipeline-lifecycle noise and
        // are not probed.
        if let Some(sink_pad) = sink_elem.static_pad("sink") {
            let caps_seen = Arc::new(AtomicBool::new(false));
            sink_pad.add_probe(gstreamer::PadProbeType::EVENT_DOWNSTREAM, move |_pad, info| {
                if let Some(gstreamer::PadProbeData::Event(ref ev)) = info.data
                    && let gstreamer::EventView::Caps(c) = ev.view()
                {
                    if caps_seen.swap(true, Ordering::Relaxed) {
                        warn!(caps = %c.caps(), "sink pad CAPS event after initial negotiation (mid-stream renegotiation)");
                    } else {
                        tracing::debug!(caps = %c.caps(), "sink pad initial CAPS");
                    }
                }
                gstreamer::PadProbeReturn::Ok
            });
        } else {
            warn!("sink element has no `sink` pad — caps probe not installed");
        }

        Ok(Self {
            pipeline,
            appsrc,
            started: Arc::new(AtomicBool::new(false)),
            last_pts_ns: std::sync::Mutex::new(None),
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
        let pts_ns = frame.meta.timestamp.as_nanos();
        // Monotonicity probe: v4l2sink and v4l2loopback rely on PTS being
        // strictly increasing.  A zero PTS, a duplicate, or an
        // out-of-order one each manifests as a single-frame visual
        // glitch in downstream consumers (cheese, Chrome, OBS).
        {
            let mut guard = self.last_pts_ns.lock().expect("last_pts_ns poisoned");
            if pts_ns == 0 {
                warn!(seq, "outgoing frame has PTS=0 — downstream may glitch");
            } else if let Some(prev) = *guard
                && pts_ns <= prev
            {
                warn!(
                    seq,
                    pts_ns,
                    prev_pts_ns = prev,
                    "outgoing PTS is non-monotonic — downstream may glitch (one-frame flicker)"
                );
            }
            *guard = Some(pts_ns);
        }
        let buffer = frame_to_buffer(frame)?;
        trace!(seq, pts_ns, "pushing frame to appsrc");
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
