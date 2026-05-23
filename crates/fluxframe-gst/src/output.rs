//! GStreamer output pipeline construction.
//!
//! Stage 1 supported `fakesink` (CI/tests, no display) and `autovideosink`
//! (manual glance verification).  Stage 2 adds the `v4l2sink` branch for
//! `v4l2loopback` virtual cameras.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use fluxframe_core::error::PipelineError;
use fluxframe_core::frame::{PixelFormat, VideoFrame};
use gstreamer::prelude::*;
use gstreamer_app::AppSrc;
use tracing::{debug, trace, warn};

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
    /// Publishes the stream as a PipeWire node via `pipewiresink`.
    /// PipeWire's buffer pool synchronises producer↔consumer access at
    /// the protocol level — unlike v4l2loopback there is no torn-buffer
    /// race for downstream applications.  Modern apps (Firefox/Chrome
    /// via xdg-desktop-portal-pipewire, OBS with PW backend, recent
    /// cheese) see this node as a camera; older purely-V4L2 apps need
    /// the `pipewire-v4l2` shim to bridge it.
    Pipewire {
        /// PipeWire node name advertised to consumers.  `None` lets
        /// `pipewiresink` pick the default.
        node_name: Option<String>,
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

/// Latest composite published by the effect chain.  The writer thread
/// re-publishes this at the configured `fps` so v4l2sink sees a steady
/// frame rate even when the chain stalls.
struct LatestComposite {
    /// Packed pixel data in whatever format the appsrc declares.
    /// `Arc<[u8]>` so cloning into the writer thread is a refcount bump
    /// rather than a per-frame buffer copy.
    bytes: Arc<[u8]>,
    /// Effect-chain sequence number — propagated for tracing only.
    seq: u64,
}

/// Output pipeline owning the GStreamer elements.
///
/// The processing worker calls [`OutputPipeline::push_frame`] for every
/// processed [`VideoFrame`]; this crate hides the `appsrc` plumbing.
///
/// The pipeline runs a dedicated *writer thread* that pushes the
/// latest composite into `appsrc` at `fps` cadence — independently of
/// how often the effect chain manages to produce a new one.  This
/// closes the read/write race window in v4l2loopback (consumer at
/// 30 fps reading while we wrote at ~15 fps produced visibly-torn
/// "half-old, half-new" frames).
pub struct OutputPipeline {
    pipeline: gstreamer::Pipeline,
    appsrc: AppSrc,
    started: Arc<AtomicBool>,
    /// Target write cadence in Hz.  Cached at build time so the writer
    /// thread can compute its sleep interval.  The actual writer ticks
    /// at `WRITER_TICK_MULTIPLIER × fps` so v4l2loopback's read/write
    /// race window shrinks proportionally — consumers reading at the
    /// nominal `fps` see torn frames roughly `1/MULTIPLIER` as often.
    fps: u32,
    /// Latest composite from the effect chain.  Cloned cheaply (the
    /// inner `Arc` is bumped, not the pixel bytes) by the writer thread
    /// every tick.
    latest: Arc<std::sync::Mutex<Option<LatestComposite>>>,
    /// Writer thread handle.  Joined on `stop`.
    writer_handle: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// Writer pushes the latest composite into appsrc this many times per
/// nominal output frame.  Repeated identical writes overwrite each
/// other in v4l2loopback's buffer pool with no extra cost to consumers
/// (each consumer DQBUF still gets the latest), and the higher rate
/// statistically narrows the window where a consumer DQBUF lands
/// mid-write — observably the dominant cause of the horizontal-seam
/// torn frames operators see at the nominal cadence.
///
/// Capped further down to ensure the writer never ticks faster than
/// `WRITER_MAX_HZ` regardless of operator-requested `fps`.
const WRITER_TICK_MULTIPLIER: u32 = 3;
const WRITER_MAX_HZ: u32 = 120;

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

        let sink_elem = build_sink_element(sink)?;
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
            fps: fps.max(1),
            latest: Arc::new(std::sync::Mutex::new(None)),
            writer_handle: std::sync::Mutex::new(None),
        })
    }

    /// Transition the pipeline to PLAYING and spawn the writer thread
    /// that re-publishes the latest composite at `fps` cadence.
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

        // Spawn the writer thread.  It owns its own monotonic PTS
        // counter so consumers see strictly-monotonic timestamps with
        // no drift even when the supervisor stalls the effect chain.
        let started = Arc::clone(&self.started);
        let latest = Arc::clone(&self.latest);
        let appsrc = self.appsrc.clone();
        let writer_hz = (self.fps.saturating_mul(WRITER_TICK_MULTIPLIER)).min(WRITER_MAX_HZ);
        let interval = Duration::from_nanos(1_000_000_000_u64 / u64::from(writer_hz));
        debug!(
            sink_fps = self.fps,
            writer_hz, "spawning output writer thread"
        );
        let handle = std::thread::Builder::new()
            .name("fluxframe-out-writer".into())
            .spawn(move || writer_loop(&started, &latest, &appsrc, interval))
            .map_err(|e| PipelineError::Runtime {
                reason: format!("writer thread spawn failed: {e}"),
            })?;
        *self.writer_handle.lock().expect("writer_handle poisoned") = Some(handle);
        Ok(())
    }

    /// Publish a processed frame as the latest composite.  The writer
    /// thread reads this slot at every tick; if a new frame has not
    /// arrived since the last tick the previous one is re-pushed so
    /// downstream sees a steady framerate.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Runtime`] when called before `start`.
    pub fn push_frame(&self, frame: VideoFrame) -> Result<(), PipelineError> {
        if !self.started.load(Ordering::Acquire) {
            return Err(PipelineError::Runtime {
                reason: "output pipeline not started".into(),
            });
        }
        // Pull pixel bytes into an `Arc<Vec<u8>>` so handing the slot to
        // the writer is a refcount bump, not a buffer copy.  We
        // discard `frame`'s PTS / duration deliberately: the writer
        // synthesises its own clock to keep cadence steady.
        let seq = frame.meta.sequence;
        let bytes: Arc<[u8]> = match frame.data {
            fluxframe_core::frame::FrameBuffer::Owned(v) => Arc::from(v.into_boxed_slice()),
            fluxframe_core::frame::FrameBuffer::Shared(arc) => arc,
        };
        let composite = LatestComposite { bytes, seq };
        *self.latest.lock().expect("latest poisoned") = Some(composite);
        trace!(seq, "published composite to writer thread");
        Ok(())
    }

    /// Stop the pipeline; signals EOS to the sink, joins the writer
    /// thread and tears down.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::StateChangeFailed`] on teardown failure.
    pub fn stop(&self) -> Result<(), PipelineError> {
        if self.started.swap(false, Ordering::AcqRel) {
            // End-of-stream so downstream drains cleanly.
            let _ = self.appsrc.end_of_stream();
        }
        if let Some(handle) = self
            .writer_handle
            .lock()
            .expect("writer_handle poisoned")
            .take()
            && let Err(e) = handle.join()
        {
            warn!(?e, "writer thread panicked while joining");
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

/// Construct the terminal sink element for the requested
/// [`OutputSink`].  Extracted from [`OutputPipeline::build`] to keep
/// the latter at a digestible size and to centralise the sink-specific
/// quirks (loopback access checks, PipeWire client naming, …).
fn build_sink_element(sink: OutputSink) -> Result<gstreamer::Element, PipelineError> {
    match sink {
        OutputSink::Fake => make_element("fakesink", "output_sink"),
        OutputSink::Auto => make_element("autovideosink", "output_sink"),
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
            Ok(elem)
        }
        OutputSink::Pipewire { node_name } => {
            // `pipewiresink` is provided by `gst-plugin-pipewire`.
            // `make_element` surfaces a structured MissingElement
            // error (with install hint) if the plugin is missing.
            let elem = make_element("pipewiresink", "output_sink")?;
            // Advertise this stream as a video *source* in the PipeWire
            // graph so consumers (Firefox via xdg-desktop-portal,
            // OBS-PW, helvum) discover it as a camera.  Without
            // `media.class = Video/Source` pipewiresink defaults to
            // looking for a target consumer and exits with
            // "no target node available" — exactly the smoke-test
            // failure mode operators first hit on Stage 5.
            let name = node_name.as_deref().unwrap_or("fluxframe");
            let props = gstreamer::Structure::builder("properties")
                .field("media.class", "Video/Source")
                .field("media.role", "Camera")
                .field("node.name", name)
                .field("node.description", "FluxFrame Camera")
                .build();
            elem.set_property("stream-properties", props);
            elem.set_property("client-name", name);
            Ok(elem)
        }
    }
}

/// Writer thread loop.  Wakes every `interval`, takes a cheap `Arc`
/// clone of the latest composite (or the previous one if the chain
/// has not delivered a new frame in time), wraps it in a fresh
/// `gst::Buffer` with a fabricated monotonic PTS and pushes it to
/// `appsrc`.  Exits when `started` flips to `false`.
fn writer_loop(
    started: &Arc<AtomicBool>,
    latest: &Arc<std::sync::Mutex<Option<LatestComposite>>>,
    appsrc: &AppSrc,
    interval: Duration,
) {
    let start_instant = Instant::now();
    let mut next_tick = start_instant + interval;
    let mut last_pushed_seq: Option<u64> = None;
    let mut frames_pushed: u64 = 0;
    let mut frames_duplicated: u64 = 0;
    while started.load(Ordering::Acquire) {
        let now = Instant::now();
        if now < next_tick {
            std::thread::sleep(next_tick - now);
        } else {
            // We are late.  Skip past missed ticks rather than burst the
            // entire backlog into appsrc (which would defeat the point
            // of a steady cadence).
            while next_tick < now {
                next_tick += interval;
            }
        }
        next_tick += interval;

        // Snapshot the latest composite.  Cloning the `Arc<Vec<u8>>` is
        // a single refcount bump regardless of buffer size.
        let snapshot = latest
            .lock()
            .expect("latest poisoned")
            .as_ref()
            .map(|c| (Arc::clone(&c.bytes), c.seq));
        let Some((bytes, seq)) = snapshot else {
            // Effect chain hasn't produced anything yet — nothing to
            // push.  Cheap idle.
            continue;
        };
        let is_dup = last_pushed_seq == Some(seq);
        if is_dup {
            frames_duplicated += 1;
        }
        last_pushed_seq = Some(seq);

        // Build the gst::Buffer with a clock-derived PTS.  We deliberately
        // synthesise PTS here (rather than reuse the supervisor's frame
        // timestamp) so v4l2sink sees a strictly-monotonic 1/fps cadence
        // even when the chain produces frames in bursts.
        let pts_ns = u64::try_from(
            next_tick
                .saturating_duration_since(start_instant)
                .as_nanos(),
        )
        .unwrap_or(u64::MAX);
        match build_buffer(&bytes, pts_ns, interval) {
            Ok(buf) => {
                if let Err(e) = appsrc.push_buffer(buf) {
                    // The most common failure here is the shutdown race —
                    // `stop()` sent EOS but the writer's loop iteration
                    // was already past the `started` check.  Stay silent
                    // in that case; surface anything else.
                    if started.load(Ordering::Acquire) {
                        warn!(error = %e, seq, "writer: appsrc.push_buffer failed");
                    }
                }
                frames_pushed += 1;
            }
            Err(e) => {
                warn!(error = %e, seq, "writer: buffer build failed");
            }
        }
    }
    debug!(
        frames_pushed,
        frames_duplicated, "output writer thread exited"
    );
}

/// Wrap the latest composite bytes into a fresh `gst::Buffer` ready
/// for `appsrc.push_buffer`.  PTS is supplied by the caller (writer
/// thread) so consumers see a strictly-monotonic clock.
fn build_buffer(
    bytes: &[u8],
    pts_ns: u64,
    duration: Duration,
) -> Result<gstreamer::Buffer, PipelineError> {
    let mut buffer =
        gstreamer::Buffer::with_size(bytes.len()).map_err(|e| PipelineError::Runtime {
            reason: format!("Buffer::with_size failed: {e}"),
        })?;
    {
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
        buffer_ref.set_pts(gstreamer::ClockTime::from_nseconds(pts_ns));
        if let Ok(d_ns) = u64::try_from(duration.as_nanos()) {
            buffer_ref.set_duration(gstreamer::ClockTime::from_nseconds(d_ns));
        }
    }
    Ok(buffer)
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
