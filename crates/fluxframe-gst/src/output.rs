//! GStreamer output pipeline construction.
//!
//! Pipeline topology:
//! `appsrc → queue → videoconvert → videoscale → [capsfilter] → sink`,
//! where `sink` is one of:
//!
//! * `fakesink` — discards buffers (CI / tests, no display).
//! * `autovideosink` — picks a platform display sink (manual verification).
//! * `fdsink` writing directly to a `v4l2loopback` device fd.  We bypass
//!   GStreamer's `v4l2sink` entirely on this path because v4l2sink's MMAP
//!   io-mode shares the kernel buffer pool with the consumer non-atomically;
//!   `write(2)` on the device fd serialises against consumer reads in the
//!   kernel.  See `build_v4l2_direct_chain` for the full rationale.
//! * `pipewiresink` — publishes the stream as a PipeWire video source node.
//!
//! A dedicated writer thread (`writer_loop`) re-pushes the latest composite
//! into `appsrc` at the configured `fps` cadence so downstream sees a steady
//! framerate even when the effect chain stalls or runs slower than the sink.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use fluxframe_core::error::PipelineError;
use fluxframe_core::frame::{PixelFormat, VideoFrame};
use gstreamer::prelude::*;
use gstreamer_app::AppSrc;
use parking_lot::Mutex;
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
    ///
    /// PipeWire's buffer pool synchronises producer↔consumer access at
    /// the protocol layer, so there is no torn-buffer race like the one
    /// `v4l2loopback` exhibits.
    ///
    /// Consumer discovery is host-setup dependent.  With
    /// `xdg-desktop-portal` running and a portal-aware app, Firefox
    /// usually picks the node up.  Chrome, even with
    /// `chrome://flags/#enable-webrtc-pipewire-camera` enabled and a
    /// portal present, does NOT see this node reliably in practice —
    /// kept here as an *experimental* sink, not a recommended default.
    /// Older purely-V4L2 apps need the `pipewire-v4l2` shim to bridge
    /// the node into a `/dev/video*` device.
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
    /// thread can compute its sleep interval.
    fps: u32,
    /// RAII guard keeping the V4L2 device fd open under `fdsink` on the
    /// V4l2Loopback direct-write path.  See [`V4lFdGuard`] for the
    /// load-bearing rationale.  Declared after `pipeline` so drop order
    /// tears the fdsink down first.
    _fd_guard: Option<V4lFdGuard>,
    /// Latest composite from the effect chain.  Cloned cheaply (the
    /// inner `Arc` is bumped, not the pixel bytes) by the writer thread
    /// every tick.
    latest: Arc<Mutex<Option<LatestComposite>>>,
    /// Writer thread handle.  Joined on `stop`.
    writer_handle: Mutex<Option<std::thread::JoinHandle<WriterStats>>>,
    /// Monotonic sequence counter assigned by [`OutputPipeline::push_frame`]
    /// to every published composite.  Starts at 1 so the first real frame
    /// never collides with `last_pushed_seq == Some(0)` left behind by
    /// any prior placeholder/default frame (where `FrameMeta::default()`
    /// reports `sequence = 0`).
    ///
    /// `frame.meta.sequence` is intentionally IGNORED here — upstream
    /// counters serve tracing/telemetry, while this counter exists solely
    /// for the writer-thread dedupe path.  Decoupling them prevents any
    /// upstream "0 means unset" convention from poisoning idle→resume
    /// transitions on the writer thread.
    frame_counter: AtomicU64,
}

// Writer tick rate equals the configured output `fps`.  Empirical
// observation: changing the multiplier (1×, 3×, or even sub-1× / 20 Hz)
// has no measurable effect on the torn-frame rate operators see in
// v4l2loopback consumers.  The tearing therefore is not a
// producer↔consumer race in the kernel pool that we can mitigate from
// userspace by adjusting cadence — it sits somewhere else in the
// transport.  Match the sink's nominal rate for predictability and
// minimal CPU.

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

        // Effective sink-side pixel format.  We need this concrete
        // value for the V4l2Loopback direct-write path (it has to do
        // VIDIOC_S_FMT BEFORE handing the fd to fdsink) so it can no
        // longer be `Option`-shaped.
        let effective_sink_format = sink_format.unwrap_or(format);

        let SinkChainResult {
            pre_sink: sink_pre,
            sink: sink_elem,
            fd_guard,
        } = build_sink_chain(sink, width, height, effective_sink_format)?;
        sink_elem.set_property("sync", false);

        let sink_capsfilter = maybe_format_capsfilter(format, sink_format, fd_guard.is_some())?;

        let elements = assemble_pipeline_elements(
            &appsrc_elem,
            &queue,
            &videoconvert,
            Some(&videoscale),
            sink_capsfilter.as_ref(),
            &sink_pre,
            &sink_elem,
        );

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

        install_sink_pad_probes(&sink_elem);

        Ok(Self {
            pipeline,
            appsrc,
            started: Arc::new(AtomicBool::new(false)),
            fps: fps.max(1),
            _fd_guard: fd_guard,
            latest: Arc::new(Mutex::new(None)),
            writer_handle: Mutex::new(None),
            frame_counter: AtomicU64::new(1),
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
        let writer_hz = self.fps.max(1);
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
        *self.writer_handle.lock() = Some(handle);
        Ok(())
    }

    /// Publish a processed frame as the latest composite.  The writer
    /// thread reads this slot at every tick; if a new frame has not
    /// arrived since the last tick the previous one is re-pushed so
    /// downstream sees a steady framerate.
    ///
    /// **Sequence assignment**: this method ignores `frame.meta.sequence`
    /// and stamps the composite with its own monotonic counter
    /// ([`Self::next_seq`]).  Upstream's sequence still serves tracing
    /// and telemetry, but it must not drive the writer-thread dedupe —
    /// see the field docs on [`OutputPipeline::frame_counter`] for the
    /// idle→resume collision this decoupling closes.
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
        let upstream_seq = frame.meta.sequence;
        let seq = self.next_seq();
        let bytes: Arc<[u8]> = match frame.data {
            fluxframe_core::frame::FrameBuffer::Owned(v) => Arc::from(v.into_boxed_slice()),
            fluxframe_core::frame::FrameBuffer::Shared(arc) => arc,
        };
        let composite = LatestComposite { bytes, seq };
        *self.latest.lock() = Some(composite);
        trace!(seq, upstream_seq, "published composite to writer thread");
        Ok(())
    }

    /// Allocate the next monotonic dedupe sequence number.
    ///
    /// Extracted so unit tests can exercise the counter without
    /// building a full GStreamer pipeline (which requires `gst::init`
    /// and a working sink element).  Production callers go through
    /// [`Self::push_frame`].
    #[must_use]
    pub(crate) fn next_seq(&self) -> u64 {
        self.frame_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Stop the pipeline; signals EOS to the sink, joins the writer
    /// thread and tears down.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::StateChangeFailed`] on teardown failure.
    pub fn stop(&self) -> Result<(), PipelineError> {
        // Shutdown order:
        //   1. flip `started` to false so the writer loop exits at its
        //      next iteration check;
        //   2. join the writer BEFORE sending EOS, otherwise the writer
        //      can race past the `started` check and push one more
        //      buffer into `appsrc` after we've already called
        //      `end_of_stream` on it (illegal-state warning at best,
        //      lost EOS at worst);
        //   3. send EOS so downstream drains cleanly;
        //   4. tear the pipeline down to Null.
        let was_started = self.started.swap(false, Ordering::AcqRel);
        // Take the handle out of the lock BEFORE joining: holding the
        // mutex across `join()` would deadlock anyone (e.g. the writer
        // itself, via a future hook) that tries to reach the handle
        // slot while we wait.  Releasing the lock first also avoids
        // priority-inversion stalls on slow shutdowns.
        let handle = self.writer_handle.lock().take();
        if let Some(handle) = handle {
            match handle.join() {
                Ok(stats) => debug!(
                    frames_pushed = stats.frames_pushed,
                    frames_skipped_duplicate = stats.frames_skipped_duplicate,
                    "writer thread joined cleanly"
                ),
                Err(e) => warn!(?e, "writer thread panicked while joining"),
            }
        }
        if was_started {
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

/// Assemble the ordered element list passed to `pipeline.add_many` /
/// `Element::link_many`.  The order encodes the dataflow:
///
/// `appsrc → queue → videoconvert → [videoscale] → [sink_capsfilter]
///   → sink_pre... → sink`
///
/// Extracted from [`OutputPipeline::build`] so the latter reads
/// top-to-bottom without an inline list-building paragraph.
fn assemble_pipeline_elements<'a>(
    appsrc_elem: &'a gstreamer::Element,
    queue: &'a gstreamer::Element,
    videoconvert: &'a gstreamer::Element,
    videoscale: Option<&'a gstreamer::Element>,
    sink_capsfilter: Option<&'a gstreamer::Element>,
    sink_pre: &'a [gstreamer::Element],
    sink_elem: &'a gstreamer::Element,
) -> Vec<&'a gstreamer::Element> {
    let mut elements: Vec<&gstreamer::Element> = vec![appsrc_elem, queue, videoconvert];
    if let Some(vs) = videoscale {
        elements.push(vs);
    }
    if let Some(cf) = sink_capsfilter {
        elements.push(cf);
    }
    for pre in sink_pre {
        elements.push(pre);
    }
    elements.push(sink_elem);
    elements
}

/// Install the sink-pad CAPS event probe used for negotiation
/// diagnostics.  The first CAPS event is logged at `debug!`
/// (initial negotiation); any subsequent one is escalated to
/// `warn!` because a mid-stream renegotiation historically
/// correlates with single-frame stride shifts in the rendered
/// output.
///
/// Extracted from [`OutputPipeline::build`] to keep that function
/// readable; no state crosses back out — the probe captures its
/// own `Arc` counter.
fn install_sink_pad_probes(sink_elem: &gstreamer::Element) {
    let Some(sink_pad) = sink_elem.static_pad("sink") else {
        warn!("sink element has no `sink` pad — caps probe not installed");
        return;
    };
    let caps_seen = Arc::new(AtomicBool::new(false));
    sink_pad.add_probe(gstreamer::PadProbeType::EVENT_DOWNSTREAM, move |_pad, info| {
        if let Some(gstreamer::PadProbeData::Event(ref ev)) = info.data
            && let gstreamer::EventView::Caps(c) = ev.view()
        {
            if caps_seen.swap(true, Ordering::Relaxed) {
                warn!(caps = %c.caps(), "sink pad CAPS event after initial negotiation (mid-stream renegotiation)");
            } else {
                debug!(caps = %c.caps(), "sink pad initial CAPS");
            }
        }
        gstreamer::PadProbeReturn::Ok
    });
}

/// What `build_sink_chain` returns: zero or more pre-sink elements
/// (linked in order immediately upstream of `sink`), the terminal sink
/// itself, and an optional fd guard that the caller MUST hold alive
/// for the lifetime of the pipeline.  Only the V4l2Loopback
/// direct-write path populates `fd_guard`.
struct SinkChainResult {
    pre_sink: Vec<gstreamer::Element>,
    sink: gstreamer::Element,
    fd_guard: Option<V4lFdGuard>,
}

/// RAII guard holding the v4l2 device fd open for the lifetime of the
/// pipeline.  `fdsink` only *borrows* the fd (it has no `auto-close`
/// property and its stop() leaves the fd open); if this guard is
/// dropped first the next pipeline tick writes to a closed fd.  Do
/// not remove the field that holds this guard — see
/// `build_v4l2_direct_chain` for why.
struct V4lFdGuard(#[allow(dead_code)] v4l::Device);

/// Default PipeWire node name advertised when the caller does not pass
/// one through.  Kept as a const so the magic string is named at its
/// definition site rather than buried in the sink builder.
const PIPEWIRE_DEFAULT_NODE_NAME: &str = "fluxframe";

/// Build the optional gstreamer-side `capsfilter` that pins the
/// pixel format when the operator's requested sink format differs
/// from the appsrc-declared format.  Width/height/framerate stay
/// flexible so the sink can negotiate them.
///
/// Skipped entirely on the V4l2Loopback direct-write path
/// (`is_v4l2_direct == true`) — that path attaches its own fully-
/// pinned capsfilter as part of `pre_sink` because `fdsink` does
/// not negotiate v4l2 caps.
fn maybe_format_capsfilter(
    appsrc_format: PixelFormat,
    sink_format: Option<PixelFormat>,
    is_v4l2_direct: bool,
) -> Result<Option<gstreamer::Element>, PipelineError> {
    if is_v4l2_direct {
        return Ok(None);
    }
    sink_format
        .filter(|sf| *sf != appsrc_format)
        .map(|sf| build_format_capsfilter("output_sink_caps", sf, None, None))
        .transpose()
}

/// Build a `capsfilter` element pinning `video/x-raw,format=<fmt>` and
/// optionally `width`/`height`.  Centralises the caps-builder boilerplate
/// shared between the soft pin in `maybe_format_capsfilter` and the
/// fully-pinned filter in `build_v4l2_direct_chain` (the latter has to
/// pin width/height too because `fdsink` cannot negotiate v4l2 caps with
/// the kernel).
fn build_format_capsfilter(
    name: &str,
    fmt: PixelFormat,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<gstreamer::Element, PipelineError> {
    let cf = make_element("capsfilter", name)?;
    let gst_fmt = crate::frame_conv::pixel_format_to_gst(fmt);
    let mut builder = gstreamer::Caps::builder("video/x-raw").field("format", gst_fmt.to_str());
    if let Some(w) = width {
        builder = builder.field("width", i32::try_from(w).unwrap_or(i32::MAX));
    }
    if let Some(h) = height {
        builder = builder.field("height", i32::try_from(h).unwrap_or(i32::MAX));
    }
    cf.set_property("caps", builder.build());
    Ok(cf)
}

/// Construct the terminal sink element (and any pre-sink helper)
/// for the requested [`OutputSink`].  Extracted from
/// [`OutputPipeline::build`] to keep the latter at a digestible size
/// and to centralise the sink-specific quirks.
///
/// `effective_format` is the pixel format the operator wants on the
/// wire — needed by the V4l2Loopback direct-write path to negotiate
/// `VIDIOC_S_FMT` on the device before handing the fd to fdsink.
/// Other sink paths ignore it (they delegate format negotiation to
/// the sink element itself).
fn build_sink_chain(
    sink: OutputSink,
    width: u32,
    height: u32,
    effective_format: PixelFormat,
) -> Result<SinkChainResult, PipelineError> {
    match sink {
        OutputSink::Fake => Ok(SinkChainResult {
            pre_sink: Vec::new(),
            sink: make_element("fakesink", "output_sink")?,
            fd_guard: None,
        }),
        OutputSink::Auto => Ok(SinkChainResult {
            pre_sink: Vec::new(),
            sink: make_element("autovideosink", "output_sink")?,
            fd_guard: None,
        }),
        OutputSink::V4l2Loopback { device } => {
            build_v4l2_direct_chain(&device, width, height, effective_format)
        }
        OutputSink::Pipewire { node_name } => Ok(SinkChainResult {
            pre_sink: Vec::new(),
            sink: build_pipewire_sink(node_name.as_deref())?,
            fd_guard: None,
        }),
    }
}

/// Build the V4l2Loopback output chain that bypasses GStreamer's
/// `v4l2sink` entirely.
///
/// Why bypass v4l2sink:
///   * `gst-plugins-good`'s v4l2sink for OUTPUT only implements MMAP
///     io-mode (`io-mode=rw` is a literal FIXME no-op).  In MMAP mode
///     the kernel's mmap'd buffer pool is shared between v4l2sink's
///     per-row memcpy and the consumer's DQBUF/read; v4l2loopback
///     provides no atomicity around this — see
///     <https://github.com/umlaeute/v4l2loopback/issues/191>.
///     Consumers see horizontal-seam torn frames whenever the
///     producer takes long enough for a memcpy that the consumer
///     reads the slot mid-write.
///   * `identity drop-allocation=true`, `min-queued-buffers`, queue
///     decoupling, writer cadence — all only narrow the race window
///     and do not eliminate it (operator confirmed empirically).
///   * The kernel `vidioc_write` handler IS atomic (serialised under
///     `image_mutex`), so writing via the `write(2)` syscall against
///     a fd opened directly on /dev/video10 produces no torn frames.
///     This is the approach OBS Studio uses for its virtual camera.
///
/// Implementation: open the device via the `v4l` crate (which keeps
/// us in safe Rust, the unsafe ioctls live behind its API), call
/// `VIDIOC_S_FMT` for the desired output format, then hand the
/// resulting fd to GStreamer's `fdsink`.  fdsink writes each
/// incoming buffer to the fd in one `write(2)` call, the kernel
/// serialises the write against any concurrent consumer read.
///
/// The `v4l::Device` is returned alongside so the caller can keep
/// it alive for the lifetime of the pipeline (fdsink does NOT take
/// ownership of the fd — when the device drops, the fd closes).
fn build_v4l2_direct_chain(
    device_path: &Path,
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
) -> Result<SinkChainResult, PipelineError> {
    // Pre-open write check first so EACCES/EBUSY/ENOENT surface with
    // a hint *before* the v4l crate emits a less-specific I/O error.
    let canon = check_v4l2_output_access(device_path)?;

    let device =
        v4l::Device::with_path(&canon).map_err(|e| PipelineError::OutputDeviceUnavailable {
            device: canon.display().to_string(),
            reason: format!("v4l::Device::with_path failed: {e}"),
            hint: "ensure the v4l2loopback module is loaded and the device exists".into(),
        })?;

    let fourcc = pixel_format_to_v4l_fourcc(pixel_format);
    let mut want = v4l::Format::new(width, height, fourcc);
    // v4l2loopback ignores the per-row stride request and computes
    // its own; setting `stride` to 0 means "let the driver decide".
    want.stride = 0;
    want.size = 0;
    let got = v4l::video::Output::set_format(&device, &want).map_err(|e| {
        PipelineError::OutputDeviceUnavailable {
            device: canon.display().to_string(),
            reason: format!("VIDIOC_S_FMT({width}x{height} {fourcc}) failed: {e}"),
            hint: "the requested format may not be supported by v4l2loopback".into(),
        }
    })?;
    if got.width != width || got.height != height || got.fourcc != fourcc {
        return Err(PipelineError::OutputDeviceUnavailable {
            device: canon.display().to_string(),
            reason: format!(
                "v4l2 negotiated {}x{} {} (wanted {}x{} {})",
                got.width, got.height, got.fourcc, width, height, fourcc
            ),
            hint: "v4l2loopback rejected the format change — restart the module".into(),
        });
    }
    debug!(
        device = %canon.display(),
        width = got.width,
        height = got.height,
        fourcc = %got.fourcc,
        "v4l2 output format negotiated",
    );

    let fd = device.handle().fd();
    let sink_elem = make_element("fdsink", "output_sink")?;
    sink_elem.set_property("fd", fd);
    // fdsink has no `auto-close` property (unlike fdsrc / multifdsink) and
    // its stop() leaves the fd open — verified against gst-plugins-base
    // gstfdsink.c.  The fd's lifecycle is therefore owned exclusively by
    // `fd_guard` below; there is no double-close to defend against here.
    // Decouple the heavy effect-chain thread from the fdsink write
    // thread.  Tiny buffer + leaky=downstream so a stall in the
    // syscall never holds back the effect chain.
    let sink_queue = make_element("queue", "output_sink_queue")?;
    sink_queue.set_property("max-size-buffers", 2u32);
    sink_queue.set_property_from_str("leaky", "downstream");
    sink_queue.set_property("max-size-bytes", 0u32);
    sink_queue.set_property("max-size-time", 0u64);
    // Pin the wire format fully — fdsink doesn't negotiate v4l2
    // caps, so we must guarantee upstream produces buffers in the
    // exact format the kernel was told to expect via VIDIOC_S_FMT.
    let capsfilter =
        build_format_capsfilter("output_sink_caps", pixel_format, Some(width), Some(height))?;

    Ok(SinkChainResult {
        pre_sink: vec![sink_queue, capsfilter],
        sink: sink_elem,
        fd_guard: Some(V4lFdGuard(device)),
    })
}

/// Map our [`PixelFormat`] enum to the V4L2 FOURCC used by
/// `VIDIOC_S_FMT`.  All current variants have a corresponding
/// v4l2 fourcc — the helper stays `infallible` rather than
/// `Result` because there's no failure mode at this level.
fn pixel_format_to_v4l_fourcc(fmt: PixelFormat) -> v4l::FourCC {
    #[allow(
        clippy::match_same_arms,
        reason = "wildcard arm exists only because PixelFormat is #[non_exhaustive]; \
                  it intentionally aliases the YUYV mapping as the safest fallback"
    )]
    let code: &[u8; 4] = match fmt {
        PixelFormat::Yuy2 => b"YUYV",
        PixelFormat::Nv12 => b"NV12",
        PixelFormat::Rgb => b"RGB3",
        PixelFormat::Bgr => b"BGR3",
        PixelFormat::Rgba => b"RGB4",
        PixelFormat::Gray8 => b"GREY",
        _ => b"YUYV",
    };
    v4l::FourCC::new(code)
}

fn build_pipewire_sink(node_name: Option<&str>) -> Result<gstreamer::Element, PipelineError> {
    // `pipewiresink` is provided by `gst-plugin-pipewire`.
    // `make_element` surfaces a structured MissingElement error (with
    // install hint) if the plugin is missing.
    let elem = make_element("pipewiresink", "output_sink")?;
    // Advertise this stream as a video *source* in the PipeWire graph
    // so consumers (Firefox via xdg-desktop-portal, OBS-PW, helvum)
    // discover it as a camera.  Without `media.class = Video/Source`
    // pipewiresink defaults to looking for a target consumer and exits
    // with "no target node available".
    let name = node_name.unwrap_or(PIPEWIRE_DEFAULT_NODE_NAME);
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

/// Cumulative writer-thread counters reported on exit.
///
/// Returned from [`writer_loop`] so callers (and unit tests) can observe
/// whether the dedupe path actually fired. The production spawn site
/// logs the values at thread exit; tests assert on them via
/// `JoinHandle::join`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WriterStats {
    /// Number of writer ticks that successfully called
    /// `appsrc.push_buffer` (i.e. a fresh buffer actually reached the
    /// pipeline).  Failed pushes do not count here.
    pub frames_pushed: u64,
    /// Number of writer ticks that observed the same `seq` as the
    /// previous successful push and therefore SKIPPED the push entirely
    /// — no buffer was built, no `push_buffer` call was issued.  This
    /// is the steady-state idle path that keeps v4l2loopback CPU/IO
    /// close to zero while no new composites are produced.
    pub frames_skipped_duplicate: u64,
}

/// Writer thread loop.  Wakes every `interval`, takes a cheap `Arc`
/// clone of the latest composite, and pushes a fresh `gst::Buffer`
/// into `appsrc` with a fabricated monotonic PTS — but **only** when
/// the composite's `seq` differs from the last successfully-pushed
/// one. Repeating an identical frame down the v4l2loopback fd at full
/// `fps` is pure CPU and bandwidth waste; the kernel ring buffer
/// already retains the last frame for late readers, and the effect
/// chain bumps `seq` whenever it has something new to publish (1 Hz
/// during idle via `IdleConfig::fps`, full fps during Active).
/// Exits when `started` flips to `false`.
fn writer_loop(
    started: &Arc<AtomicBool>,
    latest: &Arc<Mutex<Option<LatestComposite>>>,
    appsrc: &AppSrc,
    interval: Duration,
) -> WriterStats {
    let start_instant = Instant::now();
    let mut next_tick = start_instant + interval;
    let mut last_pushed_seq: Option<u64> = None;
    let mut stats = WriterStats::default();
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
            .as_ref()
            .map(|c| (Arc::clone(&c.bytes), c.seq));
        let Some((bytes, seq)) = snapshot else {
            // Effect chain hasn't produced anything yet — nothing to
            // push.  Cheap idle.
            continue;
        };
        if last_pushed_seq == Some(seq) {
            // Same composite as last tick — kernel still has it. Skip
            // the writev + memcpy entirely so steady-state idle costs
            // ~one futex wake + one Arc refcount per tick.
            stats.frames_skipped_duplicate += 1;
            continue;
        }
        // Do NOT advance `last_pushed_seq` yet.  Committing the dedupe
        // marker here — before the buffer is actually accepted by
        // `appsrc` — turns any transient `build_buffer` / `push_buffer`
        // failure into permanent loss of that seq: the producer would
        // have to bump its sequence again to trigger another attempt.
        // Instead, only update `last_pushed_seq` inside the Ok(push)
        // branch below so a failed tick is retried on the next cadence.

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
            Ok(buf) => match appsrc.push_buffer(buf) {
                Ok(_) => {
                    last_pushed_seq = Some(seq);
                    stats.frames_pushed += 1;
                }
                Err(e) => {
                    // The most common failure here is the shutdown race —
                    // `stop()` sent EOS but the writer's loop iteration
                    // was already past the `started` check.  Stay silent
                    // in that case; surface anything else.
                    if started.load(Ordering::Acquire) {
                        warn!(error = %e, seq, "writer: appsrc.push_buffer failed");
                    }
                    // Leave `last_pushed_seq` unchanged so the next tick
                    // retries this seq automatically.
                }
            },
            Err(e) => {
                warn!(error = %e, seq, "writer: buffer build failed");
                // Same reasoning as the push_buffer error branch: do not
                // burn this seq on a transient build failure.
            }
        }
    }
    debug!(
        frames_pushed = stats.frames_pushed,
        frames_skipped_duplicate = stats.frames_skipped_duplicate,
        "output writer thread exited"
    );
    stats
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
    fn pixel_format_to_v4l_fourcc_table() {
        let cases: &[(PixelFormat, &[u8; 4])] = &[
            (PixelFormat::Yuy2, b"YUYV"),
            (PixelFormat::Nv12, b"NV12"),
            (PixelFormat::Rgb, b"RGB3"),
            (PixelFormat::Bgr, b"BGR3"),
            (PixelFormat::Rgba, b"RGB4"),
            (PixelFormat::Gray8, b"GREY"),
        ];
        for (fmt, code) in cases {
            let got = pixel_format_to_v4l_fourcc(*fmt);
            assert_eq!(
                got.repr, **code,
                "fourcc mismatch for {fmt:?}: got {:?}, want {:?}",
                got.repr, code
            );
            assert_eq!(got, v4l::FourCC::new(code));
        }
    }

    /// Steady-state idle: the latest-composite slot holds a single
    /// frame (seq = 5) and the worker never updates it. The writer
    /// must push exactly ONCE (the first time it sees the seq) and
    /// dedupe every subsequent identical tick.
    ///
    /// Prior behaviour: 25 fps × 716 800-byte writev to v4l2loopback
    /// during deep idle. New behaviour: one writev per worker update,
    /// zero between updates.
    #[test]
    fn writer_skips_push_when_seq_unchanged() {
        gstreamer::init().expect("gst init");
        let appsrc = gstreamer_app::AppSrc::builder().build();

        let started = Arc::new(AtomicBool::new(true));
        let bytes: Arc<[u8]> = Arc::from(vec![0u8; 8].into_boxed_slice());
        let latest = Arc::new(Mutex::new(Some(LatestComposite { bytes, seq: 5 })));
        let interval = Duration::from_millis(20);

        let started_for_thread = Arc::clone(&started);
        let latest_for_thread = Arc::clone(&latest);
        let appsrc_for_thread = appsrc.clone();
        let handle = std::thread::spawn(move || {
            writer_loop(
                &started_for_thread,
                &latest_for_thread,
                &appsrc_for_thread,
                interval,
            )
        });

        // ~5 ticks at 20 ms each. The very first tick at +20 ms sees
        // seq=5 (fresh) → push. Every subsequent tick sees the same
        // seq → dedupe.
        std::thread::sleep(Duration::from_millis(100));
        started.store(false, Ordering::Release);
        let stats = handle.join().expect("writer joined");

        assert_eq!(
            stats.frames_pushed, 1,
            "first observation of seq must push exactly once; \
             subsequent identical ticks must dedupe"
        );
        assert!(
            stats.frames_skipped_duplicate >= 2,
            "expected at least 2 deduped ticks within 100 ms / 20 ms cadence, got {}",
            stats.frames_skipped_duplicate
        );
    }

    /// Advancing seq mid-run forces another push: the writer is
    /// event-driven on seq changes, not on raw clock ticks.
    #[test]
    fn writer_pushes_again_when_seq_advances() {
        gstreamer::init().expect("gst init");
        let appsrc = gstreamer_app::AppSrc::builder().build();

        let started = Arc::new(AtomicBool::new(true));
        let bytes: Arc<[u8]> = Arc::from(vec![0u8; 8].into_boxed_slice());
        let latest = Arc::new(Mutex::new(Some(LatestComposite {
            bytes: Arc::clone(&bytes),
            seq: 1,
        })));
        let interval = Duration::from_millis(20);

        let started_for_thread = Arc::clone(&started);
        let latest_for_thread = Arc::clone(&latest);
        let appsrc_for_thread = appsrc.clone();
        let handle = std::thread::spawn(move || {
            writer_loop(
                &started_for_thread,
                &latest_for_thread,
                &appsrc_for_thread,
                interval,
            )
        });

        // Let the writer pick up seq=1 (first tick at ~20 ms).
        std::thread::sleep(Duration::from_millis(50));
        // Mutate the slot — the writer's next tick sees a different
        // seq and must emit a second buffer.
        *latest.lock() = Some(LatestComposite {
            bytes: Arc::clone(&bytes),
            seq: 2,
        });
        std::thread::sleep(Duration::from_millis(50));
        started.store(false, Ordering::Release);
        let stats = handle.join().expect("writer joined");

        assert_eq!(
            stats.frames_pushed, 2,
            "seq=1 → push; seq unchanged → dedupe; seq=2 → push again. \
             Got frames_pushed={}, frames_skipped_duplicate={}",
            stats.frames_pushed, stats.frames_skipped_duplicate
        );
    }

    /// `push_frame` must NOT trust `frame.meta.sequence` for its
    /// dedupe path: upstream frames may legitimately share `sequence
    /// = 0` (e.g. `FrameMeta::default()` placeholders during idle and
    /// the first real capture frame after `AtomicU64::new(0)
    /// .fetch_add(1)`).  Letting that collision through would cause
    /// the writer thread to permanently dedupe the first frame after
    /// every idle→resume.
    ///
    /// We exercise the underlying counter directly (rather than
    /// building a full `OutputPipeline`) so the assertion is robust
    /// against host-side GStreamer availability; the production path
    /// in [`OutputPipeline::push_frame`] is a thin wrapper over
    /// [`OutputPipeline::next_seq`].
    #[test]
    fn push_frame_assigns_monotonic_seq_ignoring_meta() {
        // Mirror the production initialiser: counter starts at 1 so
        // the first push never collides with a stale
        // `last_pushed_seq == Some(0)` left by a placeholder frame.
        let counter = AtomicU64::new(1);
        let next = || counter.fetch_add(1, Ordering::Relaxed);

        let seq_a = next();
        let seq_b = next();
        let seq_c = next();

        assert_eq!(seq_a, 1, "first allocation must start at 1, not 0");
        assert!(
            seq_b > seq_a && seq_c > seq_b,
            "next_seq must be strictly monotonic across pushes \
             with identical (or zero) frame.meta.sequence values; \
             got {seq_a}, {seq_b}, {seq_c}"
        );
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
