//! GStreamer input pipeline construction.
//!
//! Stage 1 supported only the synthetic `videotestsrc` backend; Stage 2
//! adds the V4L2 capture path.  All backends share the same downstream
//! plumbing — `videoconvert → capsfilter → appsink` — so producing a
//! [`VideoFrame`] from a captured `gst::Sample` happens in exactly one
//! place ([`crate::frame_conv::sample_to_frame`]).

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use fluxframe_core::error::PipelineError;
use fluxframe_core::frame::PixelFormat;
use gstreamer::prelude::*;
use gstreamer_app::AppSink;
use tracing::{debug, error, trace};

use crate::frame_conv::sample_to_frame;
use crate::slot::LatestFrameSlot;
use crate::util::{build_caps, check_v4l2_input_access, make_element};
use crate::v4l2_caps::{CaptureEncoding, NativeMode};

/// Capture pipeline owning the GStreamer elements and the shared frame slot.
pub struct InputPipeline {
    pipeline: gstreamer::Pipeline,
    slot: LatestFrameSlot,
    /// Per-pipeline monotonic frame counter.  Lives in an [`Arc`] so the
    /// `appsink` callback can borrow it for the lifetime of the pipeline.
    sequence: Arc<AtomicU64>,
}

/// Negotiated capture parameters.
///
/// `#[non_exhaustive]` so adding optional fields (e.g. device path, colour
/// range hint) in Stage 2 is not a breaking change for downstream crates.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct InputParams {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate (frames per second).
    pub fps: u32,
    /// Pixel format expected downstream.
    pub format: PixelFormat,
}

impl InputParams {
    /// Construct a fully-specified [`InputParams`].
    ///
    /// Provided because the struct is `#[non_exhaustive]`, so cross-crate
    /// callers cannot use the record literal syntax.  Inside this crate
    /// the literal still works.
    #[must_use]
    pub fn new(width: u32, height: u32, fps: u32, format: PixelFormat) -> Self {
        Self {
            width,
            height,
            fps,
            format,
        }
    }
}

impl InputPipeline {
    /// Build a `videotestsrc`-based pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] if any required element cannot be
    /// instantiated (missing GStreamer plugin) or if the pipeline cannot
    /// be linked.
    pub fn build_testsrc(params: InputParams) -> Result<Self, PipelineError> {
        let testsrc = make_element("videotestsrc", "input_src")?;
        testsrc.set_property("is-live", true);
        testsrc.set_property_from_str("pattern", "smpte");

        Self::build(&[testsrc], &[], params)
    }

    /// Build a `v4l2src`-based pipeline pointed at `device`, pinning the
    /// camera to `mode` (see [`crate::v4l2_caps::detect_native_mode`]).
    /// `params` describes the frames delivered downstream; the pipeline
    /// scales and converts from `mode` to them.
    ///
    /// Pre-opens `device` for reading so a missing device, busy device or
    /// permission failure surfaces as a structured
    /// [`PipelineError::InputDeviceUnavailable`] *before* the GStreamer
    /// pipeline is constructed.  Without this pre-check the user sees an
    /// opaque GStreamer state-change error during `set_state(Playing)`.
    ///
    /// # Errors
    ///
    /// * [`PipelineError::InputDeviceUnavailable`] when the device cannot
    ///   be opened (missing, busy, permission denied).
    /// * [`PipelineError::MissingElement`] when the `v4l2src` plugin is
    ///   not registered.
    /// * Other [`PipelineError`] variants from element/link construction.
    pub fn build_v4l2(
        device: &Path,
        mode: &NativeMode,
        params: InputParams,
    ) -> Result<Self, PipelineError> {
        // The check returns the canonicalised path (symlinks resolved,
        // verified to be a /dev/ character device); feed that to v4l2src
        // rather than the user-supplied original so the pipeline targets
        // exactly the inode the access check authorised.
        let canon = check_v4l2_input_access(device)?;
        let src = make_element("v4l2src", "input_src")?;
        src.set_property_from_str("device", canon.to_string_lossy().as_ref());
        // `do-timestamp=true` so frames arriving from the camera carry a
        // PTS based on the running clock — required for the downstream
        // `appsink` to publish meaningful capture timestamps.
        src.set_property("do-timestamp", true);
        // Without pinned caps `v4l2src` fixates a mode of its own, which
        // need not be the one `mode` was chosen for (size, rate or
        // encoding).
        let src_caps = make_element("capsfilter", "input_src_caps")?;
        src_caps.set_property("caps", mode.source_caps()?);
        let decode = match mode.encoding {
            CaptureEncoding::Raw(_) => Vec::new(),
            CaptureEncoding::Mjpeg => vec![make_element("jpegdec", "input_jpegdec")?],
        };
        Self::build(&[src, src_caps], &decode, params)
    }

    /// Assemble `source… → queue → decode… → videoconvert → videoscale →
    /// videorate → capsfilter → appsink`.  `decode` sits after the leaky
    /// queue so a slow consumer drops compressed frames, not decoded ones.
    fn build(
        source: &[gstreamer::Element],
        decode: &[gstreamer::Element],
        params: InputParams,
    ) -> Result<Self, PipelineError> {
        let pipeline = gstreamer::Pipeline::with_name("fluxframe-input");

        let queue = make_element("queue", "input_queue")?;
        queue.set_property("max-size-buffers", 1u32);
        queue.set_property_from_str("leaky", "downstream");
        queue.set_property("max-size-bytes", 0u32);
        queue.set_property("max-size-time", 0u64);

        let videoconvert = make_element("videoconvert", "input_videoconvert")?;
        let videoscale = make_element("videoscale", "input_videoscale")?;
        // Preserve aspect ratio when the camera's native resolution
        // differs from the operator's `[input] width/height`. Without
        // this, scaling 640×480 (4:3) to 1280×720 (16:9) silently
        // stretches the image. `add-borders=true` pads with black
        // bars instead.
        videoscale.set_property("add-borders", true);
        // `videorate` enforces `params.fps` BEFORE the effect chain sees
        // frames.  Without it, testsrc happily generates at its own rate
        // and v4l2 cameras ignore `framerate` hints in capsfilter — both
        // would push every captured frame through the chain, defeating
        // the operator's ability to lower CPU load via `[input] fps`.
        let videorate = make_element("videorate", "input_videorate")?;
        let capsfilter = make_element("capsfilter", "input_capsfilter")?;

        let caps = build_caps(params.width, params.height, params.fps, params.format)?;
        capsfilter.set_property("caps", &caps);

        let appsink_elem = make_element("appsink", "input_sink")?;
        appsink_elem.set_property("sync", false);
        appsink_elem.set_property("max-buffers", 1u32);
        appsink_elem.set_property("drop", true);

        let chain: Vec<&gstreamer::Element> = source
            .iter()
            .chain([&queue])
            .chain(decode)
            .chain([
                &videoconvert,
                &videoscale,
                &videorate,
                &capsfilter,
                &appsink_elem,
            ])
            .collect();

        pipeline
            .add_many(chain.iter().copied())
            .map_err(|e| PipelineError::Runtime {
                reason: format!("pipeline.add_many failed: {e}"),
            })?;

        gstreamer::Element::link_many(chain.iter().copied()).map_err(|e| {
            PipelineError::Runtime {
                reason: format!("element link failed: {e}"),
            }
        })?;

        let appsink =
            appsink_elem
                .dynamic_cast::<AppSink>()
                .map_err(|_| PipelineError::Runtime {
                    reason: "input_sink is not an AppSink".into(),
                })?;

        let slot = LatestFrameSlot::new();
        let sequence = Arc::new(AtomicU64::new(0));
        attach_appsink_callbacks(&appsink, slot.clone(), Arc::clone(&sequence));

        Ok(Self {
            pipeline,
            slot,
            sequence,
        })
    }

    /// Shared frame slot — published frames land here.  Clones cheaply.
    #[must_use]
    pub fn slot(&self) -> LatestFrameSlot {
        self.slot.clone()
    }

    /// Snapshot of the per-pipeline frame counter — equals the sequence
    /// number that will be assigned to the *next* captured frame.
    ///
    /// Exposed primarily for tests and metrics; the value is read with
    /// `Relaxed` ordering and therefore carries no synchronisation guarantee
    /// relative to other observations.
    #[must_use]
    pub fn frames_captured(&self) -> u64 {
        self.sequence.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Transition the pipeline to PLAYING.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::StateChangeFailed`] if GStreamer refuses
    /// the state change.
    pub fn start(&self) -> Result<(), PipelineError> {
        self.pipeline
            .set_state(gstreamer::State::Playing)
            .map(|_| ())
            .map_err(|e| PipelineError::StateChangeFailed {
                reason: format!("set_state(Playing) failed: {e}"),
            })
    }

    /// Re-acquire the camera on the **existing** pipeline: drop to
    /// `Null` and drive back to `Playing`.
    ///
    /// Idle→resume calls this instead of a bare [`Self::start`]. The
    /// explicit `Null` first is an idempotent reset — it recovers from a
    /// prior resume that left the pipeline wedged in a partial/`Ready`
    /// state, and `v4l2src` re-`open(2)`s the device on the `Null→Ready`
    /// transition, so a camera unplugged-and-replugged onto the *same*
    /// `/dev/videoN` node is picked up. Deliberately does **not**
    /// recreate the pipeline or its elements: the frame slot and the
    /// bus belong to this `gstreamer::Pipeline` and the supervisor holds
    /// clones of both, so recreating them would orphan the worker's slot
    /// read and the bus watch. A camera that re-enumerated onto a
    /// *different* node is out of scope here — that path is handled by
    /// the supervisor re-running device auto-selection.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::StateChangeFailed`] if either state
    /// change fails.
    pub fn reacquire(&self) -> Result<(), PipelineError> {
        self.pipeline_to_null()?;
        self.start()
    }

    /// Stop the pipeline and release device resources.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::StateChangeFailed`] on teardown failure.
    pub fn stop(&self) -> Result<(), PipelineError> {
        self.slot.close();
        self.pipeline_to_null()
    }

    /// Drop the pipeline to `Null` without closing the frame slot.
    ///
    /// Stage 15 idle mode uses this to release the V4L2 device (and
    /// turn the camera LED off on most UVC cameras) while keeping the
    /// worker loop running for placeholder pushes. The pipeline can
    /// later be resumed via [`Self::start`]; the slot stays open so a
    /// resumed pipeline finds the worker ready to receive frames.
    ///
    /// Distinct from [`Self::stop`], which signals supervisor shutdown
    /// by closing the slot. Calling `set_state_null` followed by
    /// `start` is reversible — both transitions are GStreamer
    /// internal state changes, no underlying I/O fan-out.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::StateChangeFailed`] if GStreamer
    /// refuses the state change.
    pub fn set_state_null(&self) -> Result<(), PipelineError> {
        self.pipeline_to_null()
    }

    /// Transition the underlying GStreamer pipeline to `Null`.
    ///
    /// Shared helper for [`Self::stop`] and [`Self::set_state_null`];
    /// only the implementation deduplicates, the two callers carry
    /// distinct semantic contracts (shutdown vs idle-mode pause).
    fn pipeline_to_null(&self) -> Result<(), PipelineError> {
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

fn attach_appsink_callbacks(appsink: &AppSink, slot: LatestFrameSlot, sequence: Arc<AtomicU64>) {
    appsink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                let sample = sink.pull_sample().map_err(|_| gstreamer::FlowError::Eos)?;
                match sample_to_frame(&sample, &sequence) {
                    Ok(frame) => {
                        trace!(seq = frame.meta.sequence, "captured frame");
                        slot.push(frame);
                        Ok(gstreamer::FlowSuccess::Ok)
                    }
                    Err(e) => {
                        error!(error = %e, "sample_to_frame failed; dropping sample");
                        // Returning Ok keeps the pipeline alive across a bad
                        // sample; a sustained failure surfaces through bus
                        // messages handled by the runtime supervisor.
                        Ok(gstreamer::FlowSuccess::Ok)
                    }
                }
            })
            .build(),
    );
    debug!("input appsink callbacks installed");
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    /// The MJPEG capture path (`queue → jpegdec → videoconvert …`) must
    /// deliver decoded RGB frames at the requested size. A camera is not
    /// needed: `videotestsrc → jpegenc` stands in for an MJPEG `v4l2src`.
    #[test]
    #[ignore = "requires GStreamer plugins; run with `cargo test -- --ignored`"]
    fn mjpeg_decode_chain_produces_rgb_frames() {
        crate::init().expect("gstreamer init");
        let testsrc = make_element("videotestsrc", "input_src").expect("videotestsrc");
        testsrc.set_property("is-live", true);
        let jpegenc = make_element("jpegenc", "test_jpegenc").expect("jpegenc");
        let jpegdec = make_element("jpegdec", "input_jpegdec").expect("jpegdec");

        let params = InputParams::new(320, 180, 30, PixelFormat::Rgb);
        let input = InputPipeline::build(&[testsrc, jpegenc], &[jpegdec], params).expect("build");
        input.start().expect("start");

        let slot = input.slot();
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut frame = None;
        while frame.is_none() && Instant::now() < deadline {
            frame = slot.recv_timeout(Duration::from_millis(100));
        }
        input.stop().expect("stop");

        let frame = frame.expect("at least one decoded frame within 2 s");
        assert_eq!((frame.width, frame.height), (320, 180));
        assert_eq!(frame.format, PixelFormat::Rgb);
    }
}
