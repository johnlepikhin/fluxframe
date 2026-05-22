//! GStreamer input pipeline construction.
//!
//! Stage 1 supports the synthetic `videotestsrc` backend only.  The V4L2
//! backend lands in Stage 2.  All backends share the same downstream
//! plumbing — `videoconvert → capsfilter → appsink` — so producing a
//! [`VideoFrame`] from a captured `gst::Sample` happens in exactly one
//! place ([`crate::frame_conv::sample_to_frame`]).

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use fluxframe_core::error::PipelineError;
use fluxframe_core::frame::PixelFormat;
use gstreamer::prelude::*;
use gstreamer_app::AppSink;
use tracing::{debug, error, trace};

use crate::frame_conv::sample_to_frame;
use crate::slot::LatestFrameSlot;
use crate::util::{build_caps, make_element};

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

        Self::build(&testsrc, params)
    }

    fn build(source: &gstreamer::Element, params: InputParams) -> Result<Self, PipelineError> {
        let pipeline = gstreamer::Pipeline::with_name("fluxframe-input");

        let queue = make_element("queue", "input_queue")?;
        queue.set_property("max-size-buffers", 1u32);
        queue.set_property_from_str("leaky", "downstream");
        queue.set_property("max-size-bytes", 0u32);
        queue.set_property("max-size-time", 0u64);

        let videoconvert = make_element("videoconvert", "input_videoconvert")?;
        let videoscale = make_element("videoscale", "input_videoscale")?;
        let capsfilter = make_element("capsfilter", "input_capsfilter")?;

        let caps = build_caps(params.width, params.height, params.fps, params.format)?;
        capsfilter.set_property("caps", &caps);

        let appsink_elem = make_element("appsink", "input_sink")?;
        appsink_elem.set_property("sync", false);
        appsink_elem.set_property("max-buffers", 1u32);
        appsink_elem.set_property("drop", true);

        pipeline
            .add_many([
                source,
                &queue,
                &videoconvert,
                &videoscale,
                &capsfilter,
                &appsink_elem,
            ])
            .map_err(|e| PipelineError::Runtime {
                reason: format!("pipeline.add_many failed: {e}"),
            })?;

        gstreamer::Element::link_many([
            source,
            &queue,
            &videoconvert,
            &videoscale,
            &capsfilter,
            &appsink_elem,
        ])
        .map_err(|e| PipelineError::Runtime {
            reason: format!("element link failed: {e}"),
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
        self.sequence
            .load(std::sync::atomic::Ordering::Relaxed)
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

    /// Stop the pipeline and release device resources.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::StateChangeFailed`] on teardown failure.
    pub fn stop(&self) -> Result<(), PipelineError> {
        self.slot.close();
        self.pipeline
            .set_state(gstreamer::State::Null)
            .map(|_| ())
            .map_err(|e| PipelineError::StateChangeFailed {
                reason: format!("set_state(Null) failed: {e}"),
            })
    }

    /// Borrow the underlying pipeline so the runtime supervisor can attach
    /// a bus listener.
    ///
    /// Leaks the GStreamer type by design — the bus is the only sanctioned
    /// integration point between this crate and the runtime layer.  Do not
    /// use this handle for state changes; route those through [`Self::start`]
    /// and [`Self::stop`].
    #[must_use]
    pub fn pipeline_for_bus(&self) -> &gstreamer::Pipeline {
        &self.pipeline
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
