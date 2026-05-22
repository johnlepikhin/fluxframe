//! GStreamer input pipeline construction.
//!
//! Stage 1 supports the synthetic `videotestsrc` backend only.  The V4L2
//! backend lands in Stage 2.  All backends share the same downstream
//! plumbing — `videoconvert → capsfilter → appsink` — so producing a
//! [`VideoFrame`] from a captured `gst::Sample` happens in exactly one
//! place ([`crate::frame_conv::sample_to_frame`]).

use fluxframe_core::error::PipelineError;
use fluxframe_core::frame::PixelFormat;
use gstreamer::prelude::*;
use gstreamer_app::AppSink;
use tracing::{debug, error, trace};

use crate::frame_conv::{pixel_format_to_gst, sample_to_frame};
use crate::slot::LatestFrameSlot;

/// Capture pipeline owning the GStreamer elements and the shared frame slot.
pub struct InputPipeline {
    pipeline: gstreamer::Pipeline,
    slot: LatestFrameSlot,
}

/// Negotiated capture parameters.
#[derive(Debug, Clone, Copy)]
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

        let caps = build_caps(params);
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
        attach_appsink_callbacks(&appsink, slot.clone());

        Ok(Self { pipeline, slot })
    }

    /// Shared frame slot — published frames land here.  Clones cheaply.
    #[must_use]
    pub fn slot(&self) -> LatestFrameSlot {
        self.slot.clone()
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

    /// Borrow the underlying pipeline for advanced operations (bus
    /// listening, querying state).  Intended for the runtime supervisor.
    #[must_use]
    pub fn pipeline(&self) -> &gstreamer::Pipeline {
        &self.pipeline
    }
}

fn attach_appsink_callbacks(appsink: &AppSink, slot: LatestFrameSlot) {
    appsink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                let sample = sink.pull_sample().map_err(|_| gstreamer::FlowError::Eos)?;
                match sample_to_frame(&sample) {
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

fn make_element(factory: &str, name: &str) -> Result<gstreamer::Element, PipelineError> {
    gstreamer::ElementFactory::make(factory)
        .name(name)
        .build()
        .map_err(|_| PipelineError::MissingElement {
            element: factory.into(),
            hint: format!("GStreamer plugin providing `{factory}` is not installed"),
        })
}

fn build_caps(params: InputParams) -> gstreamer::Caps {
    let gst_fmt = pixel_format_to_gst(params.format);
    gstreamer::Caps::builder("video/x-raw")
        .field("format", gst_fmt.to_str())
        .field("width", i32::try_from(params.width).unwrap_or(i32::MAX))
        .field("height", i32::try_from(params.height).unwrap_or(i32::MAX))
        .field(
            "framerate",
            gstreamer::Fraction::new(i32::try_from(params.fps).unwrap_or(i32::MAX), 1),
        )
        .build()
}
