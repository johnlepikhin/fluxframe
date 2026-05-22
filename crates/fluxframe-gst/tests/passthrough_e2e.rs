//! End-to-end Stage 1 smoke: videotestsrc -> effect chain -> fakesink.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use fluxframe_core::frame::PixelFormat;
use fluxframe_gst::input::{InputParams, InputPipeline};
use fluxframe_gst::output::{OutputParams, OutputPipeline, OutputSink};

#[test]
#[ignore = "requires GStreamer plugins; run with `cargo test -- --ignored`"]
fn passthrough_testsrc_to_fakesink_produces_frames() {
    fluxframe_gst::init().expect("gstreamer init");

    let params = InputParams::new(320, 240, 30, PixelFormat::Rgb);
    let input = InputPipeline::build_testsrc(params).expect("build input");

    let output_params = OutputParams::new(320, 240, 30, PixelFormat::Yuy2, OutputSink::Fake);
    let output = OutputPipeline::build(output_params).expect("build output");

    output.start().expect("output start");
    input.start().expect("input start");

    let slot = input.slot();
    let processed = Arc::new(AtomicU64::new(0));

    let deadline = Instant::now() + Duration::from_millis(1500);
    while Instant::now() < deadline {
        if let Some(frame) = slot.recv_timeout(Duration::from_millis(100)) {
            output.push_frame(frame).expect("push_frame");
            processed.fetch_add(1, Ordering::Relaxed);
        }
    }

    input.stop().expect("input stop");
    output.stop().expect("output stop");

    let count = processed.load(Ordering::Relaxed);
    assert!(
        count >= 10,
        "expected at least 10 frames through the pipeline within 1.5s, got {count}"
    );
}
