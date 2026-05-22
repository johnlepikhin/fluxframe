//! Stage 2 V4L2 loopback smoke (requires `v4l2loopback` modprobed).
//!
//! Skipped by default via `#[ignore]`; run via:
//!   cargo test -p fluxframe-gst --tests -- --ignored v4l2_loopback
//!
//! The test wires a `videotestsrc` capture pipeline to a `v4l2sink` output
//! pipeline pointed at `/dev/video10` and asserts that at least a handful
//! of frames make the full round trip in 1.5 s of wall-clock.  It's a
//! smoke check, not a conformance suite — the goal is to catch obvious
//! regressions in caps negotiation or pipeline lifecycle.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use fluxframe_core::frame::PixelFormat;
use fluxframe_gst::input::{InputParams, InputPipeline};
use fluxframe_gst::output::{OutputParams, OutputPipeline, OutputSink};

const LOOPBACK_PATH: &str = "/dev/video10";

#[test]
#[ignore = "requires v4l2loopback modprobed at /dev/video10; run with --ignored"]
fn testsrc_to_v4l2loopback_produces_frames() {
    if !Path::new(LOOPBACK_PATH).exists() {
        eprintln!("/dev/video10 not present, skipping; load v4l2loopback first");
        return;
    }
    fluxframe_gst::init().expect("gstreamer init");

    let input = InputPipeline::build_testsrc(InputParams::new(320, 240, 30, PixelFormat::Rgb))
        .expect("build input");
    let output = OutputPipeline::build(OutputParams::new(
        320,
        240,
        30,
        PixelFormat::Yuy2,
        OutputSink::V4l2Loopback {
            device: LOOPBACK_PATH.into(),
        },
    ))
    .expect("build output");

    output.start().expect("output start");
    input.start().expect("input start");

    let slot = input.slot();
    let count = Arc::new(AtomicU64::new(0));
    let deadline = Instant::now() + Duration::from_millis(1500);

    while Instant::now() < deadline {
        if let Some(frame) = slot.recv_timeout(Duration::from_millis(100)) {
            output.push_frame(frame).expect("push_frame");
            count.fetch_add(1, Ordering::Relaxed);
        }
    }

    input.stop().expect("input stop");
    output.stop().expect("output stop");

    let n = count.load(Ordering::Relaxed);
    assert!(
        n >= 5,
        "expected >=5 frames through testsrc->v4l2loopback in 1.5s, got {n}"
    );
}
