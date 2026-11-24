//! Pipelined segmentation in [`CompositeEffect`]: inference for frame N
//! overlaps the processing of frame N-1, and the output trails the input
//! by exactly one frame.
//!
//! A mock asynchronous engine marks pixel `k % 4` of a 2×2 frame as
//! foreground, where `k` is read from the frame itself (red channel =
//! `k * 10`).  The background chain fills everything else, so each
//! emitted frame shows which mask it was composed with: the mask must
//! always come from the same frame.

#![cfg(feature = "ml")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use fluxframe_core::context::{FrameContext, FramingSource, ProcessingContext};
use fluxframe_core::error::InferenceError;
use fluxframe_core::frame::{FrameBuffer, FrameMeta, PixelFormat, Timestamp, VideoFrame};
use fluxframe_core::geometry::Region;
use fluxframe_core::plane::PlaneEffect;
use fluxframe_core::traits::{
    InferenceEngine, InferenceInput, InferenceOutput, ModelInfo, RawEffectParams, VideoEffect,
};
use fluxframe_effects::composite::{CompositeEffect, SegmentationBase, SegmentationConfig, Slot};
use fluxframe_effects::ml::ModelConfig;
use fluxframe_effects::plane_effects::ColorFillEffect;

const FILL: [u8; 3] = [1, 2, 3];

/// What the mock engine saw; shared with the test.
#[derive(Default)]
struct EngineLog {
    /// Frame indices whose pass was started.
    started: Vec<u8>,
    resets: usize,
    /// Fail the `wait` of the pass for this frame index.
    fail_wait_on: Option<u8>,
}

struct AsyncMock {
    log: Arc<Mutex<EngineLog>>,
    in_flight: Option<u8>,
}

impl AsyncMock {
    fn mask_for(k: u8) -> InferenceOutput {
        let mut data = vec![0.0_f32; 4];
        data[usize::from(k % 4)] = 1.0;
        InferenceOutput {
            data,
            shape: vec![2, 2],
        }
    }
}

impl InferenceEngine for AsyncMock {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: Arc::from("async-mock"),
            input_width: 2,
            input_height: 2,
            input_format: PixelFormat::Rgb,
        }
    }

    fn infer(&mut self, input: InferenceInput<'_>) -> Result<InferenceOutput, InferenceError> {
        self.start(input)?;
        self.wait()
    }

    fn supports_async(&self) -> bool {
        true
    }

    fn start(&mut self, input: InferenceInput<'_>) -> Result<(), InferenceError> {
        assert!(self.in_flight.is_none(), "one pass in flight at most");
        // NHWC with input_scale = 1: element 0 is the red byte of pixel 0.
        let k = (input.data[0] / 10.0).round() as u8;
        self.log.lock().unwrap().started.push(k);
        self.in_flight = Some(k);
        Ok(())
    }

    fn wait(&mut self) -> Result<InferenceOutput, InferenceError> {
        let k = self.in_flight.take().expect("wait after start");
        if self.log.lock().unwrap().fail_wait_on == Some(k) {
            return Err(InferenceError::InferenceFailed {
                reason: "forced".into(),
            });
        }
        Ok(Self::mask_for(k))
    }

    fn reset_state(&mut self) {
        self.in_flight = None;
        self.log.lock().unwrap().resets += 1;
    }
}

struct Sidecar {
    dir: PathBuf,
    model: PathBuf,
}

impl Sidecar {
    fn new(tag: &str, pipeline: bool) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "fluxframe-composite-pipeline-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let model = dir.join("m.onnx");
        std::fs::write(&model, b"").expect("model");
        std::fs::write(
            dir.join("m.toml"),
            format!(
                "name = \"mock\"\ninput_width = 2\ninput_height = 2\ninput_scale = 1.0\n\
                 output_type = \"mask\"\npipeline = {pipeline}\n"
            ),
        )
        .expect("sidecar");
        Self { dir, model }
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn composite(sidecar: &Sidecar, log: &Arc<Mutex<EngineLog>>) -> CompositeEffect {
    let engine_log = Arc::clone(log);
    let segmentation = SegmentationBase::new(SegmentationConfig {
        model: sidecar.model.clone(),
        fallback_threshold: 3,
    })
    .with_inference_factory(
        move |_: &Path,
              _: ModelConfig|
              -> Result<Box<dyn InferenceEngine + Send>, InferenceError> {
            Ok(Box::new(AsyncMock {
                log: engine_log,
                in_flight: None,
            }))
        },
    );
    let mut fill = ColorFillEffect::new(FILL);
    let params: RawEffectParams = toml::from_str("rgb = [1, 2, 3]").expect("toml");
    fill.configure(params).expect("configure");
    let bg: Vec<Slot<dyn PlaneEffect>> = vec![Slot::new(Box::new(fill), true)];
    let mut c = CompositeEffect::new(segmentation, Vec::new(), bg, Vec::new(), Vec::new());
    c.prepare(&ProcessingContext {
        width: 2,
        height: 2,
        format: PixelFormat::Rgb,
        fps: 30,
        counters: None,
    })
    .expect("prepare");
    log.lock().unwrap().started.clear();
    c
}

/// Frame `k`: every pixel `[k * 10, k, 200]`, sequence `k`.
fn frame(k: u8) -> VideoFrame {
    let data = [k * 10, k, 200].repeat(4);
    let meta = FrameMeta {
        sequence: u64::from(k),
        timestamp: Timestamp::from_nanos(u64::from(k) * 1_000),
        ..FrameMeta::default()
    };
    VideoFrame::new_packed(FrameBuffer::Owned(data), 2, 2, PixelFormat::Rgb, meta).expect("frame")
}

/// Context as the runtime builds it for frame `k`: its own sequence and
/// a framing source whose x origin is `k`.
fn ctx_for(k: u8) -> FrameContext {
    let mut ctx = FrameContext {
        frame_sequence: u64::from(k),
        frame_timestamp: Timestamp::from_nanos(u64::from(k) * 1_000),
        ..FrameContext::default()
    };
    ctx.framing.source = Some(FramingSource {
        width: 100,
        height: 100,
        region: Region {
            x: f32::from(k),
            ..Region::full(50, 50)
        },
    });
    ctx
}

/// Pixels of an emitted frame composed with frame `k`'s mask.
fn composed(k: u8) -> Vec<u8> {
    (0..4u8)
        .flat_map(|i| if i == k % 4 { [k * 10, k, 200] } else { FILL })
        .collect()
}

fn run(c: &mut CompositeEffect, k: u8) -> (VideoFrame, FrameContext) {
    let mut f = frame(k);
    let mut ctx = ctx_for(k);
    c.process(&mut f, &mut ctx).expect("process");
    (f, ctx)
}

#[test]
fn output_trails_input_by_one_frame_with_its_own_mask_and_metadata() {
    let sidecar = Sidecar::new("order", true);
    let log = Arc::new(Mutex::new(EngineLog::default()));
    let mut c = composite(&sidecar, &log);
    assert_eq!(c.latency_frames(), 1);

    let (_, ctx) = run(&mut c, 1);
    assert!(ctx.output_withheld, "nothing to emit for the first frame");

    for k in 2..=6 {
        let (out, ctx) = run(&mut c, k);
        let prev = k - 1;
        assert!(!ctx.output_withheld);
        assert!(!ctx.fallback_active);
        assert_eq!(out.meta.sequence, u64::from(prev), "frame N-1 is emitted");
        assert_eq!(ctx.frame_sequence, u64::from(prev));
        assert_eq!(
            ctx.framing.source.map(|s| s.region.x),
            Some(f32::from(prev)),
            "the framing source travels with its frame"
        );
        assert_eq!(
            out.data.as_slice(),
            composed(prev).as_slice(),
            "frame {prev}"
        );
    }
    assert_eq!(log.lock().unwrap().started, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn without_pipeline_flag_an_async_engine_runs_inline() {
    let sidecar = Sidecar::new("inline", false);
    let log = Arc::new(Mutex::new(EngineLog::default()));
    let mut c = composite(&sidecar, &log);
    assert_eq!(c.latency_frames(), 0);
    let (out, ctx) = run(&mut c, 3);
    assert!(!ctx.output_withheld);
    assert_eq!(out.data.as_slice(), composed(3).as_slice());
}

#[test]
fn failed_pass_emits_its_frame_unprocessed_and_resets_state() {
    let sidecar = Sidecar::new("fail", true);
    let log = Arc::new(Mutex::new(EngineLog::default()));
    let mut c = composite(&sidecar, &log);
    log.lock().unwrap().fail_wait_on = Some(2);
    let resets_before = log.lock().unwrap().resets;

    run(&mut c, 1);
    let (out, _) = run(&mut c, 2);
    assert_eq!(out.data.as_slice(), composed(1).as_slice());

    let (out, ctx) = run(&mut c, 3);
    assert!(ctx.fallback_active, "frame 2 had no mask");
    assert_eq!(out.meta.sequence, 2);
    assert_eq!(
        out.data.as_slice(),
        frame(2).data.as_slice(),
        "passed through"
    );
    assert_eq!(log.lock().unwrap().resets, resets_before + 1, "state reset");

    let (out, ctx) = run(&mut c, 4);
    assert!(!ctx.fallback_active, "recovers on the next pass");
    assert_eq!(out.data.as_slice(), composed(3).as_slice());
}

#[test]
fn abort_drops_the_held_frame_and_restarts_the_pipeline() {
    let sidecar = Sidecar::new("abort", true);
    let log = Arc::new(Mutex::new(EngineLog::default()));
    let mut c = composite(&sidecar, &log);
    run(&mut c, 1);
    run(&mut c, 2);
    let resets_before = log.lock().unwrap().resets;

    c.abort_pipeline();
    assert_eq!(log.lock().unwrap().resets, resets_before + 1);

    let (_, ctx) = run(&mut c, 3);
    assert!(ctx.output_withheld, "frame 2 was dropped with the abort");
    let (out, _) = run(&mut c, 4);
    assert_eq!(out.data.as_slice(), composed(3).as_slice());
}

#[test]
fn stale_held_frame_is_dropped() {
    let sidecar = Sidecar::new("stale", true);
    let log = Arc::new(Mutex::new(EngineLog::default()));
    let mut c = composite(&sidecar, &log);
    run(&mut c, 1);
    std::thread::sleep(std::time::Duration::from_millis(600));
    let (_, ctx) = run(&mut c, 2);
    assert!(ctx.output_withheld, "frame 1 is too old to emit");
    let (out, _) = run(&mut c, 3);
    assert_eq!(out.data.as_slice(), composed(2).as_slice());
}

#[test]
fn shutdown_aborts_the_pipeline() {
    let sidecar = Sidecar::new("shutdown", true);
    let log = Arc::new(Mutex::new(EngineLog::default()));
    let mut c = composite(&sidecar, &log);
    run(&mut c, 1);
    let resets_before = log.lock().unwrap().resets;
    c.shutdown().expect("shutdown");
    assert_eq!(log.lock().unwrap().resets, resets_before + 1);
}

#[test]
fn framing_jump_resets_model_state_but_keeps_the_stream() {
    let sidecar = Sidecar::new("jump", true);
    let log = Arc::new(Mutex::new(EngineLog::default()));
    let mut c = composite(&sidecar, &log);
    run(&mut c, 1);
    run(&mut c, 2);
    let resets_before = log.lock().unwrap().resets;

    // A one-pixel glide per frame (as in `ctx_for`) is not a jump.
    run(&mut c, 3);
    assert_eq!(log.lock().unwrap().resets, resets_before);

    // The window leaps by 20 px of a 50 px window.
    let mut f = frame(4);
    let mut ctx = ctx_for(4);
    if let Some(source) = ctx.framing.source.as_mut() {
        source.region.x = 24.0;
    }
    c.process(&mut f, &mut ctx).expect("process");
    assert_eq!(log.lock().unwrap().resets, resets_before + 1, "state reset");
    assert!(!ctx.output_withheld, "the held frame is still emitted");
    assert_eq!(f.data.as_slice(), composed(3).as_slice());
}
