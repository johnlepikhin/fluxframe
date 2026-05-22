//! End-to-end runtime supervisor.
//!
//! Owns the lifetime of input/output pipelines, the processing worker
//! thread and the Ctrl-C signal handler.  Stage 1 wires `videotestsrc`
//! → effect chain → fakesink/autovideosink only; V4L2 input lands in
//! Stage 2.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::{EffectError, PipelineError};
use fluxframe_core::{FluxConfig, FluxError};
use fluxframe_effects::{EffectChain, EffectRegistry, PassthroughEffect};
use fluxframe_gst::input::{InputParams, InputPipeline};
use fluxframe_gst::output::{OutputParams, OutputPipeline, OutputSink};
use tracing::{debug, error, info, warn};

/// How long the worker waits on an empty frame slot before re-checking the
/// shutdown flag.  Short enough to be responsive to Ctrl-C, long enough to
/// avoid spinning when the source briefly stalls.
const WORKER_POLL_TIMEOUT: Duration = Duration::from_millis(50);

/// Build a registry pre-populated with the effects available in Stage 1.
#[must_use]
pub fn default_registry() -> EffectRegistry {
    let mut registry = EffectRegistry::new();
    registry.register(
        PassthroughEffect::NAME,
        Box::new(|| -> Box<dyn fluxframe_core::traits::VideoEffect> {
            Box::new(PassthroughEffect::new())
        }),
    );
    registry
}

/// Decide whether `cfg` describes a synthetic source.
#[must_use]
pub fn is_testsrc_input(cfg: &FluxConfig) -> bool {
    use fluxframe_core::config::BackendKind;
    matches!(cfg.input.backend, BackendKind::Testsrc) || cfg.input.device == "testsrc"
}

/// Stage 1 entry point: run a passthrough-style chain on `videotestsrc`,
/// emitting to a fakesink or autovideosink.
///
/// # Errors
///
/// Propagates [`FluxError`] from pipeline construction, effect chain
/// preparation, or runtime failures.
pub fn run_testsrc_chain(cfg: &FluxConfig, mut chain: EffectChain) -> Result<(), FluxError> {
    fluxframe_gst::init()?;

    let input_params = InputParams {
        width: cfg.input.width,
        height: cfg.input.height,
        fps: cfg.input.fps,
        format: cfg.input.format,
    };
    let output_params = OutputParams {
        width: cfg.output.width,
        height: cfg.output.height,
        fps: cfg.output.fps,
        format: cfg.output.format,
        sink: resolve_output_sink(cfg),
    };

    let input = InputPipeline::build_testsrc(input_params)?;
    let output = OutputPipeline::build(output_params)?;

    let processing_ctx = ProcessingContext {
        width: cfg.input.width,
        height: cfg.input.height,
        format: cfg.input.format,
        fps: cfg.input.fps,
    };
    chain
        .prepare_all(&processing_ctx)
        .map_err(FluxError::from)?;

    info!(
        effects = ?chain.names(),
        input_format = ?cfg.input.format,
        output_format = ?cfg.output.format,
        "starting Stage 1 pipeline (testsrc -> {})",
        output_sink_label(output_params.sink)
    );

    output.start()?;
    input.start()?;

    let running = Arc::new(AtomicBool::new(true));
    install_ctrlc_handler(&running)?;

    let slot = input.slot();
    let shutdown_slot = slot.clone();
    let shutdown_flag = Arc::clone(&running);

    // Watchdog: closes the slot once the shutdown flag flips so the worker
    // wakes from any `take` immediately.
    let watchdog = thread::Builder::new()
        .name("fluxframe-watchdog".into())
        .spawn(move || {
            while shutdown_flag.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(100));
            }
            shutdown_slot.close();
        })
        .map_err(|e| {
            FluxError::from(PipelineError::Runtime {
                reason: format!("failed to spawn watchdog thread: {e}"),
            })
        })?;

    let worker_running = Arc::clone(&running);
    let worker_slot = slot.clone();

    // Process frames on the main thread so any failure short-circuits
    // shutdown without crossing thread boundaries.
    let result = process_loop(&worker_running, &worker_slot, &mut chain, &output);

    running.store(false, Ordering::Release);
    if let Err(e) = watchdog.join() {
        warn!(?e, "watchdog thread panicked during join");
    }

    if let Err(e) = input.stop() {
        warn!(error = %e, "input.stop failed");
    }
    if let Err(e) = output.stop() {
        warn!(error = %e, "output.stop failed");
    }
    if let Err(e) = chain.shutdown_all() {
        warn!(error = %e, "effect chain shutdown reported an error");
    }

    debug!(
        dropped = slot.dropped_count(),
        "capture-side dropped frames"
    );

    result
}

fn process_loop(
    running: &AtomicBool,
    slot: &fluxframe_gst::LatestFrameSlot,
    chain: &mut EffectChain,
    output: &OutputPipeline,
) -> Result<(), FluxError> {
    let mut frame_context = FrameContext::default();
    while running.load(Ordering::Acquire) {
        let Some(mut frame) = slot.take(WORKER_POLL_TIMEOUT) else {
            // Either timeout (no frame within the poll window) or slot
            // closed by shutdown.  Re-check the flag and continue.
            continue;
        };
        frame_context.frame_sequence = frame.meta.sequence;
        frame_context.frame_timestamp = frame.meta.timestamp;
        frame_context.fallback_active = false;

        if let Err(e) = chain.process(&mut frame, &mut frame_context) {
            error!(error = %e, "effect chain failed; stopping");
            return Err(map_effect_err(e));
        }
        if let Err(e) = output.push_frame(&frame) {
            error!(error = %e, "output.push_frame failed; stopping");
            return Err(e.into());
        }
    }
    Ok(())
}

fn install_ctrlc_handler(running: &Arc<AtomicBool>) -> Result<(), FluxError> {
    let flag = Arc::clone(running);
    ctrlc::set_handler(move || {
        info!("Ctrl-C received — shutting down");
        flag.store(false, Ordering::Release);
    })
    .map_err(|e| {
        FluxError::from(PipelineError::Runtime {
            reason: format!("failed to install Ctrl-C handler: {e}"),
        })
    })
}

fn resolve_output_sink(cfg: &FluxConfig) -> OutputSink {
    // Stage 1: route everything to fakesink unless the user explicitly
    // asked for the auto sink via the device string.  Stage 2 will map
    // a real `/dev/videoN` path onto `v4l2sink`.
    if cfg.output.device == "auto" {
        OutputSink::Auto
    } else {
        OutputSink::Fake
    }
}

fn output_sink_label(sink: OutputSink) -> &'static str {
    match sink {
        OutputSink::Fake => "fakesink",
        OutputSink::Auto => "autovideosink",
        _ => "unknown-sink",
    }
}

fn map_effect_err(e: EffectError) -> FluxError {
    FluxError::from(e)
}
