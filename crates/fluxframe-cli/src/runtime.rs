//! End-to-end runtime supervisor.
//!
//! Owns the lifetime of input/output pipelines, the processing worker
//! thread and the Ctrl-C signal handler.  Stage 1 wires `videotestsrc`
//! → effect chain → fakesink/autovideosink only; V4L2 input lands in
//! Stage 2.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::PipelineError;
use fluxframe_core::{FluxConfig, FluxError};
use fluxframe_effects::{EffectChain, EffectRegistry, PassthroughEffect};
use fluxframe_gst::LatestFrameSlot;
use fluxframe_gst::input::{InputParams, InputPipeline};
use fluxframe_gst::output::{OutputParams, OutputPipeline, OutputSink};
use gstreamer::MessageView;
use gstreamer::prelude::*;
use parking_lot::Mutex;
use tracing::{debug, error, info, warn};

/// How long the worker waits on an empty frame slot before re-checking the
/// shutdown flag.  Short enough to be responsive to Ctrl-C, long enough to
/// avoid spinning when the source briefly stalls.
const WORKER_POLL_TIMEOUT: Duration = Duration::from_millis(50);

/// How long the bus listener blocks on `timed_pop_filtered`.  Same trade-off
/// as `WORKER_POLL_TIMEOUT`: short enough to react to shutdown promptly.
const BUS_POLL_TIMEOUT: Duration = Duration::from_millis(50);

/// Periodic interval at which the bus listener logs the cumulative drop
/// counter (only when it changed since the previous report).
const DROP_REPORT_INTERVAL: Duration = Duration::from_secs(5);

/// Globally-installed Ctrl-C state.  The first invocation registers the
/// signal handler; subsequent invocations reuse the same flag and slot
/// registry.  Without this, calling [`run_testsrc_chain`] twice in the same
/// process (e.g. integration tests) would panic in `ctrlc::set_handler`.
static SHUTDOWN_INSTALLED: OnceLock<Arc<AtomicBool>> = OnceLock::new();
static SLOTS: OnceLock<Mutex<Vec<LatestFrameSlot>>> = OnceLock::new();

fn slots_registry() -> &'static Mutex<Vec<LatestFrameSlot>> {
    SLOTS.get_or_init(|| Mutex::new(Vec::new()))
}

fn install_or_reuse_ctrlc() -> Arc<AtomicBool> {
    // `get_or_init` cannot return `Result`, so we install via a small wrapper
    // that panics on the very first install failure.  `set_handler` only
    // fails once globally (subsequent installations would error with
    // `MultipleHandlers`), so the panic here is unreachable in well-formed
    // code paths.
    SHUTDOWN_INSTALLED
        .get_or_init(|| {
            let flag = Arc::new(AtomicBool::new(true));
            let flag_handler = Arc::clone(&flag);
            ctrlc::set_handler(move || {
                info!("Ctrl-C received - shutting down");
                flag_handler.store(false, Ordering::Release);
                for slot in slots_registry().lock().iter() {
                    slot.close();
                }
            })
            .expect("Ctrl-C handler installed once at process start");
            flag
        })
        .clone()
}

fn register_slot(slot: LatestFrameSlot) {
    slots_registry().lock().push(slot);
}

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

    let (input, output, processing_ctx, sink_label) = build_pipelines(cfg)?;
    chain.prepare_all(&processing_ctx).map_err(FluxError::from)?;

    info!(
        effects = ?chain.names(),
        input_format = ?cfg.input.format,
        output_format = ?cfg.output.format,
        "starting Stage 1 pipeline (testsrc -> {sink_label})",
    );

    output.start()?;
    input.start()?;

    let running = install_or_reuse_ctrlc();
    // Re-arm the flag in case a previous run flipped it off (e.g. tests
    // exercising the same process twice).  Safe to do before registering
    // the new slot: the Ctrl-C handler reads the flag, not the running
    // state of any specific pipeline.
    running.store(true, Ordering::Release);

    let slot = input.slot();
    register_slot(slot.clone());

    // Bus listener relays GStreamer fatal errors and EOS into the shared
    // shutdown flag, and surfaces any captured error back to the caller.
    let bus_error: Arc<Mutex<Option<FluxError>>> = Arc::new(Mutex::new(None));
    let bus_handle = spawn_bus_listener(
        input.pipeline_for_bus(),
        output.pipeline_for_bus(),
        Arc::clone(&running),
        slot.clone(),
        Arc::clone(&bus_error),
    )?;

    let process_result = run_process_loop(&running, &slot, &mut chain, &output);

    // Ensure the bus listener wakes and exits.
    running.store(false, Ordering::Release);
    slot.close();
    if let Err(e) = bus_handle.join() {
        warn!(?e, "bus listener thread panicked during join");
    }

    teardown(&input, &output, &mut chain);

    debug!(
        dropped = slot.dropped_count(),
        "capture-side dropped frames"
    );

    // Surface bus-reported errors when the processing loop itself was
    // clean, so the operator sees the real cause of shutdown.
    let bus_err = bus_error.lock().take();
    match (process_result, bus_err) {
        (Ok(()), Some(e)) | (Err(e), _) => Err(e),
        (Ok(()), None) => Ok(()),
    }
}

fn build_pipelines(
    cfg: &FluxConfig,
) -> Result<
    (
        InputPipeline,
        OutputPipeline,
        ProcessingContext,
        &'static str,
    ),
    FluxError,
> {
    let input_params = InputParams::new(
        cfg.input.width,
        cfg.input.height,
        cfg.input.fps,
        cfg.input.format,
    );
    let sink = resolve_output_sink(cfg);
    let output_params = OutputParams::new(
        cfg.output.width,
        cfg.output.height,
        cfg.output.fps,
        cfg.output.format,
        sink,
    );

    let input = InputPipeline::build_testsrc(input_params)?;
    let output = OutputPipeline::build(output_params)?;

    let processing_ctx = ProcessingContext {
        width: cfg.input.width,
        height: cfg.input.height,
        format: cfg.input.format,
        fps: cfg.input.fps,
    };

    Ok((input, output, processing_ctx, output_sink_label(sink)))
}

fn spawn_bus_listener(
    input_pipeline: &gstreamer::Pipeline,
    output_pipeline: &gstreamer::Pipeline,
    running: Arc<AtomicBool>,
    slot: LatestFrameSlot,
    bus_error: Arc<Mutex<Option<FluxError>>>,
) -> Result<JoinHandle<()>, FluxError> {
    let buses = [
        ("input", input_pipeline.bus()),
        ("output", output_pipeline.bus()),
    ];
    let Some(buses): Option<Vec<(&'static str, gstreamer::Bus)>> = buses
        .into_iter()
        .map(|(name, bus)| bus.map(|b| (name, b)))
        .collect()
    else {
        return Err(FluxError::from(PipelineError::Runtime {
            reason: "GStreamer pipeline has no bus".into(),
        }));
    };

    let initial_dropped = slot.dropped_count();

    thread::Builder::new()
        .name("fluxframe-bus".into())
        .spawn(move || {
            let mut last_dropped = initial_dropped;
            let mut last_report = Instant::now();
            while running.load(Ordering::Acquire) {
                for (label, bus) in &buses {
                    let Some(msg) = bus.timed_pop_filtered(
                        gstreamer::ClockTime::from_mseconds(
                            BUS_POLL_TIMEOUT.as_millis() as u64,
                        ),
                        &[
                            gstreamer::MessageType::Error,
                            gstreamer::MessageType::Warning,
                            gstreamer::MessageType::Eos,
                        ],
                    ) else {
                        continue;
                    };
                    match msg.view() {
                        MessageView::Error(err) => {
                            let reason = format!(
                                "{} bus error from {:?}: {} (debug: {})",
                                label,
                                err.src().map(|s| s.path_string().to_string()),
                                err.error(),
                                err.debug().unwrap_or_default(),
                            );
                            error!(error = %reason, "pipeline fatal error");
                            let mut guard = bus_error.lock();
                            if guard.is_none() {
                                *guard = Some(FluxError::from(PipelineError::Runtime {
                                    reason,
                                }));
                            }
                            running.store(false, Ordering::Release);
                            slot.close();
                        }
                        MessageView::Warning(w) => {
                            warn!(
                                bus = label,
                                error = %w.error(),
                                debug = ?w.debug(),
                                "pipeline warning",
                            );
                        }
                        MessageView::Eos(_) => {
                            info!(bus = label, "pipeline EOS");
                            running.store(false, Ordering::Release);
                            slot.close();
                        }
                        _ => {}
                    }
                }

                if last_report.elapsed() >= DROP_REPORT_INTERVAL {
                    let current = slot.dropped_count();
                    if current != last_dropped {
                        info!(
                            dropped = current,
                            "capture-side drop summary",
                        );
                        last_dropped = current;
                    }
                    last_report = Instant::now();
                }
            }
        })
        .map_err(|e| {
            FluxError::from(PipelineError::Runtime {
                reason: format!("failed to spawn bus listener thread: {e}"),
            })
        })
}

fn run_process_loop(
    running: &AtomicBool,
    slot: &LatestFrameSlot,
    chain: &mut EffectChain,
    output: &OutputPipeline,
) -> Result<(), FluxError> {
    let mut frame_context = FrameContext::default();
    while running.load(Ordering::Acquire) {
        let Some(mut frame) = slot.recv_timeout(WORKER_POLL_TIMEOUT) else {
            // Either timeout (no frame within the poll window) or slot
            // closed by shutdown.  Re-check the flag and continue.
            continue;
        };
        frame_context.frame_sequence = frame.meta.sequence;
        frame_context.frame_timestamp = frame.meta.timestamp;
        frame_context.fallback_active = false;

        if let Err(e) = chain.process(&mut frame, &mut frame_context) {
            error!(error = %e, "effect chain failed; stopping");
            return Err(FluxError::from(e));
        }
        if let Err(e) = output.push_frame(frame) {
            error!(error = %e, "output.push_frame failed; stopping");
            return Err(e.into());
        }
    }
    Ok(())
}

fn teardown(input: &InputPipeline, output: &OutputPipeline, chain: &mut EffectChain) {
    if let Err(e) = input.stop() {
        warn!(error = %e, "input.stop failed");
    }
    if let Err(e) = output.stop() {
        warn!(error = %e, "output.stop failed");
    }
    if let Err(e) = chain.shutdown_all() {
        warn!(error = %e, "effect chain shutdown reported an error");
    }
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
        // `OutputSink` is `#[non_exhaustive]` so the match must include a
        // catch-all.  Inside this crate it is exhaustive at compile time,
        // but the lint requires a wildcard arm.
        _ => "unknown-sink",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxframe_core::config::BackendKind;

    fn base_cfg() -> FluxConfig {
        // Default Test configuration: start from `FluxConfig::default()`.
        // The fields tweaked here are the ones exercised by each test.
        FluxConfig::default()
    }

    #[test]
    fn is_testsrc_input_recognises_backend_enum() {
        let mut cfg = base_cfg();
        cfg.input.backend = BackendKind::Testsrc;
        cfg.input.device = "anything".into();
        assert!(is_testsrc_input(&cfg));
    }

    #[test]
    fn is_testsrc_input_recognises_device_string() {
        let mut cfg = base_cfg();
        cfg.input.backend = BackendKind::V4l2;
        cfg.input.device = "testsrc".into();
        assert!(is_testsrc_input(&cfg));
    }

    #[test]
    fn is_testsrc_input_rejects_real_device() {
        let mut cfg = base_cfg();
        cfg.input.backend = BackendKind::V4l2;
        cfg.input.device = "/dev/video0".into();
        assert!(!is_testsrc_input(&cfg));
    }

    #[test]
    fn resolve_output_sink_maps_auto() {
        let mut cfg = base_cfg();
        cfg.output.device = "auto".into();
        assert_eq!(resolve_output_sink(&cfg), OutputSink::Auto);
    }

    #[test]
    fn resolve_output_sink_defaults_to_fake() {
        let mut cfg = base_cfg();
        cfg.output.device = "/dev/video10".into();
        assert_eq!(resolve_output_sink(&cfg), OutputSink::Fake);
    }

    #[test]
    fn output_sink_label_covers_each_variant() {
        assert_eq!(output_sink_label(OutputSink::Fake), "fakesink");
        assert_eq!(output_sink_label(OutputSink::Auto), "autovideosink");
    }
}
