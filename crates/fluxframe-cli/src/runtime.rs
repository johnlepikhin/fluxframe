//! End-to-end runtime supervisor.
//!
//! Owns the lifetime of input/output pipelines, the processing worker
//! thread and the Ctrl-C signal handler.  Stage 1 wired `videotestsrc`
//! → effect chain → fakesink/autovideosink only; Stage 2 adds the V4L2
//! capture and v4l2loopback sink paths.
//!
//! GStreamer is intentionally absent from this module's surface: bus
//! events arrive through the typed [`fluxframe_gst::BusEvent`] enum,
//! draining is owned by [`fluxframe_gst::BusListener`], and the supervisor
//! itself only juggles `std::sync` primitives.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::PipelineError;
use fluxframe_core::{FluxConfig, FluxError};
use fluxframe_effects::{EffectChain, EffectRegistry, PassthroughEffect};
use fluxframe_gst::input::{InputParams, InputPipeline};
use fluxframe_gst::output::{OutputParams, OutputPipeline, OutputSink};
use fluxframe_gst::{BusEvent, BusListener, LatestFrameSlot, WatchedPipeline};
use tracing::{error, info, warn};

/// How long the worker waits on an empty frame slot before re-checking the
/// shutdown flag.  Short enough to be responsive to Ctrl-C, long enough to
/// avoid spinning when the source briefly stalls.
const WORKER_POLL_TIMEOUT: Duration = Duration::from_millis(50);

// TODO(stage-5/metrics): re-introduce a periodic drop-summary tick once the
// metrics subsystem owns observability.  The previous implementation lived
// inside the bus listener thread, which was the wrong layer: the bus thread
// should only translate GStreamer messages, not poll counters.  For Stage 1
// we log the cumulative drop count once at teardown.

/// Process-wide registry of live [`RunToken`]s.  The Ctrl-C signal handler
/// walks this list and broadcasts a shutdown request to every concurrent
/// run.  Tokens are stored as [`Weak`] references so a run that has
/// already torn down does not keep its slot alive.
static REGISTERED_TOKENS: OnceLock<Mutex<Vec<Weak<RunToken>>>> = OnceLock::new();

fn tokens_registry() -> &'static Mutex<Vec<Weak<RunToken>>> {
    REGISTERED_TOKENS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Per-run shutdown state.  Held inside an [`Arc`] so the signal handler
/// can flip the flag and close the slot without owning the run.
struct RunToken {
    /// Shutdown flag: `true` while the run is active, set to `false` by
    /// the signal handler or the supervisor itself to request termination.
    flag: Arc<AtomicBool>,
    /// Frame slot to close when shutdown is requested, so the worker
    /// loop wakes from `recv_timeout` immediately.
    slot: LatestFrameSlot,
}

/// Install the Ctrl-C handler at most once for the entire process.  If the
/// handler cannot be installed (e.g. another component already claimed it)
/// we log a warning and continue without one: tests routinely run inside
/// harnesses that grab SIGINT for themselves.
fn install_ctrlc_once() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let result = ctrlc::set_handler(|| {
            info!("Ctrl-C received - broadcasting shutdown");
            let mut reg = tokens_registry()
                .lock()
                .expect("tokens registry poisoned");
            reg.retain(|w| w.upgrade().is_some());
            for token in reg.iter().filter_map(Weak::upgrade) {
                token.flag.store(false, Ordering::Release);
                token.slot.close();
            }
        });
        if let Err(e) = result {
            warn!(
                error = %e,
                "could not install Ctrl-C handler (another one is already set); continuing without one",
            );
        }
    });
}

/// RAII guard that owns a [`RunToken`] for the duration of a single run.
///
/// On drop the guard removes its token's weak reference from the global
/// registry, which keeps the registry from growing unboundedly when many
/// runs come and go inside the same process (the integration test suite
/// is the obvious caller).
struct TokenGuard {
    token: Arc<RunToken>,
}

impl Drop for TokenGuard {
    fn drop(&mut self) {
        let reg = tokens_registry();
        let mut tokens = reg.lock().expect("tokens registry poisoned");
        tokens.retain(|w| {
            w.upgrade()
                .is_some_and(|other| !Arc::ptr_eq(&other, &self.token))
        });
    }
}

/// Register a fresh [`RunToken`] for the current run and return a guard
/// plus the per-run shutdown flag.  The guard must remain in scope until
/// the run finishes; dropping it removes the token from the registry.
fn register_token(slot: LatestFrameSlot) -> (TokenGuard, Arc<AtomicBool>) {
    install_ctrlc_once();
    let token = Arc::new(RunToken {
        flag: Arc::new(AtomicBool::new(true)),
        slot,
    });
    let flag = Arc::clone(&token.flag);
    let weak = Arc::downgrade(&token);
    tokens_registry()
        .lock()
        .expect("tokens registry poisoned")
        .push(weak);
    (TokenGuard { token }, flag)
}

/// Build a registry pre-populated with the effects available in Stage 1.
#[must_use]
pub(crate) fn default_registry() -> EffectRegistry {
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
pub(crate) fn is_testsrc_input(cfg: &FluxConfig) -> bool {
    use fluxframe_core::config::BackendKind;
    matches!(cfg.input.backend, BackendKind::Testsrc) || cfg.input.device == "testsrc"
}

/// Decide whether `cfg` describes a V4L2 capture device.
///
/// Both the explicit `BackendKind::V4l2` selection and a `/dev/...` device
/// path under `BackendKind::Auto` route to the V4L2 capture pipeline.  The
/// path-prefix heuristic is documented in the Stage 2 plan: with the
/// `Auto` default, the only way the user can mean "real camera" is to
/// point at a kernel device node.
#[must_use]
pub(crate) fn is_v4l2_input(cfg: &FluxConfig) -> bool {
    use fluxframe_core::config::BackendKind;
    matches!(cfg.input.backend, BackendKind::V4l2) || cfg.input.device.starts_with("/dev/")
}

/// Run a passthrough-style chain on `videotestsrc`, emitting to the sink
/// resolved from the configuration.
///
/// # Errors
///
/// Propagates [`FluxError`] from pipeline construction, effect chain
/// preparation, or runtime failures.
#[tracing::instrument(skip_all, fields(sink = ?cfg.output.device))]
pub(crate) fn run_testsrc_chain(cfg: &FluxConfig, chain: EffectChain) -> Result<(), FluxError> {
    run_chain(cfg, chain, "testsrc", InputPipeline::build_testsrc)
}

/// Run the effect chain against a V4L2 capture device.
///
/// # Errors
///
/// Propagates [`FluxError`] from pipeline construction (including a
/// structured [`PipelineError::InputDeviceUnavailable`] when the
/// pre-open check fails), effect chain preparation, or runtime failures.
#[tracing::instrument(skip_all, fields(device = %cfg.input.device, sink = ?cfg.output.device))]
pub(crate) fn run_v4l2_chain(cfg: &FluxConfig, chain: EffectChain) -> Result<(), FluxError> {
    let device = PathBuf::from(&cfg.input.device);
    run_chain(cfg, chain, "v4l2src", move |params| {
        InputPipeline::build_v4l2(device, params)
    })
}

/// Shared driver for all input backends: takes a closure that builds the
/// input pipeline so the bus-listener / processing-loop / teardown plumbing
/// lives in exactly one place.
fn run_chain<F>(
    cfg: &FluxConfig,
    mut chain: EffectChain,
    source_label: &'static str,
    input_builder: F,
) -> Result<(), FluxError>
where
    F: FnOnce(InputParams) -> Result<InputPipeline, PipelineError>,
{
    fluxframe_gst::init()?;

    let (input, output, processing_ctx, sink_label) = build_pipelines(cfg, input_builder)?;
    chain
        .prepare_all(&processing_ctx)
        .map_err(FluxError::from)?;

    info!(
        effects = ?chain.names(),
        input_format = ?cfg.input.format,
        output_format = ?cfg.output.format,
        "starting pipeline ({source_label} -> {sink_label})",
    );

    output.start()?;
    input.start()?;

    let slot = input.slot();
    // The token guard MUST live until the end of this function: when it
    // drops it removes the run's weak entry from the global registry.
    let (_token_guard, running) = register_token(slot.clone());

    // Bus listener relays GStreamer fatal errors and EOS into the shared
    // shutdown flag, and surfaces any captured error back to the caller.
    let bus_error: Arc<Mutex<Option<FluxError>>> = Arc::new(Mutex::new(None));
    let _bus_listener = build_bus_listener(
        &input,
        &output,
        Arc::clone(&running),
        Arc::clone(&bus_error),
        slot.clone(),
    );

    let process_result = run_process_loop(&running, &slot, &mut chain, &output);

    // Ensure the bus listener wakes and exits.  Dropping `_bus_listener`
    // at the end of the function calls `BusListener::stop` via Drop, but
    // we also flip the flag here so the listener observes shutdown even
    // before the drop runs.
    running.store(false, Ordering::Release);
    slot.close();

    teardown(&input, &output, &mut chain);

    tracing::debug!(
        dropped = slot.dropped_count(),
        "total capture-side dropped frames",
    );

    // Surface bus-reported errors when the processing loop itself was
    // clean, so the operator sees the real cause of shutdown.
    let bus_err = bus_error.lock().expect("bus_error mutex poisoned").take();
    match (process_result, bus_err) {
        (Ok(()), Some(e)) | (Err(e), _) => Err(e),
        (Ok(()), None) => Ok(()),
    }
}

fn build_pipelines<F>(
    cfg: &FluxConfig,
    input_builder: F,
) -> Result<
    (
        InputPipeline,
        OutputPipeline,
        ProcessingContext,
        &'static str,
    ),
    FluxError,
>
where
    F: FnOnce(InputParams) -> Result<InputPipeline, PipelineError>,
{
    let input_params = InputParams::new(
        cfg.input.width,
        cfg.input.height,
        cfg.input.fps,
        cfg.input.format,
    );
    let sink = resolve_output_sink(cfg);
    let sink_label = output_sink_label(&sink);
    let output_params = OutputParams::new(
        cfg.output.width,
        cfg.output.height,
        cfg.output.fps,
        cfg.output.format,
        sink,
    );

    let input = input_builder(input_params)?;
    let output = OutputPipeline::build(output_params)?;

    let processing_ctx = ProcessingContext {
        width: cfg.input.width,
        height: cfg.input.height,
        format: cfg.input.format,
        fps: cfg.input.fps,
    };

    Ok((input, output, processing_ctx, sink_label))
}

/// Build the [`BusListener`] watching both pipelines.  The listener is
/// returned so the caller can keep it alive (its `Drop` joins the thread).
fn build_bus_listener(
    input: &InputPipeline,
    output: &OutputPipeline,
    running: Arc<AtomicBool>,
    bus_error: Arc<Mutex<Option<FluxError>>>,
    slot: LatestFrameSlot,
) -> BusListener {
    let pipelines = vec![
        WatchedPipeline {
            label: "input",
            bus: input.bus(),
        },
        WatchedPipeline {
            label: "output",
            bus: output.bus(),
        },
    ];
    BusListener::spawn(pipelines, move |event| {
        on_bus_event(&event, &running, &bus_error, &slot);
    })
}

/// Translate a [`BusEvent`] into supervisor side-effects.  Extracted so
/// the closure passed to [`BusListener::spawn`] stays a one-liner and so
/// the handler is unit-testable.
fn on_bus_event(
    event: &BusEvent,
    running: &AtomicBool,
    bus_error: &Mutex<Option<FluxError>>,
    slot: &LatestFrameSlot,
) {
    match event {
        BusEvent::FatalError {
            element,
            message,
            debug: debug_payload,
            source_label,
        } => {
            error!(
                source = %source_label,
                %element,
                %message,
                debug = ?debug_payload,
                "bus fatal error",
            );
            let err = FluxError::from(PipelineError::BusError {
                element: element.clone(),
                message: message.clone(),
                debug: debug_payload.clone(),
            });
            let mut guard = bus_error.lock().expect("bus_error mutex poisoned");
            if guard.is_none() {
                *guard = Some(err);
            }
            running.store(false, Ordering::Release);
            slot.close();
        }
        BusEvent::Warning {
            source_label,
            element,
            message,
            debug: debug_payload,
        } => {
            warn!(
                source = %source_label,
                %element,
                %message,
                debug = ?debug_payload,
                "bus warning",
            );
        }
        BusEvent::Eos { source_label } => {
            info!(source = %source_label, "pipeline EOS");
            running.store(false, Ordering::Release);
            slot.close();
        }
        // `BusEvent` is `#[non_exhaustive]`: future variants land here
        // until the supervisor is taught to interpret them.
        _ => {
            warn!(?event, "unhandled bus event variant");
        }
    }
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
    // Routing rules (see Stage 2 plan, §"Output side"):
    //   * "auto"      -> autovideosink (manual glance verification).
    //   * "fakesink"  -> fakesink (CI / dev smoke without a loopback).
    //   * /dev/...    -> v4l2sink to a loopback device.
    //   * everything else -> fakesink (CI / tests).
    match cfg.output.device.as_str() {
        "auto" => OutputSink::Auto,
        "fakesink" => OutputSink::Fake,
        device if device.starts_with("/dev/") => OutputSink::V4l2Loopback {
            device: PathBuf::from(device),
        },
        _ => OutputSink::Fake,
    }
}

fn output_sink_label(sink: &OutputSink) -> &'static str {
    // `OutputSink` is not `#[non_exhaustive]` inside the workspace, so
    // this match is genuinely exhaustive: adding a variant will produce
    // a compile-time prompt here.  Borrowed because `V4l2Loopback` owns
    // a `PathBuf` and is therefore no longer `Copy`.
    match sink {
        OutputSink::Fake => "fakesink",
        OutputSink::Auto => "autovideosink",
        OutputSink::V4l2Loopback { .. } => "v4l2sink",
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
    fn resolve_output_sink_maps_dev_path_to_v4l2_loopback() {
        let mut cfg = base_cfg();
        cfg.output.device = "/dev/video10".into();
        assert_eq!(
            resolve_output_sink(&cfg),
            OutputSink::V4l2Loopback {
                device: PathBuf::from("/dev/video10"),
            }
        );
    }

    #[test]
    fn resolve_output_sink_defaults_to_fake() {
        let mut cfg = base_cfg();
        cfg.output.device = "fakesink".into();
        assert_eq!(resolve_output_sink(&cfg), OutputSink::Fake);
    }

    #[test]
    fn output_sink_label_covers_each_variant() {
        assert_eq!(output_sink_label(&OutputSink::Fake), "fakesink");
        assert_eq!(output_sink_label(&OutputSink::Auto), "autovideosink");
        assert_eq!(
            output_sink_label(&OutputSink::V4l2Loopback {
                device: PathBuf::from("/dev/video10"),
            }),
            "v4l2sink"
        );
    }

    #[test]
    fn is_v4l2_input_recognises_backend() {
        let mut cfg = base_cfg();
        cfg.input.backend = BackendKind::V4l2;
        cfg.input.device = "/dev/video0".into();
        assert!(is_v4l2_input(&cfg));
    }

    #[test]
    fn is_v4l2_input_recognises_dev_path() {
        let mut cfg = base_cfg();
        cfg.input.backend = BackendKind::Auto;
        cfg.input.device = "/dev/video2".into();
        assert!(is_v4l2_input(&cfg));
    }

    #[test]
    fn is_v4l2_input_rejects_testsrc() {
        let mut cfg = base_cfg();
        cfg.input.backend = BackendKind::Auto;
        cfg.input.device = "testsrc".into();
        assert!(!is_v4l2_input(&cfg));
    }
}
