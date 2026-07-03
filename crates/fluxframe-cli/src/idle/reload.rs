//! Off-worker reload coordinator.
//!
//! The Stage 15 supervisor reaches `ResumeActive` when the consumer
//! detector flips back to `Present` after one or more idle ticks. At
//! that point two heavy operations have to happen before real frames
//! can flow again:
//!
//! 1. The input GStreamer pipeline (currently in `Null`) is brought
//!    back to `Playing`, which on UVC cameras takes 50–300 ms for
//!    USB renegotiation.
//! 2. The ONNX session is still resident (it is kept warm across
//!    `Idle` — the `DeepIdle` state that used to drop it was removed
//!    in Stage 16), so `engine_reloader` is a no-op today. The hook is
//!    kept for the future RAM-reclamation path that would rebuild the
//!    session via [`fluxframe_effects::backend::build_inference_engine`]
//!    (another 300–700 ms of disk + ORT init).
//!
//! Doing either on the worker thread would stall the placeholder
//! cadence and leave the just-reconnected consumer staring at the
//! stale ring-buffer frame for the full warmup window. The
//! supervisor instead spawns a one-shot thread that does both
//! synchronously and flips `engine_ready` to `true` only when the
//! whole chain is back online. The worker keeps publishing the
//! placeholder at `idle.fps` until then, so the consumer sees a
//! living stream throughout the cold start.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use fluxframe_core::FluxError;
use fluxframe_gst::input::InputPipeline;
use tracing::{debug, info};

/// Outcome of a single resume attempt. The three variants map 1:1 to
/// the reload thread's three exits so the supervisor can act on each
/// distinctly — crucially, only [`ResumeOutcome::InputFailed`] implies
/// a possibly-gone camera and should trigger a chain restart + device
/// re-enumeration; an [`ResumeOutcome::EngineFailed`] is an ORT/disk
/// problem and must not.
#[derive(Debug)]
pub(crate) enum ResumeOutcome {
    /// The full chain was restored. The warmup duration is logged by the
    /// reload thread itself (at info), so it is not carried here.
    Completed,
    /// `input.reacquire()` failed — the camera may be gone or have
    /// re-enumerated onto a different node. Transient `FluxError` so the
    /// supervisor can bounce the chain into `run_auto` re-enumeration.
    InputFailed(FluxError),
    /// The engine rebuild closure failed (ORT/disk). Not a camera
    /// problem — the worker stays in placeholder mode.
    EngineFailed(String),
}

/// Spawn a one-shot reload thread.
///
/// The thread:
///
/// 1. Calls [`InputPipeline::reacquire`] to drive the input pipeline
///    `Null → Playing` (re-opening the camera device). On failure,
///    returns [`ResumeOutcome::InputFailed`]; `engine_ready` stays
///    `false` and the supervisor restarts the chain to re-select the
///    camera.
/// 2. Runs `engine_reloader` — caller-supplied closure that does the
///    expensive ONNX rebuild (or is a no-op when the engine was never
///    dropped). On error, returns [`ResumeOutcome::EngineFailed`].
/// 3. Flips `engine_ready` to `true` and returns
///    [`ResumeOutcome::Completed`] so the worker resumes the full chain.
///
/// The closure form keeps `fluxframe-cli::idle` free of a hard
/// dependency on `fluxframe-effects` — the supervisor (which has both
/// crates in scope) constructs the closure with whatever engine-rebuild
/// path the active `ml` feature selects.
pub(crate) fn spawn_reload_thread<F>(
    input: Arc<InputPipeline>,
    engine_ready: Arc<AtomicBool>,
    engine_reloader: F,
) -> JoinHandle<ResumeOutcome>
where
    F: FnOnce() -> Result<(), String> + Send + 'static,
{
    thread::Builder::new()
        .name("fluxframe-reload".into())
        .spawn(move || {
            let start = Instant::now();
            info!(target: "fluxframe::idle", "reload thread started");

            if let Err(e) = input.reacquire() {
                // `debug` not `warn`: the reaper (`tick_idle`) re-logs this
                // at `warn` with the restart decision, so warning here too
                // would triple-log one failure (reload thread → reaper →
                // run_auto). The structural `InputFailed` outcome is the
                // authoritative signal.
                debug!(
                    target: "fluxframe::idle",
                    error = %e,
                    "reload: input reacquire failed"
                );
                return ResumeOutcome::InputFailed(FluxError::from(e));
            }

            if let Err(reason) = engine_reloader() {
                // `debug` not `warn`, same as the input path above: the
                // reaper (`tick_idle`) re-logs the `EngineFailed` outcome
                // at `warn`, so warning here too would double-log.
                debug!(
                    target: "fluxframe::idle",
                    error = %reason,
                    "reload: engine rebuild failed"
                );
                return ResumeOutcome::EngineFailed(reason);
            }

            engine_ready.store(true, Ordering::Release);
            info!(
                target: "fluxframe::idle",
                elapsed_ms = start.elapsed().as_millis() as u64,
                "reload thread completed — full chain restored"
            );
            ResumeOutcome::Completed
        })
        .expect("failed to spawn fluxframe-reload OS thread")
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxframe_core::frame::PixelFormat;
    use fluxframe_gst::input::{InputParams, InputPipeline};

    fn make_test_input() -> Arc<InputPipeline> {
        fluxframe_gst::init().expect("gst init");
        let params = InputParams::new(64, 36, 5, PixelFormat::Rgb);
        Arc::new(InputPipeline::build_testsrc(params).expect("testsrc pipeline"))
    }

    #[test]
    fn happy_path_flips_engine_ready() {
        let input = make_test_input();
        let engine_ready = Arc::new(AtomicBool::new(false));
        let handle = spawn_reload_thread(Arc::clone(&input), Arc::clone(&engine_ready), || Ok(()));
        let outcome = handle.join().expect("thread join");
        assert!(
            engine_ready.load(Ordering::Acquire),
            "engine_ready must flip on happy path"
        );
        assert!(
            matches!(outcome, ResumeOutcome::Completed),
            "happy path yields Completed, got {outcome:?}"
        );
        // Tidy up: bring the pipeline back to a state where Drop is
        // safe (testsrc otherwise holds the gst loop briefly).
        input.stop().expect("stop pipeline");
    }

    #[test]
    fn engine_reload_failure_keeps_engine_ready_false() {
        let input = make_test_input();
        let engine_ready = Arc::new(AtomicBool::new(false));
        let handle = spawn_reload_thread(Arc::clone(&input), Arc::clone(&engine_ready), || {
            Err("synthetic ORT init failure".into())
        });
        let outcome = handle.join().expect("thread join");
        assert!(
            !engine_ready.load(Ordering::Acquire),
            "engine_ready must stay false when reloader returns Err"
        );
        assert!(
            matches!(outcome, ResumeOutcome::EngineFailed(_)),
            "reloader error yields EngineFailed, got {outcome:?}"
        );
        input.stop().expect("stop pipeline");
    }
}
