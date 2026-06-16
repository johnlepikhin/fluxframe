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
//! 2. If we were in `DeepIdle`, the ONNX session was dropped — it
//!    has to be rebuilt via [`fluxframe_effects::backend::
//!    build_inference_engine`], which is another 300–700 ms of disk
//!    + ORT init.
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

use fluxframe_gst::input::InputPipeline;
use tracing::{info, warn};

/// Result of a single resume attempt — exposed for tests and future
/// supervisor instrumentation. `Ok` carries the wall-clock cost of
/// the warmup so the supervisor can publish a `resume_latency_ms`
/// metric later (Step 5).
#[derive(Debug)]
pub(crate) struct ResumeOutcome {
    /// Wall-clock time from the start of the reload thread to the
    /// instant `engine_ready` flipped to `true`.
    pub elapsed: std::time::Duration,
}

/// Spawn a one-shot reload thread.
///
/// The thread:
///
/// 1. Calls `input.start()` to drive the input pipeline `Null →
///    Playing`. On failure, logs and exits; `engine_ready` stays
///    `false` so the worker remains in placeholder mode and the
///    supervisor's control socket can issue `reload` to retry.
/// 2. Runs `engine_reloader` — caller-supplied closure that does the
///    expensive ONNX rebuild (or is a no-op when the engine was
///    never dropped). The closure returns a `Result<(), _>`; on
///    error, same fallback as #1.
/// 3. Flips `engine_ready` to `true` so the worker resumes the full
///    chain on its next iteration.
///
/// The closure form keeps `fluxframe-cli::idle` free of a hard
/// dependency on `fluxframe-effects` — the supervisor (which has
/// both crates in scope) constructs the closure with whatever
/// engine-rebuild path the active `ml` feature selects.
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

            if let Err(e) = input.start() {
                warn!(
                    target: "fluxframe::idle",
                    error = %e,
                    "reload: input.start failed — staying in placeholder mode"
                );
                return ResumeOutcome {
                    elapsed: start.elapsed(),
                };
            }

            if let Err(reason) = engine_reloader() {
                warn!(
                    target: "fluxframe::idle",
                    error = %reason,
                    "reload: engine rebuild failed — staying in placeholder mode"
                );
                return ResumeOutcome {
                    elapsed: start.elapsed(),
                };
            }

            engine_ready.store(true, Ordering::Release);
            let elapsed = start.elapsed();
            info!(
                target: "fluxframe::idle",
                elapsed_ms = elapsed.as_millis() as u64,
                "reload thread completed — full chain restored"
            );
            ResumeOutcome { elapsed }
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
        assert!(outcome.elapsed > std::time::Duration::ZERO);
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
        let _outcome = handle.join().expect("thread join");
        assert!(
            !engine_ready.load(Ordering::Acquire),
            "engine_ready must stay false when reloader returns Err"
        );
        input.stop().expect("stop pipeline");
    }
}
