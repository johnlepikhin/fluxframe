//! `fluxframe` CLI entry point.
//!
//! Parses arguments, initialises tracing, dispatches to a subcommand and
//! renders the §27 "Error / Hint" diagnostic format on stderr when a
//! command fails.

#![warn(missing_docs)]

use std::process::ExitCode;

use clap::Parser;
use tracing::error;

mod cli;
mod commands;
mod config_merge;
mod control;
mod idle;
mod logging;
mod metrics_reporter;
mod persist;
mod preset;
mod runtime;
mod runtime_metrics;
mod thread_cpu;

fn main() -> ExitCode {
    let args = cli::Cli::parse();

    if let Err(e) = logging::init(args.verbose) {
        eprintln!("failed to initialise logging: {e}");
        return ExitCode::from(2);
    }

    match commands::dispatch(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            use fluxframe_core::Diagnostic;
            // Print §27 canonical format on stderr independent of RUST_LOG,
            // unless the producer (e.g. `fluxframe check`) already rendered
            // each underlying failure via `FluxError::Aggregated`.  Doing
            // the §27 render twice would print the same lines for the
            // primary failure and confuse the operator.
            if !e.is_aggregated() {
                eprintln!("Error: {}", e.reason());
                if let Some(hint) = e.hint() {
                    eprintln!("Hint: {hint}");
                }
            }
            error!(error = %e, "command failed");
            ExitCode::FAILURE
        }
    }
}

/// Rayon worker count when neither the environment nor the config
/// picks one.
///
/// Two, not "cores minus a reserve": the frame-resolution kernels
/// (resize, compose, sharpen, pixelate, vignette, mirror) are
/// memory-bound, so extra threads mostly wait on the same DRAM and the
/// wait shows up as CPU time.  Measured at 800×448 on a 22-thread
/// Meteor Lake: 4 threads — 32 % of a core, processing p50 8.8 ms;
/// 3 — 29 %, 9.1 ms; 2 — 27.6 %, 9.7 ms.  The frame budget at 25–30 fps
/// is 33–40 ms, so the millisecond is free and the CPU is not.
const RAYON_AUTO_THREADS: usize = 2;

/// Decide the rayon worker-thread count.  Precedence: the
/// `FLUXFRAME_RAYON_THREADS` environment override (a positive
/// integer, taken verbatim), then a positive `[realtime]
/// processing_threads` from the config, then [`RAYON_AUTO_THREADS`].
fn compute_rayon_target(env_override: Option<&str>, config_threads: u32) -> usize {
    env_override
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .or_else(|| usize::try_from(config_threads).ok().filter(|n| *n > 0))
        .unwrap_or(RAYON_AUTO_THREADS)
}

/// Build Rayon's global thread pool with a deliberately small worker
/// count.  Must run before anything touches rayon, and after the
/// config is loaded — `commands::run` calls it right after validation.
///
/// The cap exists for two reasons.  On many-core machines the default
/// `available_parallelism()` pool spread the image kernels across
/// every core, and the saturated user-time left the kernel's USB-isoc
/// soft-IRQ chronically late to service the UVC camera — packets
/// dropped silently and frames arrived torn *before* they reached the
/// pipeline.  And past a couple of threads the kernels are
/// memory-bound, so more workers only burn CPU waiting on DRAM (see
/// [`RAYON_AUTO_THREADS`]).
pub(crate) fn cap_rayon_pool(config_threads: u32) {
    let env_value = std::env::var("FLUXFRAME_RAYON_THREADS").ok();
    let target = compute_rayon_target(env_value.as_deref(), config_threads);
    // Named so the metrics reporter's per-thread CPU accounting can
    // tell pool workers from everything else (see `thread_cpu`).
    match rayon::ThreadPoolBuilder::new()
        .num_threads(target)
        .thread_name(|i| format!("rayon-{i}"))
        .build_global()
    {
        Ok(()) => {
            tracing::info!(threads = target, "rayon pool sized");
        }
        Err(e) => {
            // build_global() can only run once per process; subsequent
            // calls error.  In a CLI binary this only happens if a
            // dependency raced ahead of `main`, which is unlikely but
            // not a hard failure.
            tracing::debug!(error = %e, "rayon pool already initialised; cap not applied");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RAYON_AUTO_THREADS, compute_rayon_target};

    #[test]
    fn auto_when_nothing_is_set() {
        assert_eq!(compute_rayon_target(None, 0), RAYON_AUTO_THREADS);
    }

    #[test]
    fn config_positive_wins_over_auto() {
        assert_eq!(compute_rayon_target(None, 3), 3);
    }

    #[test]
    fn env_override_positive_wins_over_config() {
        // Explicit operator opt-in, taken verbatim.
        assert_eq!(compute_rayon_target(Some("8"), 3), 8);
    }

    #[test]
    fn env_override_zero_is_ignored() {
        // `0` would mean "no threads" — treat as unset, fall through.
        assert_eq!(compute_rayon_target(Some("0"), 0), RAYON_AUTO_THREADS);
        assert_eq!(compute_rayon_target(Some("0"), 3), 3);
    }

    #[test]
    fn env_override_non_numeric_is_ignored() {
        assert_eq!(compute_rayon_target(Some("abc"), 0), RAYON_AUTO_THREADS);
    }

    #[test]
    fn env_override_empty_is_ignored() {
        assert_eq!(compute_rayon_target(Some(""), 0), RAYON_AUTO_THREADS);
    }

    #[test]
    fn env_override_negative_is_ignored() {
        // "-1" won't parse as usize, so the next source is used.
        assert_eq!(compute_rayon_target(Some("-1"), 0), RAYON_AUTO_THREADS);
    }
}
