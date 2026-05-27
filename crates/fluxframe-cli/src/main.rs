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
mod logging;
mod metrics_reporter;
mod preset;
mod runtime;
mod runtime_metrics;

fn main() -> ExitCode {
    let args = cli::Cli::parse();

    if let Err(e) = logging::init(args.verbose) {
        eprintln!("failed to initialise logging: {e}");
        return ExitCode::from(2);
    }

    cap_rayon_pool();

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

/// Cores reserved for kernel work (USB-isoc soft IRQ servicing the UVC
/// camera).  Saturating Rayon across these starves the kernel and
/// produces torn camera frames upstream of our pipeline.
const RAYON_RESERVE_CORES: usize = 2;
/// Minimum rayon worker count — below this image processing throughput
/// drops noticeably for our 1280x720 working size.
const RAYON_MIN_THREADS: usize = 2;
/// Maximum rayon worker count — image-processing scaling flattens out
/// past this for our frame sizes; more threads only burn kernel time.
const RAYON_MAX_THREADS: usize = 4;

/// Decide the rayon worker-thread count from an optional env override
/// and the system's available parallelism.
///
/// If `env_override` is `Some` and parses to a positive `usize`, that
/// value wins verbatim (no clamping — operator opt-in).  Otherwise the
/// auto path reserves [`RAYON_RESERVE_CORES`] for the kernel and clamps
/// the result into `[RAYON_MIN_THREADS, RAYON_MAX_THREADS]`.
fn compute_rayon_target(env_override: Option<&str>, available: usize) -> usize {
    env_override
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| {
            available
                .saturating_sub(RAYON_RESERVE_CORES)
                .clamp(RAYON_MIN_THREADS, RAYON_MAX_THREADS)
        })
}

/// Pin Rayon's global thread pool to a small subset of cores.
///
/// On many-core machines (operator's setup is 22 cores) the default
/// `available_parallelism()` makes Rayon spread image-processing
/// parallelism across every core.  The composite/blur passes then
/// saturate every core's user-time, leaving the kernel's USB-isoc
/// soft-IRQ chronically late to service the UVC camera — packets
/// drop silently and we get torn camera frames *before* they reach
/// our pipeline.  Capping Rayon to a few cores eliminates that
/// cross-talk: image processing stays plenty fast (a 1280×720 blur
/// fits comfortably in 4 threads) and the rest of the box is free
/// for the kernel + system services.
///
/// `FLUXFRAME_RAYON_THREADS` overrides the auto-pick when set to a
/// positive integer.
fn cap_rayon_pool() {
    let env_value = std::env::var("FLUXFRAME_RAYON_THREADS").ok();
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(RAYON_MIN_THREADS);
    let target = compute_rayon_target(env_value.as_deref(), available);
    match rayon::ThreadPoolBuilder::new()
        .num_threads(target)
        .build_global()
    {
        Ok(()) => {
            tracing::debug!(threads = target, "rayon global pool capped");
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
    use super::compute_rayon_target;

    #[test]
    fn auto_path_caps_at_max_on_large_box() {
        assert_eq!(compute_rayon_target(None, 22), 4);
    }

    #[test]
    fn auto_path_clamps_to_floor_on_small_box() {
        // 3 - 2 = 1, clamped up to RAYON_MIN_THREADS (2).
        assert_eq!(compute_rayon_target(None, 3), 2);
    }

    #[test]
    fn auto_path_handles_tiny_box_via_saturating_sub() {
        // 1.saturating_sub(2) = 0, clamped up to RAYON_MIN_THREADS (2).
        assert_eq!(compute_rayon_target(None, 1), 2);
    }

    #[test]
    fn env_override_positive_wins_over_cap() {
        // Explicit operator opt-in bypasses the clamp ceiling.
        assert_eq!(compute_rayon_target(Some("8"), 22), 8);
    }

    #[test]
    fn env_override_zero_is_ignored() {
        // `0` would mean "no threads" — treat as unset, fall back to auto.
        assert_eq!(compute_rayon_target(Some("0"), 22), 4);
    }

    #[test]
    fn env_override_non_numeric_is_ignored() {
        assert_eq!(compute_rayon_target(Some("abc"), 22), 4);
    }

    #[test]
    fn env_override_empty_is_ignored() {
        assert_eq!(compute_rayon_target(Some(""), 22), 4);
    }

    #[test]
    fn env_override_negative_is_ignored() {
        // "-1" won't parse as usize, so the auto path runs.
        assert_eq!(compute_rayon_target(Some("-1"), 22), 4);
    }
}
