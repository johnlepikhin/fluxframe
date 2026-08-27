//! `tracing` initialisation.
//!
//! Three sources can set the filter, in descending precedence:
//!
//! 1. `RUST_LOG` — the operator's escape hatch, always wins.
//! 2. `-v` / `-vv` on the command line.
//! 3. `[logging] level` from the config file.
//!
//! The config file cannot participate at `init` time: the subscriber
//! must be installed before the config is parsed, because config-load
//! failures are themselves reported through `tracing`.  So `init`
//! installs a [`reload`]-able filter and [`apply_config_level`] swaps
//! it in later, once the daemon has a validated config — but only when
//! neither of the two higher-precedence sources spoke up.

use std::io::IsTerminal;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use tracing::{debug, warn};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::Registry;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt, reload};

/// Reload handle for the global env-filter, published by [`init`] and
/// consumed by [`apply_config_level`].
static FILTER_RELOAD: OnceLock<reload::Handle<EnvFilter, Registry>> = OnceLock::new();

/// `true` when the filter was pinned by `RUST_LOG` or by `-v`, so a
/// later config-driven update must not silently override the
/// operator's explicit choice.
static FILTER_PINNED: AtomicBool = AtomicBool::new(false);

/// Map a `-v` repeat count onto a level name.
fn verbosity_level(verbosity: u8) -> &'static str {
    match verbosity {
        0 => "info",
        1 => "debug",
        _ => "trace",
    }
}

/// Turn a `[logging] level` value into an `EnvFilter` directive string.
///
/// A bare level (`"info"`, `"debug"`) is scoped to the `fluxframe`
/// target, matching what [`init`] installs by default.  Without the
/// scoping, `level = "info"` would widen the filter to every dependency
/// — GStreamer, `ort`, `wgpu` — and bury our own output, which is the
/// opposite of what an operator setting a *level* expects.
///
/// Anything containing `=` is already a directive list and passes
/// through verbatim, so a per-target override such as
/// `"fluxframe=info,fluxframe::metrics=debug"` works as written.
fn normalise_directives(level: &str) -> String {
    let trimmed = level.trim();
    if trimmed.contains('=') {
        trimmed.to_string()
    } else {
        format!("fluxframe={trimmed}")
    }
}

/// Initialise the global `tracing` subscriber.
///
/// `verbosity` is the count of `-v` flags on the command line: `0` maps
/// to `info`, `1` to `debug`, anything higher to `trace`.  The default
/// filter is overridden if `RUST_LOG` is set in the environment.
///
/// # Errors
///
/// Returns an error if a global tracing subscriber has already been
/// installed (typically because `init` was called twice in the same
/// process).
pub fn init(verbosity: u8) -> Result<()> {
    let from_env = EnvFilter::try_from_default_env().ok();
    // Either higher-precedence source pins the filter for the process.
    FILTER_PINNED.store(from_env.is_some() || verbosity > 0, Ordering::Release);

    let filter = from_env
        .unwrap_or_else(|| EnvFilter::new(format!("fluxframe={}", verbosity_level(verbosity))));

    let (filter_layer, handle) = reload::Layer::new(filter);
    Registry::default()
        .with(filter_layer)
        .with(
            fmt::layer()
                .with_target(true)
                .with_level(true)
                // `fmt` colours unconditionally by default, and under a
                // service manager stdout is a log file — the escape
                // sequences end up in it verbatim, which makes the file
                // hostile to `grep` exactly when someone is reading it
                // during an incident.
                .with_ansi(std::io::stdout().is_terminal()),
        )
        .try_init()
        .map_err(|e| anyhow::anyhow!("tracing init failed: {e}"))?;

    // `set` can only fail if `init` ran twice, which `try_init` above
    // already rejected — so the error is unreachable rather than
    // ignorable, and dropping it keeps the signature clean.
    let _ = FILTER_RELOAD.set(handle);
    Ok(())
}

/// Apply the config file's `[logging] level` to the live subscriber.
///
/// No-op when `RUST_LOG` or `-v` already pinned the filter, when
/// [`init`] was never called (unit tests), or when the value is not a
/// valid filter expression — in the last case the current filter stays
/// and the operator gets a warning, because silently running at the
/// wrong verbosity is worse than a bad config line.
pub fn apply_config_level(level: &str) {
    if FILTER_PINNED.load(Ordering::Acquire) {
        debug!(
            configured = level,
            "[logging] level ignored — RUST_LOG or -v takes precedence"
        );
        return;
    }
    let Some(handle) = FILTER_RELOAD.get() else {
        return;
    };
    let directives = normalise_directives(level);
    match EnvFilter::try_new(&directives) {
        Ok(filter) => match handle.reload(filter) {
            Ok(()) => debug!(filter = %directives, "log filter updated from config"),
            Err(e) => warn!(error = %e, "failed to swap the log filter; keeping the current one"),
        },
        Err(e) => warn!(
            configured = level,
            error = %e,
            "[logging] level is not a valid filter expression; keeping the current filter"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{normalise_directives, verbosity_level};

    #[test]
    fn verbosity_maps_onto_levels() {
        assert_eq!(verbosity_level(0), "info");
        assert_eq!(verbosity_level(1), "debug");
        assert_eq!(verbosity_level(2), "trace");
        assert_eq!(verbosity_level(9), "trace");
    }

    #[test]
    fn bare_level_is_scoped_to_our_crates() {
        // The default config value must reproduce exactly what `init`
        // installs, otherwise merely having a config file would change
        // the daemon's verbosity.
        assert_eq!(normalise_directives("info"), "fluxframe=info");
        assert_eq!(normalise_directives("debug"), "fluxframe=debug");
    }

    #[test]
    fn surrounding_whitespace_is_tolerated() {
        assert_eq!(normalise_directives("  warn \n"), "fluxframe=warn");
    }

    #[test]
    fn directive_list_passes_through_verbatim() {
        let directives = "fluxframe=info,fluxframe::metrics=debug";
        assert_eq!(normalise_directives(directives), directives);
    }

    #[test]
    fn single_target_directive_is_not_rescoped() {
        assert_eq!(normalise_directives("gstreamer=trace"), "gstreamer=trace");
    }
}
