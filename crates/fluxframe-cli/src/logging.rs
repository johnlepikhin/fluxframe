//! `tracing` initialisation.  Verbosity from `-v` flags layers on top of
//! `RUST_LOG`, so `RUST_LOG=fluxframe_gst=trace` still works.

use anyhow::Result;
use tracing_subscriber::{EnvFilter, fmt};

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
    let default_level = match verbosity {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("fluxframe={default_level}")));

    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_level(true)
        .try_init()
        .map_err(|e| anyhow::anyhow!("tracing init failed: {e}"))?;

    Ok(())
}
