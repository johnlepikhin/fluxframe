//! `fluxframe check` — GStreamer init smoke test runs in Stage 0; full
//! device/model checks land in Stage 2/3.

use fluxframe_core::FluxError;
use tracing::{info, warn};

use crate::cli::CheckArgs;

/// Entry point for `fluxframe check`.
///
/// # Errors
///
/// Returns a [`FluxError::Pipeline`] when GStreamer initialisation fails.
pub fn run(_args: CheckArgs) -> Result<(), FluxError> {
    fluxframe_gst::init().map_err(FluxError::from)?;
    info!("gstreamer init: ok");
    warn!("device, model and effect-chain checks are not implemented yet (Stage 2/3)");
    Ok(())
}
