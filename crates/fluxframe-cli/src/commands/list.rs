//! `fluxframe list` — Stage 2 will implement V4L2 enumeration.

use fluxframe_core::FluxError;
use tracing::warn;

use crate::cli::ListArgs;

/// Entry point for `fluxframe list`.
///
/// # Errors
///
/// Reserved for future device-enumeration errors; the Stage 0 stub is
/// currently infallible.
#[expect(
    clippy::unnecessary_wraps,
    reason = "stage-0 stub; will return Result in stage 2"
)]
pub fn run(_args: ListArgs) -> Result<(), FluxError> {
    warn!("`list` is not implemented yet (planned for Stage 2)");
    Ok(())
}
