//! `fluxframe benchmark` — implemented in Stage 3 (inference benchmark)
//! and Stage 5 (full pipeline benchmark).

use fluxframe_core::FluxError;
use tracing::warn;

use crate::cli::BenchmarkArgs;

/// Entry point for `fluxframe benchmark`.
///
/// # Errors
///
/// Reserved for future benchmark-pipeline errors; the Stage 0 stub is
/// currently infallible.
#[expect(
    clippy::unnecessary_wraps,
    reason = "stage-0 stub; will return Result in stage 3/5"
)]
pub fn run(_args: BenchmarkArgs) -> Result<(), FluxError> {
    warn!("`benchmark` is not implemented yet (planned for Stage 3/5)");
    Ok(())
}
