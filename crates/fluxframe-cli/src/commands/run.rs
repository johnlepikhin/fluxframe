//! `fluxframe run` — pipeline construction lands in Stage 1.

use fluxframe_core::FluxError;
use tracing::{info, warn};

use crate::cli::RunArgs;
use crate::config_merge::{CliOverrides, apply, load};

/// Entry point for `fluxframe run`.
///
/// # Errors
///
/// Returns a [`FluxError`] when the config file cannot be loaded, the
/// merged configuration fails validation, or (in later stages) when
/// the pipeline fails to construct or run.
pub fn run(args: RunArgs) -> Result<(), FluxError> {
    let overrides = CliOverrides {
        input: args.common.input,
        output: args.common.output,
        effect: args.common.effect,
        width: args.width,
        height: args.height,
        fps: args.fps,
    };

    let cfg = load(args.common.config.as_deref())?;
    let cfg = apply(cfg, &overrides);
    cfg.validate()?;

    info!(
        input = %cfg.input.device,
        output = %cfg.output.device,
        width = cfg.input.width,
        height = cfg.input.height,
        fps = cfg.input.fps,
        chain = ?cfg.effects.chain,
        "merged configuration"
    );

    warn!("`run` pipeline is not implemented yet (planned for Stage 1)");
    Ok(())
}
