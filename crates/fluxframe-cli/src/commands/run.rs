//! `fluxframe run` — Stage 1 wires synthetic source → effect chain → sink.
//! V4L2 input lands in Stage 2.

use fluxframe_core::error::EffectError;
use fluxframe_core::{FluxError, normalise_effect_name};
use tracing::info;

use crate::cli::RunArgs;
use crate::config_merge::{CliOverrides, apply, load};
use crate::runtime::{default_registry, is_testsrc_input, run_testsrc_chain};

/// Entry point for `fluxframe run`.
///
/// # Errors
///
/// Returns a [`FluxError`] when the config file cannot be loaded, the
/// merged configuration fails validation, the effect chain cannot be
/// built, or the runtime pipeline fails.
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

    let chain_names: Vec<String> = if cfg.effects.chain.is_empty() {
        return Err(FluxError::Config {
            reason: "effect chain is empty".into(),
            hint: Some("pass --effect passthrough or set [effects].chain in the config".into()),
        });
    } else {
        cfg.effects
            .chain
            .iter()
            .map(|n| normalise_effect_name(n))
            .collect()
    };

    let registry = default_registry();
    let chain = registry.build_chain(&chain_names).map_err(map_effect_err)?;

    if is_testsrc_input(&cfg) {
        run_testsrc_chain(&cfg, chain)
    } else {
        Err(FluxError::Config {
            reason: format!("input '{}' is not handled in Stage 1", cfg.input.device),
            hint: Some("use --input testsrc for Stage 1; V4L2 capture lands in Stage 2".into()),
        })
    }
}

fn map_effect_err(e: EffectError) -> FluxError {
    FluxError::from(e)
}
