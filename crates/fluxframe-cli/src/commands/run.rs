//! `fluxframe run` — dispatch the configured input backend onto the
//! shared runtime supervisor (`testsrc` and V4L2 are both supported as
//! of Stage 2).

use fluxframe_core::{FluxError, normalise_effect_name};
use tracing::info;

use crate::cli::RunArgs;
use crate::config_merge::{CliOverrides, apply, load};
use crate::runtime::{
    default_registry, is_testsrc_input, is_v4l2_input, run_testsrc_chain, run_v4l2_chain,
};

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
    let chain = registry
        .build_chain(&chain_names)
        .map_err(FluxError::from)?;

    // Dispatch order: `testsrc` first so an explicit `--input testsrc` wins
    // over the `/dev/` heuristic in `is_v4l2_input`.  Anything not matched
    // by either rule (e.g. a future RTSP URL) is rejected with a hint
    // listing the currently supported inputs.
    if is_testsrc_input(&cfg) {
        run_testsrc_chain(&cfg, chain)
    } else if is_v4l2_input(&cfg) {
        run_v4l2_chain(&cfg, chain)
    } else {
        Err(FluxError::Config {
            reason: format!("input '{}' is not handled", cfg.input.device),
            hint: Some("supported inputs: testsrc, /dev/video* (V4L2)".into()),
        })
    }
}
