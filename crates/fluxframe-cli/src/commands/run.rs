//! `fluxframe run` — dispatch the configured input backend onto the
//! shared runtime supervisor (`testsrc` and V4L2 are both supported as
//! of Stage 2).

use fluxframe_core::{FluxError, normalise_effect_name};
use tracing::info;

use crate::cli::RunArgs;
use crate::config_merge::{CliOverrides, apply, load};
use crate::runtime::{
    InputSpec, classify_input, default_registry, run_testsrc_chain, run_v4l2_chain,
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
        model: args.common.model,
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

    // Dispatch via `classify_input` so the testsrc-vs-V4L2 triage lives in
    // exactly one place (shared with `commands::check`).  Adding a variant
    // to `InputSpec` turns this into a compile error and prompts the
    // developer to teach the dispatch about it.
    match classify_input(&cfg) {
        InputSpec::Testsrc => run_testsrc_chain(&cfg, chain),
        InputSpec::V4l2(_) => run_v4l2_chain(&cfg, chain),
        InputSpec::Unsupported(d) => Err(FluxError::Config {
            reason: format!("input '{d}' is not supported"),
            hint: Some("supported inputs: testsrc, /dev/video* (V4L2)".into()),
        }),
    }
}
