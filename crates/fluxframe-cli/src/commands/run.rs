//! `fluxframe run` — dispatch the configured input backend onto the
//! shared runtime supervisor (`testsrc` and V4L2 are both supported as
//! of Stage 2).

use fluxframe_core::traits::VideoEffect;
use fluxframe_core::{FluxError, normalise_effect_name};
use fluxframe_effects::EffectChain;
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

    let chain_names: Vec<String> = cfg
        .effects
        .chain
        .iter()
        .map(|n| normalise_effect_name(n))
        .collect();

    // Build the composite effect from the [mask]/[background]/[foreground]
    // sections (Stage 9). When present, it is prepended to whatever
    // `effects.chain` requested so additional filters can run after the
    // segmented composite.
    let mut effects: Vec<Box<dyn VideoEffect>> = Vec::new();
    #[cfg(feature = "ml")]
    if let Some(mask_section) = cfg.mask.as_ref() {
        use fluxframe_effects::composite::CompositeBuilder;
        use fluxframe_effects::{mask_effects, plane_effects};
        let mask_registry = mask_effects::default_registry();
        let plane_registry = plane_effects::default_registry();
        let composite = CompositeBuilder::new(&mask_registry, &plane_registry)
            .build(
                mask_section,
                cfg.background.as_ref(),
                cfg.foreground.as_ref(),
            )
            .map_err(FluxError::from)?;
        effects.push(Box::new(composite));
    }
    #[cfg(not(feature = "ml"))]
    if cfg.mask.is_some() {
        return Err(FluxError::Config {
            reason: "[mask] section present but the binary was built without the `ml` feature; \
                     rebuild with `--features fluxframe-effects/ml` or remove the [mask] section"
                .into(),
            hint: None,
        });
    }

    if !chain_names.is_empty() {
        let registry = default_registry();
        let standalone = registry
            .build_chain(&chain_names)
            .map_err(FluxError::from)?;
        effects.extend(standalone.into_effects());
    }

    if effects.is_empty() {
        return Err(FluxError::Config {
            reason: "effect chain is empty (no [mask] section and no [effects].chain)".into(),
            hint: Some(
                "either configure [mask]/[background] for the composite pipeline or list at \
                 least one effect under [effects].chain (e.g. passthrough)"
                    .into(),
            ),
        });
    }
    let chain = EffectChain::new(effects);

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
