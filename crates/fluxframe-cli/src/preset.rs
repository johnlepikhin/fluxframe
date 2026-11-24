//! Named-preset resolution and pipeline construction.
//!
//! Single source of truth for the `--preset NAME` flag, shared between
//! `fluxframe run` and `fluxframe check` so they always agree on
//! diagnostics and feature-gated behaviour.
//!
//! Two responsibilities live here:
//!
//! 1. [`resolve`] — turn an optional `--preset NAME` into a
//!    `(name, &Preset)` pair, with a structured §27 error when the
//!    requested preset (or the implicit `default`) is missing.
//! 2. [`build_chain`] — turn a `&Preset` into an [`EffectChain`],
//!    enforcing the configuration invariants both subcommands rely on
//!    (bg/fg without mask is a config bug; mask without the `ml`
//!    feature is a build mismatch).

use fluxframe_core::context::ProcessingContext;
use fluxframe_core::paths::ConfigBase;
use fluxframe_core::traits::VideoEffect;
use fluxframe_core::{FluxConfig, FluxError, Preset};
use fluxframe_effects::{EffectChain, PassthroughEffect};
use tracing::{info, warn};

/// Default preset name looked up when `--preset` is omitted.
pub const DEFAULT_PRESET_NAME: &str = "default";

/// Resolve the preset selected by the CLI.
///
/// Rules:
/// - `--preset NAME` given → exact lookup; structured error with the
///   list of available presets if missing.
/// - `--preset` omitted → [`DEFAULT_PRESET_NAME`]; structured error
///   pointing at the missing default if absent.
///
/// Returns the resolved name *and* a reference to the preset itself so
/// callers do not have to re-enter `cfg.presets.get(...)` later (and
/// the existence proof is carried in the type).
///
/// # Errors
///
/// Returns [`FluxError::Config`] when the requested preset (or the
/// implicit `default`) does not exist in `cfg.presets`.
pub fn resolve<'a>(
    cfg: &'a FluxConfig,
    requested: Option<&str>,
) -> Result<(&'a str, &'a Preset), FluxError> {
    let name = requested.unwrap_or(DEFAULT_PRESET_NAME);
    if let Some((key, preset)) = cfg.presets.get_key_value(name) {
        return Ok((key.as_str(), preset));
    }
    Err(resolution_error(cfg, requested))
}

/// Build the structured error returned when a preset cannot be
/// resolved. Centralised so both `run` and `check` produce identical
/// diagnostics.
fn resolution_error(cfg: &FluxConfig, requested: Option<&str>) -> FluxError {
    let available: Vec<&str> = cfg.presets.keys().map(String::as_str).collect();
    let reason = match requested {
        Some(name) => format!("preset '{name}' is not defined in the config"),
        None => format!("no --preset given and no '{DEFAULT_PRESET_NAME}' preset in the config"),
    };
    let hint = if available.is_empty() {
        format!("add a [presets.{DEFAULT_PRESET_NAME}] section to your fluxframe.toml")
    } else {
        format!("available presets: {}", available.join(", "))
    };
    FluxError::Config {
        reason,
        hint: Some(hint),
    }
}

/// A preset whose chain is built and prepared: the one asked for, or
/// a fallback of it ([`prepare_with_fallback`]).
pub(crate) struct PreparedPreset {
    pub(crate) chain: EffectChain,
    pub(crate) name: String,
    pub(crate) preset: Preset,
}

/// Build and prepare the chain of preset `name`.  While a preset cannot
/// run for lack of an accelerator and names a `fallback_preset`, the
/// fallback is tried instead (fallback presets come from `cfg`, saved
/// state).
///
/// # Errors
///
/// The last preset tried fails to build or prepare, or a fallback is
/// missing from `cfg` or revisits a preset (both rejected by config
/// validation).
pub(crate) fn prepare_with_fallback(
    cfg: &FluxConfig,
    config_base: &ConfigBase,
    name: &str,
    preset: &Preset,
    ctx: &ProcessingContext,
) -> Result<PreparedPreset, FluxError> {
    let mut name = name.to_string();
    let mut preset = preset.clone();
    let mut tried = Vec::new();
    loop {
        let mut chain = build_chain(&name, &preset, config_base)?;
        let err = match chain.prepare_all(ctx) {
            Ok(()) => {
                return Ok(PreparedPreset {
                    chain,
                    name,
                    preset,
                });
            }
            Err(e) => e,
        };
        let fallback = preset
            .fallback_preset
            .clone()
            .filter(|_| err.is_accelerator_unavailable());
        let Some(fallback) = fallback else {
            return Err(err.into());
        };
        tried.push(name);
        if tried.contains(&fallback) {
            return Err(FluxError::Config {
                reason: format!("fallback_preset cycle through '{fallback}'"),
                hint: None,
            });
        }
        let next = cfg
            .presets
            .get(&fallback)
            .ok_or_else(|| FluxError::Config {
                reason: format!("fallback_preset '{fallback}' is not defined"),
                hint: None,
            })?;
        warn!(
            preset = %tried.last().map_or("", String::as_str),
            fallback = %fallback,
            error = %err,
            "preset cannot run on this machine; switching to its fallback preset",
        );
        name = fallback;
        preset = next.clone();
    }
}

/// Build the [`EffectChain`] for the given preset.
///
/// - Preset with `[mask]` → composite chain (segmentation + sub-chains).
/// - Preset with no `[mask]`, no `[background]`, no `[foreground]` →
///   single [`PassthroughEffect`] so the chain stays non-empty.
/// - Anything else → structured error (bg/fg without mask is a config
///   bug regardless of feature flags; mask without `ml` is a build
///   mismatch).
///
/// Relative paths in the preset (the model, image parameters) are
/// resolved against `config_base`.
///
/// # Errors
///
/// Returns [`FluxError::Config`] when the preset declares
/// `[background]` or `[foreground]` without a `[mask]` section (those
/// sub-pipelines are only wired into the composite, so they would be a
/// silent no-op otherwise), or when the preset declares `[mask]` on a
/// build compiled without the `ml` feature. Composite-builder failures
/// (unknown effect names, malformed per-effect TOML, missing
/// `mask.model`, ...) propagate via `FluxError::from(EffectError)`.
pub fn build_chain(
    name: &str,
    preset: &Preset,
    config_base: &ConfigBase,
) -> Result<EffectChain, FluxError> {
    reject_planes_without_mask(name, preset)?;
    if preset.mask.is_some() {
        return build_composite_chain(name, preset, config_base);
    }
    Ok(build_passthrough_chain(name))
}

/// Hard-reject `[background]`/`[foreground]` declared without a
/// `[mask]` section.
///
/// The composite is only assembled when `mask` is present and the
/// plane sub-chains are *only* wired into the composite, so a preset
/// with bg/fg-only would silently drop the operator's configuration.
/// Surface that as a structured error in both `run` and `check`,
/// independent of the `ml` feature.
fn reject_planes_without_mask(name: &str, preset: &Preset) -> Result<(), FluxError> {
    if preset.mask.is_some() {
        return Ok(());
    }
    if preset.background.is_none() && preset.foreground.is_none() {
        return Ok(());
    }
    let mut sections: Vec<&'static str> = Vec::new();
    if preset.background.is_some() {
        sections.push("background");
    }
    if preset.foreground.is_some() {
        sections.push("foreground");
    }
    Err(FluxError::Config {
        reason: format!(
            "preset '{name}' declares [{}] but no [mask] section; \
             plane sub-pipelines are only applied through the composite \
             (which requires [mask])",
            sections.join("] / ["),
        ),
        hint: Some(
            "add a [mask] section with a segmentation `model = \"...\"` path, \
             or remove the background/foreground sections"
                .into(),
        ),
    })
}

/// Build a single-element chain wrapping a [`PassthroughEffect`] so
/// the runtime always sees a non-empty chain.
fn build_passthrough_chain(name: &str) -> EffectChain {
    info!(
        preset = %name,
        "preset has no [mask] section — running passthrough chain",
    );
    let effects: Vec<Box<dyn VideoEffect>> = vec![Box::new(PassthroughEffect::new())];
    EffectChain::new(effects)
}

/// Build a single-element chain wrapping the [`composite`] effect.
///
/// Returns a structured error on builds compiled without the `ml`
/// feature so the diagnostic matches the one `check` would print
/// (rather than the previous silent fall-through to passthrough).
///
/// [`composite`]: fluxframe_effects::composite
#[cfg(feature = "ml")]
fn build_composite_chain(
    name: &str,
    preset: &Preset,
    config_base: &ConfigBase,
) -> Result<EffectChain, FluxError> {
    use fluxframe_effects::composite::CompositeBuilder;
    use fluxframe_effects::{mask_effects, plane_effects, post_effects};

    // Unwrap is safe: `build_chain` only calls this branch after the
    // `preset.mask.is_some()` check.
    let mask_section = preset
        .mask
        .as_ref()
        .expect("build_composite_chain called without a [mask] section");
    let mask_registry = mask_effects::default_registry();
    let plane_registry = plane_effects::default_registry();
    let post_registry = post_effects::default_registry();
    let composite = CompositeBuilder::new(&mask_registry, &plane_registry, &post_registry)
        .build(
            mask_section,
            preset.background.as_ref(),
            preset.foreground.as_ref(),
            preset.post.as_ref(),
            config_base,
        )
        .map_err(FluxError::from)?;
    info!(
        preset = %name,
        "preset has a [mask] section — running composite chain",
    );
    let effects: Vec<Box<dyn VideoEffect>> = vec![Box::new(composite)];
    Ok(EffectChain::new(effects))
}

#[cfg(not(feature = "ml"))]
fn build_composite_chain(
    name: &str,
    _preset: &Preset,
    _config_base: &ConfigBase,
) -> Result<EffectChain, FluxError> {
    Err(FluxError::Config {
        reason: format!(
            "preset '{name}' has a [mask] sub-section but the binary was built \
             without the `ml` feature"
        ),
        hint: Some(
            "rebuild with `--features fluxframe-effects/ml` or remove the mask \
             section from the preset"
                .into(),
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "ml")]
    use crate::test_support::UnrunnableModel;
    use fluxframe_core::PipelineSection;

    fn cfg_with_preset(name: &str, body: &str) -> FluxConfig {
        let text = format!("[presets.{name}]\n{body}");
        FluxConfig::from_toml_str(&text).expect("preset toml parses")
    }

    // ---------- resolve ----------

    #[test]
    fn resolve_default_when_none_given() {
        let cfg = cfg_with_preset(DEFAULT_PRESET_NAME, "");
        let (name, _preset) = resolve(&cfg, None).expect("default resolves");
        assert_eq!(name, DEFAULT_PRESET_NAME);
    }

    #[test]
    fn resolve_fails_when_no_default_and_no_flag() {
        let cfg = cfg_with_preset("custom", "");
        let err = resolve(&cfg, None).expect_err("missing default must surface an error");
        let msg = format!("{err}");
        assert!(
            msg.contains(DEFAULT_PRESET_NAME),
            "expected '{DEFAULT_PRESET_NAME}' mention: {msg}",
        );
    }

    #[test]
    fn resolve_fails_when_named_preset_missing() {
        let cfg = cfg_with_preset(DEFAULT_PRESET_NAME, "");
        let err = resolve(&cfg, Some("nope")).expect_err("missing requested preset must error");
        let msg = format!("{err}");
        assert!(msg.contains("'nope'"), "expected requested name: {msg}");
    }

    #[test]
    fn resolve_picks_explicit_flag() {
        let cfg = cfg_with_preset("custom", "");
        let (name, _preset) = resolve(&cfg, Some("custom")).expect("custom resolves");
        assert_eq!(name, "custom");
    }

    // ---------- build_chain ----------

    #[test]
    fn build_chain_falls_back_to_passthrough_when_empty() {
        let preset = Preset::default();
        let chain = build_chain("raw", &preset, &ConfigBase::default())
            .expect("empty preset → passthrough");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.names(), vec![PassthroughEffect::NAME]);
    }

    #[test]
    fn build_chain_rejects_bg_without_mask() {
        let preset = Preset {
            background: Some(PipelineSection::default()),
            ..Preset::default()
        };
        // `EffectChain` is not `Debug`, so use a `let-else` instead of
        // `expect_err` to keep the panic message useful.
        let Err(err) = build_chain("bg-only", &preset, &ConfigBase::default()) else {
            panic!("bg without mask must surface as a config error");
        };
        let msg = format!("{err}");
        assert!(msg.contains("background"), "got: {msg}");
        assert!(msg.contains("mask"), "got: {msg}");
    }

    #[test]
    fn build_chain_rejects_fg_without_mask() {
        let preset = Preset {
            foreground: Some(PipelineSection::default()),
            ..Preset::default()
        };
        let Err(err) = build_chain("fg-only", &preset, &ConfigBase::default()) else {
            panic!("fg without mask must surface as a config error");
        };
        let msg = format!("{err}");
        assert!(msg.contains("foreground"), "got: {msg}");
        assert!(msg.contains("mask"), "got: {msg}");
    }

    /// Locate the repository's bundled selfie-segmentation model. The
    /// composite builder reads `model = "..."` from the preset and
    /// stores the path, but it does NOT load the file at build time
    /// (loading happens lazily in `prepare`), so any path will do for
    /// a build-only smoke test — we use the real file to stay close to
    /// production reality.
    #[cfg(feature = "ml")]
    fn segmentation_model_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/selfie_segmentation.onnx")
    }

    #[cfg(feature = "ml")]
    #[test]
    fn build_chain_with_mask_returns_composite_chain() {
        use fluxframe_effects::CompositeEffect;
        let mask = PipelineSection {
            model: Some(segmentation_model_path()),
            ..PipelineSection::default()
        };
        let preset = Preset {
            mask: Some(mask),
            ..Preset::default()
        };

        let Ok(chain) = build_chain("composite", &preset, &ConfigBase::default()) else {
            panic!("mask present → composite chain should build");
        };
        // Composite collapses the whole sub-pipeline into a single
        // `VideoEffect`, so the outer chain length is exactly 1 and
        // the single effect's name is the composite identifier (not
        // the passthrough fallback).
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.names(), vec![CompositeEffect::NAME]);
    }

    #[cfg(not(feature = "ml"))]
    #[test]
    fn build_chain_rejects_mask_without_ml_feature() {
        let mask = PipelineSection {
            model: Some(std::path::PathBuf::from("/tmp/dummy.onnx")),
            ..PipelineSection::default()
        };
        let preset = Preset {
            mask: Some(mask),
            ..Preset::default()
        };

        let Err(err) = build_chain("composite", &preset, &ConfigBase::default()) else {
            panic!("mask without ml feature must surface as a config error");
        };
        let msg = format!("{err}");
        assert!(msg.contains("ml"), "expected 'ml' mention: {msg}");
    }

    // ---------- prepare_with_fallback ----------

    /// A model restricted to a device no machine has: preparing it is
    /// always `AcceleratorUnavailable`, whatever the build and host.
    fn rgb_ctx() -> ProcessingContext {
        ProcessingContext {
            width: 8,
            height: 8,
            format: fluxframe_core::frame::PixelFormat::Rgb,
            fps: 30,
            counters: None,
        }
    }

    #[cfg(feature = "ml")]
    #[test]
    fn prepare_falls_back_when_no_accelerator_can_run_the_model() {
        let model = UnrunnableModel::new("switch");
        let cfg = model.config(Some("plain"));
        let prepared = prepare_with_fallback(
            &cfg,
            &ConfigBase::default(),
            "accel",
            &cfg.presets["accel"],
            &rgb_ctx(),
        )
        .expect("the fallback runs");
        assert_eq!(prepared.name, "plain");
        assert_eq!(prepared.preset, cfg.presets["plain"]);
    }

    #[cfg(feature = "ml")]
    #[test]
    fn prepare_without_fallback_reports_the_accelerator_error() {
        let model = UnrunnableModel::new("nofallback");
        let cfg = model.config(None);
        let err = prepare_with_fallback(
            &cfg,
            &ConfigBase::default(),
            "accel",
            &cfg.presets["accel"],
            &rgb_ctx(),
        )
        .err()
        .expect("no fallback to run");
        assert!(err.to_string().contains("NO_SUCH_DEVICE"), "{err}");
    }

    #[test]
    fn prepare_runs_a_preset_that_works_as_is() {
        let cfg = cfg_with_preset("plain", "fallback_preset = \"other\"\n[presets.other]\n");
        let prepared = prepare_with_fallback(
            &cfg,
            &ConfigBase::default(),
            "plain",
            &cfg.presets["plain"],
            &rgb_ctx(),
        )
        .expect("runs");
        assert_eq!(prepared.name, "plain");
    }
}
