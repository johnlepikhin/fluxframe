//! Build a [`CompositeEffect`] from TOML configuration.
//!
//! Wires together:
//! * [`SegmentationBase`] from the `[mask]` section's `model` and
//!   `fallback_threshold` fields.
//! * `MaskEffect` chain via the [`MaskEffectRegistry`].
//! * `PlaneEffect` chain (background) via the [`PlaneEffectRegistry`].
//! * `PlaneEffect` chain (foreground) via the same registry.
//!
//! Each named effect's parameter sub-table (`[mask.threshold]`,
//! `[background.blur]`, …) is handed to `configure` immediately so a
//! malformed payload aborts startup with a precise diagnostic rather
//! than at first frame.

use std::collections::BTreeMap;

use fluxframe_core::error::EffectError;
use fluxframe_core::paths::ConfigBase;
use fluxframe_core::{EffectMetadata, PipelineSection, resolve_effect_params};

use crate::composite::effect::{CompositeEffect, Slot, SubEffectAdapter};
use crate::composite::segmentation::{SegmentationBase, SegmentationConfig};
use crate::mask_effects::MaskEffectRegistry;
use crate::plane_effects::PlaneEffectRegistry;
use crate::post_effects::PostEffectRegistry;

/// Default consecutive-failure threshold before the segmentation
/// engine flips to its fallback strategy. Mirrors the value exported
/// by `crate::composite::segmentation` once that module publishes it;
/// kept here as a local constant until then to avoid a magic literal.
const DEFAULT_FALLBACK_THRESHOLD: u32 = 3;

/// One-shot builder turning the three TOML sections into a fully
/// configured composite effect.
///
/// Registries are passed in by reference so the same builder can be
/// used by tests with a slimmer registry (e.g. only `threshold` and
/// `blur`).
pub struct CompositeBuilder<'a> {
    mask: &'a MaskEffectRegistry,
    plane: &'a PlaneEffectRegistry,
    post: &'a PostEffectRegistry,
}

impl<'a> CompositeBuilder<'a> {
    /// Construct from the registries that supply the effect
    /// implementations.
    #[must_use]
    pub fn new(
        mask: &'a MaskEffectRegistry,
        plane: &'a PlaneEffectRegistry,
        post: &'a PostEffectRegistry,
    ) -> Self {
        Self { mask, plane, post }
    }

    /// Assemble the composite effect.
    ///
    /// `mask` is required (it carries the segmentation model). When
    /// `background` or `foreground` is `None` the corresponding chain
    /// is empty — for the background that means "use the original
    /// frame as the background" (no per-pixel transform applied
    /// before the alpha composite), and for the foreground it means
    /// the original frame stays as the foreground. `post` is the
    /// mask-aware post-composite chain (`auto_frame`, …); `None`
    /// means an empty chain. Relative paths in the sections (the model,
    /// image parameters) are resolved against `base`.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::InvalidConfig`] when an effect name is
    /// not registered, a per-effect TOML payload is malformed, or the
    /// mask section omits the required `model` field.
    pub fn build(
        &self,
        mask: &PipelineSection,
        background: Option<&PipelineSection>,
        foreground: Option<&PipelineSection>,
        post: Option<&PipelineSection>,
        base: &ConfigBase,
    ) -> Result<CompositeEffect, EffectError> {
        let segmentation = build_segmentation(mask, base)?;
        let mask_chain = build_mask_chain(self.mask, mask, base)?;
        let bg_chain = match background {
            Some(section) => build_plane_chain(self.plane, section, "background", base)?,
            None => Vec::new(),
        };
        let fg_chain = match foreground {
            Some(section) => build_plane_chain(self.plane, section, "foreground", base)?,
            None => Vec::new(),
        };
        let post_chain = match post {
            Some(section) => build_post_chain(self.post, section, base)?,
            None => Vec::new(),
        };
        let composite =
            CompositeEffect::new(segmentation, mask_chain, bg_chain, fg_chain, post_chain);
        // A disabled effect is otherwise invisible in the logs; name it
        // once per build (startup, preset switch, reload).
        for section in fluxframe_core::SubchainKind::ALL {
            let disabled = composite.disabled_effects(section);
            if !disabled.is_empty() {
                tracing::info!(section = %section, ?disabled, "composite: effects disabled by preset");
            }
        }
        Ok(composite)
    }
}

fn build_segmentation(
    section: &PipelineSection,
    base: &ConfigBase,
) -> Result<SegmentationBase, EffectError> {
    let Some(model) = section.model.as_ref() else {
        return Err(EffectError::InvalidConfig {
            name: "mask".to_string(),
            reason: "`mask.model` is required (path to the segmentation ONNX file)".into(),
            hint: Some("e.g. model = \"./models/selfie_segmentation.onnx\"".into()),
        });
    };
    let cfg = SegmentationConfig {
        model: base.resolve(model).into_owned(),
        fallback_threshold: section
            .fallback_threshold
            .unwrap_or(DEFAULT_FALLBACK_THRESHOLD),
    };
    cfg.validate()?;
    Ok(SegmentationBase::new(cfg))
}

fn build_mask_chain(
    registry: &MaskEffectRegistry,
    section: &PipelineSection,
    base: &ConfigBase,
) -> Result<Vec<Slot<dyn fluxframe_core::plane::MaskEffect>>, EffectError> {
    reject_unknown_table_keys(&section.per_effect, &section.chain, "mask")?;
    let chain = registry.build_chain(&section.chain)?;
    configure_chain(chain, section, |name| registry.metadata(name), base)
}

fn build_plane_chain(
    registry: &PlaneEffectRegistry,
    section: &PipelineSection,
    section_name: &str,
    base: &ConfigBase,
) -> Result<Vec<Slot<dyn fluxframe_core::plane::PlaneEffect>>, EffectError> {
    reject_unknown_table_keys(&section.per_effect, &section.chain, section_name)?;
    let chain = registry.build_chain(&section.chain)?;
    configure_chain(chain, section, |name| registry.metadata(name), base)
}

fn build_post_chain(
    registry: &PostEffectRegistry,
    section: &PipelineSection,
    base: &ConfigBase,
) -> Result<Vec<Slot<dyn fluxframe_core::plane::PostEffect>>, EffectError> {
    reject_unknown_table_keys(&section.per_effect, &section.chain, "post")?;
    let chain = registry.build_chain(&section.chain)?;
    configure_chain(chain, section, |name| registry.metadata(name), base)
}

/// Configure every effect of `chain` (built from `section.chain`, in
/// the same order) from its table in `section.per_effect`, resolved by
/// [`resolve_effect_params`] against `base` — the same resolution the
/// live `set_chain` path uses, so a preset behaves identically at
/// startup and after an edit. Each effect is paired with its `enabled`
/// flag.
fn configure_chain<E: ?Sized + SubEffectAdapter>(
    chain: Vec<Box<E>>,
    section: &PipelineSection,
    metadata: impl Fn(&str) -> Option<&'static EffectMetadata>,
    base: &ConfigBase,
) -> Result<Vec<Slot<E>>, EffectError> {
    chain
        .into_iter()
        .zip(&section.chain)
        .map(|(mut effect, name)| {
            let resolved =
                resolve_effect_params(name, section.per_effect.get(name), metadata(name), base)?;
            effect.configure_mut(resolved.params)?;
            Ok(Slot::new(effect, resolved.enabled))
        })
        .collect()
}

/// Every key in `per_effect` must either be in `chain` or be one of
/// the reserved control fields. Catches operator typos like
/// `[mask.threhold]` (missing s) at startup instead of letting the
/// effect silently run with defaults.
fn reject_unknown_table_keys(
    per_effect: &BTreeMap<String, toml::Value>,
    chain: &[String],
    section_name: &str,
) -> Result<(), EffectError> {
    for key in per_effect.keys() {
        if PipelineSection::is_reserved_key(key) {
            continue;
        }
        if !chain.iter().any(|n| n == key) {
            return Err(EffectError::InvalidConfig {
                name: section_name.to_string(),
                reason: format!(
                    "[{section_name}.{key}] sub-table has no matching entry in [{section_name}].chain"
                ),
                hint: Some("add the effect to the `chain` list or remove the sub-table".into()),
            });
        }
    }
    // Path-fields are not allowed on background/foreground sections —
    // they belong only to the mask section.
    if section_name != "mask"
        && (per_effect.contains_key("model") || per_effect.contains_key("fallback_threshold"))
    {
        return Err(EffectError::InvalidConfig {
            name: section_name.to_string(),
            reason: format!(
                "[{section_name}] cannot define `model` or `fallback_threshold` (mask section only)"
            ),
            hint: None,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask_effects::default_registry as default_mask_registry;
    use crate::plane_effects::default_registry as default_plane_registry;
    use crate::post_effects::default_registry as default_post_registry;
    use fluxframe_core::traits::VideoEffect;

    fn parse(section_toml: &str) -> PipelineSection {
        toml::from_str(section_toml).expect("parses")
    }

    #[test]
    fn build_segmentation_requires_model() {
        let section = parse("");
        let res = build_segmentation(&section, &ConfigBase::default());
        let Err(err) = res else {
            panic!("expected error");
        };
        let msg = format!("{err}");
        assert!(msg.contains("model"), "got: {msg}");
    }

    #[test]
    fn mask_chain_configures_each_effect() {
        let section = parse(
            r#"
model = "/tmp/dummy.onnx"
chain = ["threshold", "feather"]
[threshold]
level = 0.4
[feather]
radius = 3
"#,
        );
        let reg = default_mask_registry();
        let chain = build_mask_chain(&reg, &section, &ConfigBase::default()).expect("ok");
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].effect().name(), "threshold");
        assert_eq!(chain[1].effect().name(), "feather");
        assert!(chain.iter().all(Slot::is_enabled));
    }

    #[test]
    fn enabled_flag_is_read_and_stripped_before_configure() {
        // `blur` denies unknown fields, so a leaked `enabled` would fail
        // configure; `color_fill` has only the flag and gets defaults.
        let section = parse(
            r#"
chain = ["blur", "color_fill"]
[blur]
enabled = false
radius = 8
[color_fill]
enabled = false
"#,
        );
        let reg = default_plane_registry();
        let chain =
            build_plane_chain(&reg, &section, "background", &ConfigBase::default()).expect("ok");
        assert!(chain.iter().all(|slot| !slot.is_enabled()));
    }

    #[test]
    fn non_bool_enabled_flag_is_rejected() {
        let section = parse(
            r#"
chain = ["blur"]
[blur]
enabled = "off"
"#,
        );
        let reg = default_plane_registry();
        let Err(err) = build_plane_chain(&reg, &section, "background", &ConfigBase::default())
        else {
            panic!("expected error");
        };
        assert!(format!("{err}").contains("must be a boolean"), "{err}");
    }

    #[test]
    fn unknown_subtable_rejected() {
        let section = parse(
            r#"
chain = ["threshold"]
[threhold]   # typo
level = 0.5
"#,
        );
        let reg = default_mask_registry();
        let Err(err) = build_mask_chain(&reg, &section, &ConfigBase::default()) else {
            panic!("expected error");
        };
        assert!(format!("{err}").contains("threhold"));
    }

    #[test]
    fn unknown_effect_in_chain_rejected() {
        let section = parse(
            r#"
chain = ["nope"]
"#,
        );
        let reg = default_mask_registry();
        let Err(err) = build_mask_chain(&reg, &section, &ConfigBase::default()) else {
            panic!("expected error");
        };
        assert!(format!("{err}").contains("nope"));
    }

    #[test]
    fn background_chain_with_color_fill_configures() {
        let section = parse(
            r#"
chain = ["color_fill"]
[color_fill]
rgb = [10, 20, 30]
"#,
        );
        let reg = default_plane_registry();
        let chain =
            build_plane_chain(&reg, &section, "background", &ConfigBase::default()).expect("ok");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].effect().name(), "color_fill");
    }

    #[test]
    fn full_builder_assembles_composite() {
        let mask = parse(
            r#"
model = "/tmp/dummy.onnx"
chain = ["threshold"]
[threshold]
level = 0.5
"#,
        );
        let background = parse(
            r#"
chain = ["color_fill"]
[color_fill]
rgb = [0, 120, 215]
"#,
        );
        let mask_reg = default_mask_registry();
        let plane_reg = default_plane_registry();
        let post_reg = default_post_registry();
        let builder = CompositeBuilder::new(&mask_reg, &plane_reg, &post_reg);
        let composite = builder
            .build(&mask, Some(&background), None, None, &ConfigBase::default())
            .expect("ok");
        // `composite.name()` is the stable identifier — confirm the
        // constructor wired through.
        assert_eq!(composite.name(), CompositeEffect::NAME);
    }

    /// A relative model path means "next to the config file", whatever
    /// directory the daemon runs in.
    #[test]
    fn relative_model_path_resolves_against_the_config_base() {
        let section = parse(r#"model = "models/m.onnx""#);
        let base = ConfigBase::for_config(Some(std::path::Path::new("/etc/ff/fluxframe.toml")));
        let Ok(segmentation) = build_segmentation(&section, &base) else {
            panic!("a relative model path must build");
        };
        assert_eq!(
            segmentation.model_path(),
            std::path::Path::new("/etc/ff/models/m.onnx")
        );
    }
}
