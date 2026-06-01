//! Build a [`CompositeEffect`] from TOML configuration.
//!
//! Wires together:
//! * [`SegmentationBase`] from the `[mask]` section's `model` /
//!   `model_config` / `fallback_threshold` fields.
//! * `MaskEffect` chain via the [`MaskEffectRegistry`].
//! * `PlaneEffect` chain (background) via the [`PlaneEffectRegistry`].
//! * `PlaneEffect` chain (foreground) via the same registry.
//!
//! Each named effect's parameter sub-table (`[mask.threshold]`,
//! `[background.blur]`, …) is handed to `configure` immediately so a
//! malformed payload aborts startup with a precise diagnostic rather
//! than at first frame.

use std::collections::BTreeMap;

use fluxframe_core::PipelineSection;
use fluxframe_core::error::EffectError;

use crate::composite::effect::CompositeEffect;
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
    /// means an empty chain.
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
    ) -> Result<CompositeEffect, EffectError> {
        let segmentation = build_segmentation(mask)?;
        let mask_chain = build_mask_chain(self.mask, mask)?;
        let bg_chain = match background {
            Some(section) => build_plane_chain(self.plane, section, "background")?,
            None => Vec::new(),
        };
        let fg_chain = match foreground {
            Some(section) => build_plane_chain(self.plane, section, "foreground")?,
            None => Vec::new(),
        };
        let post_chain = match post {
            Some(section) => build_post_chain(self.post, section)?,
            None => Vec::new(),
        };
        Ok(CompositeEffect::new(
            segmentation,
            mask_chain,
            bg_chain,
            fg_chain,
            post_chain,
        ))
    }
}

/// Reserved keys in `[mask]` (and `[background]`/`[foreground]`) that
/// are NOT effect names. The composite builder must never look them
/// up in `per_effect`.
const PIPELINE_RESERVED_KEYS: &[&str] = &["chain", "model", "model_config", "fallback_threshold"];

fn build_segmentation(section: &PipelineSection) -> Result<SegmentationBase, EffectError> {
    let Some(model) = section.model.as_ref() else {
        return Err(EffectError::InvalidConfig {
            name: "mask".to_string(),
            reason: "`mask.model` is required (path to the segmentation ONNX file)".into(),
            hint: Some("e.g. model = \"./models/selfie_segmentation.onnx\"".into()),
        });
    };
    let cfg = SegmentationConfig {
        model: model.clone(),
        model_config: section.model_config.clone(),
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
) -> Result<Vec<Box<dyn fluxframe_core::plane::MaskEffect>>, EffectError> {
    reject_unknown_table_keys(&section.per_effect, &section.chain, "mask")?;
    let mut chain = registry.build_chain(&section.chain)?;
    for (effect, name) in chain.iter_mut().zip(section.chain.iter()) {
        if let Some(params) = section.per_effect.get(name).cloned() {
            effect.configure(params)?;
        } else {
            // No sub-table for this effect — call configure with an
            // empty table so the implementation's serde defaults kick in.
            effect.configure(toml::Value::Table(toml::map::Map::new()))?;
        }
    }
    Ok(chain)
}

fn build_plane_chain(
    registry: &PlaneEffectRegistry,
    section: &PipelineSection,
    section_name: &str,
) -> Result<Vec<Box<dyn fluxframe_core::plane::PlaneEffect>>, EffectError> {
    reject_unknown_table_keys(&section.per_effect, &section.chain, section_name)?;
    let mut chain = registry.build_chain(&section.chain)?;
    for (effect, name) in chain.iter_mut().zip(section.chain.iter()) {
        if let Some(params) = section.per_effect.get(name).cloned() {
            effect.configure(params)?;
        } else {
            effect.configure(toml::Value::Table(toml::map::Map::new()))?;
        }
    }
    Ok(chain)
}

fn build_post_chain(
    registry: &PostEffectRegistry,
    section: &PipelineSection,
) -> Result<Vec<Box<dyn fluxframe_core::plane::PostEffect>>, EffectError> {
    reject_unknown_table_keys(&section.per_effect, &section.chain, "post")?;
    let mut chain = registry.build_chain(&section.chain)?;
    for (effect, name) in chain.iter_mut().zip(section.chain.iter()) {
        if let Some(params) = section.per_effect.get(name).cloned() {
            effect.configure(params)?;
        } else {
            effect.configure(toml::Value::Table(toml::map::Map::new()))?;
        }
    }
    Ok(chain)
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
        if PIPELINE_RESERVED_KEYS.contains(&key.as_str()) {
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
        && (per_effect.contains_key("model")
            || per_effect.contains_key("model_config")
            || per_effect.contains_key("fallback_threshold"))
    {
        return Err(EffectError::InvalidConfig {
            name: section_name.to_string(),
            reason: format!(
                "[{section_name}] cannot define `model`, `model_config`, or `fallback_threshold` (mask section only)"
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
        let res = build_segmentation(&section);
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
        let chain = build_mask_chain(&reg, &section).expect("ok");
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].name(), "threshold");
        assert_eq!(chain[1].name(), "feather");
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
        let Err(err) = build_mask_chain(&reg, &section) else {
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
        let Err(err) = build_mask_chain(&reg, &section) else {
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
        let chain = build_plane_chain(&reg, &section, "background").expect("ok");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].name(), "color_fill");
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
            .build(&mask, Some(&background), None, None)
            .expect("ok");
        // `composite.name()` is the stable identifier — confirm the
        // constructor wired through.
        assert_eq!(composite.name(), CompositeEffect::NAME);
    }
}
