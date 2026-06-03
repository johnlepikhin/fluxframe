//! Built-in [`MaskEffect`](fluxframe_core::MaskEffect) implementations.
//!
//! Each effect wraps a primitive in [`crate::processing`] and exposes
//! a TOML schema so the composite-pipeline builder can wire it.
//!
//! Effects operate on the mask plane in place; resizing the mask
//! between model and frame resolution is handled implicitly by the
//! composite effect and is NOT exposed as a mask effect — see the
//! composite-pipeline docs.

mod dilate;
mod feather;
mod invert;
mod largest_blob;
mod passthrough;
mod registry;
mod smooth_temporal;
mod threshold;

pub use dilate::DilateMaskEffect;
pub use feather::FeatherMaskEffect;
pub use invert::InvertMaskEffect;
pub use largest_blob::LargestBlobMaskEffect;
pub use passthrough::PassthroughMaskEffect;
pub use registry::{MaskEffectFactory, MaskEffectRegistry, default_registry};
pub use smooth_temporal::SmoothTemporalMaskEffect;
pub use threshold::ThresholdMaskEffect;

use fluxframe_core::EffectError;

/// Shared `EffectError::InvalidConfig` builder; avoids retyping the
/// boilerplate in every effect.
fn invalid_config(name: &str, reason: impl Into<String>) -> EffectError {
    EffectError::InvalidConfig {
        name: name.to_string(),
        reason: reason.into(),
        hint: None,
    }
}

#[cfg(test)]
mod metadata_tests {
    //! Cross-effect consistency tests. Each effect's
    //! [`EffectMetadata`](fluxframe_core::EffectMetadata) declares the
    //! parameter defaults the GUI shows; those defaults must match
    //! `Default for Config` (the daemon source of truth) or the GUI
    //! and daemon will silently disagree.
    use super::*;
    use fluxframe_core::metadata::ParamKind;

    /// For every effect in the default registry, walk its
    /// `ParamDescriptor` defaults and assert they round-trip through
    /// the effect's `configure()` cleanly. We treat acceptance as
    /// "default matches" — an effect that rejects its own metadata
    /// default fails this test.
    #[test]
    fn metadata_defaults_round_trip_through_configure() {
        let reg = registry::default_registry();
        'effects: for name in reg.names() {
            let meta = reg.metadata(name).expect("metadata");
            let mut effect = reg.get(name).expect("factory").build();
            // Build a TOML table with each metadata default.
            let mut table = toml::map::Map::new();
            for p in meta.params {
                let value = match p.kind {
                    ParamKind::Float { default, .. } => toml::Value::Float(f64::from(default)),
                    ParamKind::Integer { default, .. } => toml::Value::Integer(default),
                    ParamKind::Bool { default } => toml::Value::Boolean(default),
                    ParamKind::Color { default } => toml::Value::Array(vec![
                        toml::Value::Integer(default[0].into()),
                        toml::Value::Integer(default[1].into()),
                        toml::Value::Integer(default[2].into()),
                    ]),
                    ParamKind::Path {
                        default: Some(p), ..
                    } => toml::Value::String(p.to_string()),
                    ParamKind::Path {
                        default: None,
                        required: false,
                        ..
                    } => continue,
                    ParamKind::Path {
                        default: None,
                        required: true,
                        ..
                    } => {
                        // image_fill-shaped param: cannot be defaulted.
                        // Skip the whole effect — it would also fail
                        // the trip with an empty table.
                        continue 'effects;
                    }
                    ParamKind::Enum { default, .. } => toml::Value::String(default.to_string()),
                };
                table.insert(p.name.to_string(), value);
            }
            let params = toml::Value::Table(table);
            effect.configure(params).unwrap_or_else(|e| {
                panic!("effect '{name}' rejected its own METADATA defaults: {e}")
            });
        }
    }
}
