//! Built-in [`PostEffect`](fluxframe_core::PostEffect) implementations
//! used by the composite effect's post-composite chain. Post-effects
//! run after the alpha-composite step and receive a read-only view of
//! the frame-resolution mask.

mod auto_frame;
mod passthrough;
mod registry;

pub use auto_frame::AutoFrameEffect;
pub use passthrough::PassthroughPostEffect;
pub use registry::{PostEffectFactory, PostEffectRegistry, default_registry};

use fluxframe_core::EffectError;

fn invalid_config(name: &str, reason: impl Into<String>) -> EffectError {
    EffectError::InvalidConfig {
        name: name.to_string(),
        reason: reason.into(),
        hint: None,
    }
}

#[cfg(test)]
mod metadata_tests {
    //! Defaults declared in `EffectMetadata` must round-trip through
    //! `configure()`.
    use super::*;
    use fluxframe_core::metadata::ParamKind;

    #[test]
    fn metadata_defaults_round_trip_through_configure() {
        let reg = registry::default_registry();
        'effects: for name in reg.names() {
            let meta = reg.metadata(name).expect("metadata");
            let mut effect = reg.get(name).expect("factory").build();
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
