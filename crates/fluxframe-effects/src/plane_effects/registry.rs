//! Name-keyed registry for [`PlaneEffect`] factories. Mirror of
//! [`crate::mask_effects::MaskEffectRegistry`] for the
//! background/foreground sub-pipelines.

use std::collections::BTreeMap;

use fluxframe_core::error::EffectError;
use fluxframe_core::plane::PlaneEffect;

/// Factory closure producing a fresh [`PlaneEffect`] instance.
pub trait PlaneEffectFactory: Send + Sync {
    /// Build a brand-new effect.
    fn build(&self) -> Box<dyn PlaneEffect>;
}

impl<F> PlaneEffectFactory for F
where
    F: Fn() -> Box<dyn PlaneEffect> + Send + Sync,
{
    fn build(&self) -> Box<dyn PlaneEffect> {
        self()
    }
}

/// Name-keyed map of plane-effect factories.
#[derive(Default)]
pub struct PlaneEffectRegistry {
    factories: BTreeMap<&'static str, Box<dyn PlaneEffectFactory>>,
}

impl PlaneEffectRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a factory under the canonical name.
    pub fn register(&mut self, name: &'static str, factory: Box<dyn PlaneEffectFactory>) {
        self.factories.insert(name, factory);
    }

    /// Look up a factory.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn PlaneEffectFactory> {
        self.factories.get(name).map(std::convert::AsRef::as_ref)
    }

    /// Sorted list of registered effect names.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.factories.keys().copied().collect()
    }

    /// Number of registered factories.
    #[must_use]
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// Returns `true` if no factory has been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// Returns `true` if a factory is registered under `name`.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// Build a chain from a list of effect names.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::InvalidConfig`] for the first unknown name.
    pub fn build_chain<S: AsRef<str>>(
        &self,
        names: &[S],
    ) -> Result<Vec<Box<dyn PlaneEffect>>, EffectError> {
        names
            .iter()
            .map(|name| {
                let n = name.as_ref();
                self.get(n).map(PlaneEffectFactory::build).ok_or_else(|| {
                    EffectError::InvalidConfig {
                        name: n.to_string(),
                        reason: "unknown plane effect: not registered".into(),
                        hint: Some("known names: see fluxframe_effects::plane_effects".into()),
                    }
                })
            })
            .collect()
    }
}

/// Build a registry pre-populated with the built-in plane effects.
#[must_use]
pub fn default_registry() -> PlaneEffectRegistry {
    use crate::plane_effects::{
        BlurPlaneEffect, ColorFillEffect, PassthroughPlaneEffect, PixelateEffect,
    };
    let mut registry = PlaneEffectRegistry::new();
    registry.register(
        PassthroughPlaneEffect::NAME,
        Box::new(|| -> Box<dyn PlaneEffect> { Box::new(PassthroughPlaneEffect::new()) }),
    );
    registry.register(
        BlurPlaneEffect::NAME,
        Box::new(|| -> Box<dyn PlaneEffect> { Box::new(BlurPlaneEffect::new()) }),
    );
    registry.register(
        ColorFillEffect::NAME,
        Box::new(|| -> Box<dyn PlaneEffect> { Box::new(ColorFillEffect::default()) }),
    );
    registry.register(
        PixelateEffect::NAME,
        Box::new(|| -> Box<dyn PlaneEffect> { Box::new(PixelateEffect::new()) }),
    );
    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_has_all_builtins() {
        let reg = default_registry();
        let names = reg.names();
        assert!(names.contains(&"blur"));
        assert!(names.contains(&"color_fill"));
        assert!(names.contains(&"passthrough"));
        assert!(names.contains(&"pixelate"));
    }

    #[test]
    fn build_chain_fails_on_unknown() {
        let reg = default_registry();
        let res = reg.build_chain(&["nonexistent"]);
        assert!(res.is_err());
    }
}
