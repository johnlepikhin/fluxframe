//! Name-keyed registry for [`MaskEffect`] factories.
//!
//! Mirrors the structure of [`crate::registry::EffectRegistry`] but
//! operates on [`MaskEffect`] trait objects. Kept as a separate type
//! rather than generic-over-trait so call sites stay self-documenting
//! and the public API does not leak associated types.

use std::collections::BTreeMap;

use fluxframe_core::error::EffectError;
use fluxframe_core::plane::MaskEffect;

/// Factory closure producing a fresh [`MaskEffect`] instance.
pub trait MaskEffectFactory: Send + Sync {
    /// Build a brand-new effect.
    fn build(&self) -> Box<dyn MaskEffect>;
}

impl<F> MaskEffectFactory for F
where
    F: Fn() -> Box<dyn MaskEffect> + Send + Sync,
{
    fn build(&self) -> Box<dyn MaskEffect> {
        self()
    }
}

/// Name-keyed map of mask-effect factories.
#[derive(Default)]
pub struct MaskEffectRegistry {
    factories: BTreeMap<&'static str, Box<dyn MaskEffectFactory>>,
}

impl MaskEffectRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a factory under the canonical name.
    pub fn register(&mut self, name: &'static str, factory: Box<dyn MaskEffectFactory>) {
        self.factories.insert(name, factory);
    }

    /// Look up a factory.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn MaskEffectFactory> {
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
    ) -> Result<Vec<Box<dyn MaskEffect>>, EffectError> {
        names
            .iter()
            .map(|name| {
                let n = name.as_ref();
                self.get(n).map(MaskEffectFactory::build).ok_or_else(|| {
                    EffectError::InvalidConfig {
                        name: n.to_string(),
                        reason: "unknown mask effect: not registered".into(),
                        hint: Some("known names: see fluxframe_effects::mask_effects".into()),
                    }
                })
            })
            .collect()
    }
}

/// Build a registry pre-populated with the built-in mask effects.
#[must_use]
pub fn default_registry() -> MaskEffectRegistry {
    use crate::mask_effects::{
        DilateMaskEffect, FeatherMaskEffect, InvertMaskEffect, PassthroughMaskEffect,
        SmoothTemporalMaskEffect, ThresholdMaskEffect,
    };
    let mut registry = MaskEffectRegistry::new();
    registry.register(
        PassthroughMaskEffect::NAME,
        Box::new(|| -> Box<dyn MaskEffect> { Box::new(PassthroughMaskEffect::new()) }),
    );
    registry.register(
        ThresholdMaskEffect::NAME,
        Box::new(|| -> Box<dyn MaskEffect> { Box::new(ThresholdMaskEffect::new()) }),
    );
    registry.register(
        DilateMaskEffect::NAME,
        Box::new(|| -> Box<dyn MaskEffect> { Box::new(DilateMaskEffect::new()) }),
    );
    registry.register(
        FeatherMaskEffect::NAME,
        Box::new(|| -> Box<dyn MaskEffect> { Box::new(FeatherMaskEffect::new()) }),
    );
    registry.register(
        SmoothTemporalMaskEffect::NAME,
        Box::new(|| -> Box<dyn MaskEffect> { Box::new(SmoothTemporalMaskEffect::new()) }),
    );
    registry.register(
        InvertMaskEffect::NAME,
        Box::new(|| -> Box<dyn MaskEffect> { Box::new(InvertMaskEffect::new()) }),
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
        assert!(names.contains(&"threshold"));
        assert!(names.contains(&"dilate"));
        assert!(names.contains(&"feather"));
        assert!(names.contains(&"smooth_temporal"));
        assert!(names.contains(&"invert"));
    }

    #[test]
    fn build_chain_fails_on_unknown() {
        let reg = default_registry();
        let res = reg.build_chain(&["nonexistent"]);
        assert!(res.is_err());
    }
}
