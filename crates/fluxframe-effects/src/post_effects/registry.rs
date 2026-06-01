//! Name-keyed registry for [`PostEffect`] factories. Twin of
//! [`crate::mask_effects::MaskEffectRegistry`] and
//! [`crate::plane_effects::PlaneEffectRegistry`], on the post-composite
//! mask-aware side.

use std::collections::BTreeMap;

use fluxframe_core::error::EffectError;
use fluxframe_core::plane::PostEffect;

/// Factory closure producing a fresh [`PostEffect`] instance.
pub trait PostEffectFactory: Send + Sync {
    /// Build a brand-new effect.
    fn build(&self) -> Box<dyn PostEffect>;
}

impl<F> PostEffectFactory for F
where
    F: Fn() -> Box<dyn PostEffect> + Send + Sync,
{
    fn build(&self) -> Box<dyn PostEffect> {
        self()
    }
}

/// Name-keyed map of post-effect factories.
#[derive(Default)]
pub struct PostEffectRegistry {
    factories: BTreeMap<&'static str, Box<dyn PostEffectFactory>>,
}

impl PostEffectRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a factory under the canonical name.
    pub fn register(&mut self, name: &'static str, factory: Box<dyn PostEffectFactory>) {
        self.factories.insert(name, factory);
    }

    /// Look up a factory.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn PostEffectFactory> {
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
    ) -> Result<Vec<Box<dyn PostEffect>>, EffectError> {
        names
            .iter()
            .map(|name| {
                let n = name.as_ref();
                self.get(n).map(PostEffectFactory::build).ok_or_else(|| {
                    EffectError::InvalidConfig {
                        name: n.to_string(),
                        reason: "unknown post effect: not registered".into(),
                        hint: Some("known names: see fluxframe_effects::post_effects".into()),
                    }
                })
            })
            .collect()
    }
}

/// Build a registry pre-populated with the built-in post effects.
#[must_use]
pub fn default_registry() -> PostEffectRegistry {
    use crate::post_effects::{AutoFrameEffect, PassthroughPostEffect};
    let mut registry = PostEffectRegistry::new();
    registry.register(
        PassthroughPostEffect::NAME,
        Box::new(|| -> Box<dyn PostEffect> { Box::new(PassthroughPostEffect::new()) }),
    );
    registry.register(
        AutoFrameEffect::NAME,
        Box::new(|| -> Box<dyn PostEffect> { Box::new(AutoFrameEffect::new()) }),
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
        assert!(names.contains(&"passthrough"));
        assert!(names.contains(&"auto_frame"));
    }

    #[test]
    fn build_chain_fails_on_unknown() {
        let reg = default_registry();
        let res = reg.build_chain(&["nonexistent"]);
        assert!(res.is_err());
    }
}
