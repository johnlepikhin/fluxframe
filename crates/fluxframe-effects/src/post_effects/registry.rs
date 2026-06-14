//! Name-keyed registry for [`PostEffect`] factories. Twin of
//! [`crate::mask_effects::MaskEffectRegistry`] and
//! [`crate::plane_effects::PlaneEffectRegistry`], on the post-composite
//! mask-aware side.
//!
//! Each entry pairs a build closure with a `&'static EffectMetadata`
//! reference. Metadata is registry-only (no method on the trait) so a
//! caller introspecting the inventory never has to build an instance.
//!
//! Storage and lookup are delegated to the generic
//! [`crate::registry_common::Registry`]. This newtype wrapper exists
//! to give the public API a concrete, self-documenting type.

use fluxframe_core::EffectMetadata;
use fluxframe_core::error::EffectError;
use fluxframe_core::plane::PostEffect;

use crate::registry_common::Registry;

/// Name-keyed map of post-effect factories.
#[derive(Default)]
pub struct PostEffectRegistry(pub(crate) Registry<dyn PostEffect>);

impl PostEffectRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a factory under the canonical name, paired with its
    /// metadata descriptor.
    pub fn register(
        &mut self,
        name: &'static str,
        factory: Box<dyn Fn() -> Box<dyn PostEffect> + Send + Sync>,
        metadata: &'static EffectMetadata,
    ) {
        self.0.register(name, factory, metadata);
    }

    /// Build a fresh effect by name.
    #[must_use]
    pub fn build(&self, name: &str) -> Option<Box<dyn PostEffect>> {
        self.0.build(name)
    }

    /// Look up the metadata for a registered name.
    #[must_use]
    pub fn metadata(&self, name: &str) -> Option<&'static EffectMetadata> {
        self.0.metadata(name)
    }

    /// Iterate every registered effect's metadata, in name order.
    pub fn iter_metadata(&self) -> impl Iterator<Item = &'static EffectMetadata> + '_ {
        self.0.iter_metadata()
    }

    /// Sorted list of registered effect names.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.0.names()
    }

    /// Number of registered factories.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if no factory has been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns `true` if a factory is registered under `name`.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.0.contains(name)
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
        self.0.build_chain(
            names,
            "post",
            "known names: see fluxframe_effects::post_effects",
        )
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
        &PassthroughPostEffect::METADATA,
    );
    registry.register(
        AutoFrameEffect::NAME,
        Box::new(|| -> Box<dyn PostEffect> { Box::new(AutoFrameEffect::new()) }),
        &AutoFrameEffect::METADATA,
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

    #[test]
    fn metadata_exposed_for_every_registered_name() {
        let reg = default_registry();
        for name in reg.names() {
            let meta = reg.metadata(name).expect("metadata for registered name");
            assert_eq!(meta.name, name, "metadata.name must match registry key");
        }
    }

    #[test]
    fn iter_metadata_yields_one_entry_per_name() {
        let reg = default_registry();
        let count = reg.iter_metadata().count();
        assert_eq!(count, reg.len());
    }
}
