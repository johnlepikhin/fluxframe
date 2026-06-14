//! Name-keyed registry for [`MaskEffect`] factories. Twin of
//! [`crate::plane_effects::PlaneEffectRegistry`] on the mask-plane side.
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
use fluxframe_core::plane::MaskEffect;

use crate::registry_common::Registry;

/// Name-keyed map of mask-effect factories.
#[derive(Default)]
pub struct MaskEffectRegistry(pub(crate) Registry<dyn MaskEffect>);

impl MaskEffectRegistry {
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
        factory: Box<dyn Fn() -> Box<dyn MaskEffect> + Send + Sync>,
        metadata: &'static EffectMetadata,
    ) {
        self.0.register(name, factory, metadata);
    }

    /// Build a fresh effect by name.
    #[must_use]
    pub fn build(&self, name: &str) -> Option<Box<dyn MaskEffect>> {
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
    ) -> Result<Vec<Box<dyn MaskEffect>>, EffectError> {
        self.0.build_chain(
            names,
            "mask",
            "known names: see fluxframe_effects::mask_effects",
        )
    }
}

/// Build a registry pre-populated with the built-in mask effects.
#[must_use]
pub fn default_registry() -> MaskEffectRegistry {
    use crate::mask_effects::{
        DilateMaskEffect, FeatherMaskEffect, InvertMaskEffect, LargestBlobMaskEffect,
        PassthroughMaskEffect, SmoothTemporalMaskEffect, ThresholdMaskEffect,
    };
    let mut registry = MaskEffectRegistry::new();
    registry.register(
        PassthroughMaskEffect::NAME,
        Box::new(|| -> Box<dyn MaskEffect> { Box::new(PassthroughMaskEffect::new()) }),
        &PassthroughMaskEffect::METADATA,
    );
    registry.register(
        LargestBlobMaskEffect::NAME,
        Box::new(|| -> Box<dyn MaskEffect> { Box::new(LargestBlobMaskEffect::new()) }),
        &LargestBlobMaskEffect::METADATA,
    );
    registry.register(
        ThresholdMaskEffect::NAME,
        Box::new(|| -> Box<dyn MaskEffect> { Box::new(ThresholdMaskEffect::new()) }),
        &ThresholdMaskEffect::METADATA,
    );
    registry.register(
        DilateMaskEffect::NAME,
        Box::new(|| -> Box<dyn MaskEffect> { Box::new(DilateMaskEffect::new()) }),
        &DilateMaskEffect::METADATA,
    );
    registry.register(
        FeatherMaskEffect::NAME,
        Box::new(|| -> Box<dyn MaskEffect> { Box::new(FeatherMaskEffect::new()) }),
        &FeatherMaskEffect::METADATA,
    );
    registry.register(
        SmoothTemporalMaskEffect::NAME,
        Box::new(|| -> Box<dyn MaskEffect> { Box::new(SmoothTemporalMaskEffect::new()) }),
        &SmoothTemporalMaskEffect::METADATA,
    );
    registry.register(
        InvertMaskEffect::NAME,
        Box::new(|| -> Box<dyn MaskEffect> { Box::new(InvertMaskEffect::new()) }),
        &InvertMaskEffect::METADATA,
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
        assert!(names.contains(&"largest_blob"));
        assert!(names.contains(&"passthrough"));
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
