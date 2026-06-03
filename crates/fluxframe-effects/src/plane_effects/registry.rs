//! Name-keyed registry for [`PlaneEffect`] factories. Mirror of
//! [`crate::mask_effects::MaskEffectRegistry`] for the
//! background/foreground sub-pipelines.
//!
//! Each entry pairs a build closure with a `&'static EffectMetadata`
//! reference. Metadata is registry-only (no method on the trait) so a
//! caller introspecting the inventory never has to build an instance.

use std::collections::BTreeMap;

use fluxframe_core::EffectMetadata;
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

/// One row of the registry: factory + descriptor.
struct Entry {
    factory: Box<dyn PlaneEffectFactory>,
    metadata: &'static EffectMetadata,
}

/// Name-keyed map of plane-effect factories.
#[derive(Default)]
pub struct PlaneEffectRegistry {
    entries: BTreeMap<&'static str, Entry>,
}

impl PlaneEffectRegistry {
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
        factory: Box<dyn PlaneEffectFactory>,
        metadata: &'static EffectMetadata,
    ) {
        self.entries.insert(name, Entry { factory, metadata });
    }

    /// Look up a factory.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn PlaneEffectFactory> {
        self.entries.get(name).map(|e| e.factory.as_ref())
    }

    /// Look up the metadata for a registered name.
    #[must_use]
    pub fn metadata(&self, name: &str) -> Option<&'static EffectMetadata> {
        self.entries.get(name).map(|e| e.metadata)
    }

    /// Iterate every registered effect's metadata, in name order.
    pub fn iter_metadata(&self) -> impl Iterator<Item = &'static EffectMetadata> + '_ {
        self.entries.values().map(|e| e.metadata)
    }

    /// Sorted list of registered effect names.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.entries.keys().copied().collect()
    }

    /// Number of registered factories.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no factory has been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns `true` if a factory is registered under `name`.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
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
    #[cfg(feature = "image-fill")]
    use crate::plane_effects::ImageFillEffect;
    use crate::plane_effects::{
        BlurPlaneEffect, ColorFillEffect, ExposureCorrectEffect, PassthroughPlaneEffect,
        PixelateEffect, SharpenEffect, VignetteEffect,
    };
    let mut registry = PlaneEffectRegistry::new();
    registry.register(
        PassthroughPlaneEffect::NAME,
        Box::new(|| -> Box<dyn PlaneEffect> { Box::new(PassthroughPlaneEffect::new()) }),
        &PassthroughPlaneEffect::METADATA,
    );
    registry.register(
        BlurPlaneEffect::NAME,
        Box::new(|| -> Box<dyn PlaneEffect> { Box::new(BlurPlaneEffect::new()) }),
        &BlurPlaneEffect::METADATA,
    );
    registry.register(
        ColorFillEffect::NAME,
        Box::new(|| -> Box<dyn PlaneEffect> { Box::new(ColorFillEffect::default()) }),
        &ColorFillEffect::METADATA,
    );
    registry.register(
        PixelateEffect::NAME,
        Box::new(|| -> Box<dyn PlaneEffect> { Box::new(PixelateEffect::new()) }),
        &PixelateEffect::METADATA,
    );
    registry.register(
        SharpenEffect::NAME,
        Box::new(|| -> Box<dyn PlaneEffect> { Box::new(SharpenEffect::new()) }),
        &SharpenEffect::METADATA,
    );
    registry.register(
        VignetteEffect::NAME,
        Box::new(|| -> Box<dyn PlaneEffect> { Box::new(VignetteEffect::new()) }),
        &VignetteEffect::METADATA,
    );
    registry.register(
        ExposureCorrectEffect::NAME,
        Box::new(|| -> Box<dyn PlaneEffect> { Box::new(ExposureCorrectEffect::new()) }),
        &ExposureCorrectEffect::METADATA,
    );
    #[cfg(feature = "image-fill")]
    registry.register(
        ImageFillEffect::NAME,
        Box::new(|| -> Box<dyn PlaneEffect> { Box::new(ImageFillEffect::new()) }),
        &ImageFillEffect::METADATA,
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
        assert!(names.contains(&"sharpen"));
        assert!(names.contains(&"vignette"));
        assert!(names.contains(&"exposure_correct"));
        #[cfg(feature = "image-fill")]
        assert!(names.contains(&"image_fill"));
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
