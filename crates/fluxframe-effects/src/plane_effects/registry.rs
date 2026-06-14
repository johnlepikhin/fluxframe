//! Name-keyed registry for [`PlaneEffect`] factories. Mirror of
//! [`crate::mask_effects::MaskEffectRegistry`] for the
//! background/foreground sub-pipelines.
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
use fluxframe_core::plane::PlaneEffect;

use crate::registry_common::Registry;

/// Name-keyed map of plane-effect factories.
#[derive(Default)]
pub struct PlaneEffectRegistry(pub(crate) Registry<dyn PlaneEffect>);

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
        factory: Box<dyn Fn() -> Box<dyn PlaneEffect> + Send + Sync>,
        metadata: &'static EffectMetadata,
    ) {
        self.0.register(name, factory, metadata);
    }

    /// Build a fresh effect by name.
    #[must_use]
    pub fn build(&self, name: &str) -> Option<Box<dyn PlaneEffect>> {
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
    ) -> Result<Vec<Box<dyn PlaneEffect>>, EffectError> {
        self.0.build_chain(
            names,
            "plane",
            "known names: see fluxframe_effects::plane_effects",
        )
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
