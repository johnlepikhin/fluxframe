//! Built-in [`PlaneEffect`](fluxframe_core::PlaneEffect) implementations
//! used by the composite effect's background and foreground chains.

mod blur;
mod color_fill;
mod exposure_correct;
pub(crate) mod helpers;
#[cfg(feature = "image-fill")]
mod image_fill;
mod passthrough;
mod pixelate;
mod registry;
mod sharpen;
mod vignette;

pub use blur::BlurPlaneEffect;
pub use color_fill::ColorFillEffect;
pub use exposure_correct::ExposureCorrectEffect;
#[cfg(feature = "image-fill")]
pub use image_fill::ImageFillEffect;
pub use passthrough::PassthroughPlaneEffect;
pub use pixelate::PixelateEffect;
pub use registry::{PlaneEffectRegistry, default_registry};
pub use sharpen::SharpenEffect;
pub use vignette::VignetteEffect;

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
    //! `configure()`. Effects whose metadata declares a
    //! `Path { required: true, default: None }` param are skipped —
    //! no defaulting strategy can supply a valid file path.
    //!
    //! Shared with mask and post via
    //! [`crate::registry_common::assert_metadata_defaults_round_trip`].
    use super::*;

    #[test]
    fn metadata_defaults_round_trip_through_configure() {
        let reg = registry::default_registry();
        crate::registry_common::assert_metadata_defaults_round_trip(&reg.0, |effect, params| {
            effect.configure(params)
        });
    }
}
