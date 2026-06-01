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
pub use registry::{PlaneEffectFactory, PlaneEffectRegistry, default_registry};
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
