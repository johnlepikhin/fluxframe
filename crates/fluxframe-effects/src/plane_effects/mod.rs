//! Built-in [`PlaneEffect`](fluxframe_core::PlaneEffect) implementations
//! used by the composite effect's background and foreground chains.

mod blur;
mod color_fill;
mod passthrough;
mod pixelate;
mod registry;

pub use blur::BlurPlaneEffect;
pub use color_fill::ColorFillEffect;
pub use passthrough::PassthroughPlaneEffect;
pub use pixelate::PixelateEffect;
pub use registry::{PlaneEffectFactory, PlaneEffectRegistry, default_registry};

use fluxframe_core::EffectError;

fn invalid_config(name: &str, reason: impl Into<String>) -> EffectError {
    EffectError::InvalidConfig {
        name: name.to_string(),
        reason: reason.into(),
        hint: None,
    }
}
