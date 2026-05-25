//! Built-in [`MaskEffect`](fluxframe_core::MaskEffect) implementations.
//!
//! Each effect wraps a primitive in [`crate::processing`] and exposes
//! a TOML schema so the composite-pipeline builder can wire it.
//!
//! Effects operate on the mask plane in place; resizing the mask
//! between model and frame resolution is handled implicitly by the
//! composite effect and is NOT exposed as a mask effect — see the
//! composite-pipeline docs.

mod dilate;
mod feather;
mod invert;
mod passthrough;
mod registry;
mod smooth_temporal;
mod threshold;

pub use dilate::DilateMaskEffect;
pub use feather::FeatherMaskEffect;
pub use invert::InvertMaskEffect;
pub use passthrough::PassthroughMaskEffect;
pub use registry::{MaskEffectFactory, MaskEffectRegistry, default_registry};
pub use smooth_temporal::SmoothTemporalMaskEffect;
pub use threshold::ThresholdMaskEffect;

use fluxframe_core::EffectError;

/// Shared `EffectError::InvalidConfig` builder; avoids retyping the
/// boilerplate in every effect.
fn invalid_config(name: &str, reason: impl Into<String>) -> EffectError {
    EffectError::InvalidConfig {
        name: name.to_string(),
        reason: reason.into(),
        hint: None,
    }
}
