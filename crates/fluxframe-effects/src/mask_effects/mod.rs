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
mod largest_blob;
mod passthrough;
mod registry;
mod smooth_temporal;
mod threshold;

pub use dilate::DilateMaskEffect;
pub use feather::FeatherMaskEffect;
pub use invert::InvertMaskEffect;
pub use largest_blob::LargestBlobMaskEffect;
pub use passthrough::PassthroughMaskEffect;
pub use registry::{MaskEffectRegistry, default_registry};
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

#[cfg(test)]
mod metadata_tests {
    //! Cross-effect consistency tests. Each effect's
    //! [`EffectMetadata`](fluxframe_core::EffectMetadata) declares the
    //! parameter defaults the GUI shows; those defaults must match
    //! `Default for Config` (the daemon source of truth) or the GUI
    //! and daemon will silently disagree.
    //!
    //! The actual loop is shared with the plane and post sections via
    //! [`crate::registry_common::assert_metadata_defaults_round_trip`];
    //! this test is the per-section entry point that wires the right
    //! `configure()` signature.
    use super::*;

    /// For every effect in the default registry, walk its
    /// `ParamDescriptor` defaults and assert they round-trip through
    /// the effect's `configure()` cleanly. We treat acceptance as
    /// "default matches" — an effect that rejects its own metadata
    /// default fails this test.
    #[test]
    fn metadata_defaults_round_trip_through_configure() {
        let reg = registry::default_registry();
        crate::registry_common::assert_metadata_defaults_round_trip(&reg.0, |effect, params| {
            effect.configure(params)
        });
    }
}
