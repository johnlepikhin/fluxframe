//! Built-in [`PostEffect`](fluxframe_core::PostEffect) implementations
//! used by the composite effect's post-composite chain. Post-effects
//! run after the alpha-composite step and receive a read-only view of
//! the frame-resolution mask.

mod auto_frame;
mod passthrough;
mod registry;

pub use auto_frame::AutoFrameEffect;
pub use passthrough::PassthroughPostEffect;
pub use registry::{PostEffectFactory, PostEffectRegistry, default_registry};

use fluxframe_core::EffectError;

fn invalid_config(name: &str, reason: impl Into<String>) -> EffectError {
    EffectError::InvalidConfig {
        name: name.to_string(),
        reason: reason.into(),
        hint: None,
    }
}
