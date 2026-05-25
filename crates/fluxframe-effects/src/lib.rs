//! Effect registry, effect chain runtime and built-in effects.
//!
//! Stage 9 reshaped the ML pipeline into a composite of three
//! sub-chains (mask, background, foreground) instead of a monolithic
//! `background_blur` effect. The `ml` and `processing` modules host
//! the inference layer and the low-level pixel/mask primitives the
//! composite stages build on.
//!
//! # Feature flags
//!
//! * `ml` (default) — enables the composite pipeline
//!   ([`composite::CompositeEffect`]) plus the inference layer
//!   ([`ml`]). Disable defaults (`default-features = false`) to
//!   compile a slim build with only the dependency-free effects
//!   ([`passthrough`]).
//! * `wgpu` (default) — enables the GPU blur backend behind
//!   [`backend::BlurBackend`].
//! * `openvino` (default) — enables the OpenVINO inference backend.

#![warn(missing_docs)]

pub mod backend;
pub mod chain;
#[cfg(feature = "ml")]
pub mod composite;
pub mod mask_effects;
#[cfg(feature = "ml")]
pub mod ml;
pub mod passthrough;
pub mod plane_effects;
pub mod processing;
pub mod registry;

pub use chain::EffectChain;
#[cfg(feature = "ml")]
pub use composite::CompositeEffect;
pub use passthrough::PassthroughEffect;
pub use registry::{EffectFactory, EffectRegistry};
