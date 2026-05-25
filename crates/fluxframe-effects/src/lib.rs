//! Effect registry, effect chain runtime and built-in effects.
//!
//! The two ML-adjacent modules (`ml`, `processing`) are stubs in Stage 0
//! and gain content in Stage 3 (inference layer) and Stage 4
//! (`background_blur`).  Keeping them as modules in this crate — rather
//! than separate crates — is a deliberate scope decision: the only
//! consumer of either is `background_blur`, so an extra crate boundary
//! would be pure overhead until a second consumer appears.
//!
//! # Feature flags
//!
//! * `ml` (default) — enables [`background_blur`] and the entire `ml`
//!   inference layer, pulling in the `ort` (ONNX Runtime) dependency.
//!   Disable defaults (`default-features = false`) to compile a slim
//!   build with only the dependency-free effects (`passthrough`).

#![warn(missing_docs)]

pub mod backend;
#[cfg(feature = "ml")]
pub mod background_blur;
pub mod chain;
#[cfg(feature = "ml")]
pub mod ml;
pub mod passthrough;
pub mod processing;
pub mod registry;

#[cfg(feature = "ml")]
pub use background_blur::{BackgroundBlurEffect, BlurConfig};
pub use chain::EffectChain;
pub use passthrough::PassthroughEffect;
pub use registry::{EffectFactory, EffectRegistry};
