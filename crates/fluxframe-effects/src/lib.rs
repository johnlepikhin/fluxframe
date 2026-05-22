//! Effect registry, effect chain runtime and built-in effects.
//!
//! The two ML-adjacent modules (`ml`, `processing`) are stubs in Stage 0
//! and gain content in Stage 3 (inference layer) and Stage 4
//! (`background_blur`).  Keeping them as modules in this crate — rather
//! than separate crates — is a deliberate scope decision: the only
//! consumer of either is `background_blur`, so an extra crate boundary
//! would be pure overhead until a second consumer appears.

#![warn(missing_docs)]

pub mod chain;
pub mod ml;
pub mod passthrough;
pub mod processing;
pub mod registry;

pub use chain::EffectChain;
pub use passthrough::PassthroughEffect;
pub use registry::{EffectFactory, EffectRegistry};
