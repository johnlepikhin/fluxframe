//! Backend abstraction layer (Stage 6).
//!
//! Provides the seam where future GPU/accelerator implementations of
//! the effect-chain's compute primitives (blur today, inference next)
//! can plug in alongside the existing CPU implementations.  The layer
//! is intentionally narrow: each backend trait covers one primitive,
//! owns its own scratch state, and is constructed by a small factory
//! that performs autodetection in priority order.  GPU implementations
//! are **not** part of this stage — the only resident candidates are
//! CPU.  See `doc/plan/stage-6-backend-abstraction.md` for the
//! extension protocol.
//!
//! Module layout:
//!
//! * [`blur`] — [`BlurBackend`] trait + [`CpuBlurBackend`].
//! * [`blur_factory`] — [`build_blur_backend`] + [`StickyBlurFallback`].
//! * [`inference`] (cfg `ml`) — [`build_inference_engine`] +
//!   [`StickyInferenceFallback`].
//! * [`overrides`] — [`BackendOverrides`] env-var parsing.

pub mod blur;
pub mod blur_factory;
#[cfg(feature = "ml")]
pub mod inference;
pub mod overrides;
#[cfg(feature = "wgpu")]
pub mod wgpu_blur;

pub use blur::{BlurBackend, CpuBlurBackend};
pub use blur_factory::{StickyBlurFallback, build_blur_backend};
#[cfg(feature = "ml")]
pub use inference::{StickyInferenceFallback, build_inference_engine};
pub use overrides::{
    BackendOverrides, BlurBackendChoice, ENV_BLUR, ENV_INFERENCE, InferenceBackendChoice,
};
#[cfg(feature = "wgpu")]
pub use wgpu_blur::WgpuBlurBackend;
