//! Pixel-buffer image processing primitives used by ML-backed effects.
//!
//! All helpers operate on plain byte/float slices — no allocation
//! inside hot paths.  Callers (typically `BackgroundBlurEffect`)
//! supply pre-sized scratch buffers.

pub mod blur;
pub mod compose;
#[cfg(feature = "image-fill")]
pub mod fit;
pub mod mask;
pub mod resize;

pub use blur::box_blur_rgb;
pub use compose::alpha_composite_rgb_in_place;
pub use mask::{dilate, feather, smooth_temporal, threshold};
pub use resize::{resize_mask_bilinear, resize_rgb_bilinear, resize_rgb_nearest};
