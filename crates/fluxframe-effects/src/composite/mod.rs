//! Composite effect: orchestrates a segmentation base + three
//! sub-pipelines (mask, background, foreground) into a single
//! [`fluxframe_core::VideoEffect`].
//!
//! See the per-submodule docs for details; this `mod.rs` re-exports
//! the public surface.

pub mod builder;
pub mod effect;
pub mod segmentation;

pub use builder::CompositeBuilder;
pub use effect::CompositeEffect;
pub use segmentation::{SegmentationBase, SegmentationConfig, SegmentationOutcome};
