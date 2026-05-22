//! ML inference layer.
//!
//! Stage 3 ships:
//! * [`ModelConfig`] — TOML schema describing the model's tensor IO
//!   shape, dtype and post-processing parameters (spec §24.2).
//! * `OnnxEngine` — concrete [`fluxframe_core::traits::InferenceEngine`]
//!   implementation backed by ONNX Runtime via the `ort` crate.
//!
//! Stage 4 (`background_blur`) is the first consumer.  The trait is
//! intentionally GStreamer-free; effects depend on
//! [`fluxframe_core::traits::InferenceEngine`] only, not on this module.

pub mod model_config;
pub mod onnx;

pub use model_config::{ModelConfig, OutputLayout, OutputType, TensorDType, TensorLayout};
pub use onnx::OnnxEngine;
