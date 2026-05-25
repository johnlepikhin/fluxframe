//! ML inference layer.
//!
//! Stage 3 ships:
//! * [`ModelConfig`] — TOML schema describing the model's tensor IO
//!   shape, dtype and post-processing parameters (spec §24.2).
//! * `OnnxEngine` — concrete [`fluxframe_core::traits::InferenceEngine`]
//!   implementation backed by ONNX Runtime via the `ort` crate.
//! * [`load_sidecar_or_placeholder`] — shared helper for loading the
//!   `<model>.toml` sidecar with a conservative 1x1 fallback.
//!
//! The composite pipeline (see [`crate::composite::CompositeEffect`])
//! is the consumer — its segmentation stage owns an
//! [`fluxframe_core::traits::InferenceEngine`].  The trait is
//! intentionally GStreamer-free; effects depend on
//! [`fluxframe_core::traits::InferenceEngine`] only, not on this module.

pub mod loader;
pub mod model_config;
pub mod onnx;
#[cfg(feature = "openvino")]
pub mod openvino_engine;

pub use loader::load_sidecar_or_placeholder;
pub use model_config::{InputLayout, ModelConfig, OutputLayout, OutputType, TensorDType};
pub use onnx::OnnxEngine;
#[cfg(feature = "openvino")]
pub use openvino_engine::OpenVinoEngine;
