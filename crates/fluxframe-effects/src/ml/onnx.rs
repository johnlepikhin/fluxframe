//! ONNX Runtime-backed [`InferenceEngine`].
//!
//! Uses the `ort` crate (2.0 series) with the `load-dynamic` feature so
//! the ONNX Runtime library is dlopened at process start instead of
//! linked at build time.  The library path is read from the
//! `ORT_DYLIB_PATH` environment variable, see the project README for
//! the Guix store path.
//!
//! This module is the only place that depends on `ort` types; the rest
//! of the crate consumes the [`fluxframe_core::traits::InferenceEngine`]
//! trait so an alternative backend can be plugged in without touching
//! the effect implementations.

use std::path::Path;
use std::sync::Arc;
use std::sync::OnceLock;

use fluxframe_core::error::InferenceError;
use fluxframe_core::traits::{InferenceEngine, InferenceInput, InferenceOutput, ModelInfo};
use ort::session::Session;
use ort::value::Tensor;
use tracing::{debug, info};

use super::model_config::ModelConfig;

/// Ensure the ONNX Runtime library is initialised exactly once per process.
///
/// Subsequent calls return the cached outcome — including the cached
/// failure when the dynamic library could not be loaded — so the second
/// engine load is cheap and deterministic.
fn ensure_runtime_initialised() {
    // `ort::init().commit()` returns `bool` in 2.0-rc.x: `true` when the
    // environment was registered, `false` when something earlier in the
    // process already registered one (also fine for us).  The dynamic
    // load failure surfaces later, from `Session::builder()`, so we
    // remap it to `BackendUnavailable` inside `OnnxEngine::load`.
    // Wrapping the call in `OnceLock` keeps it idempotent.
    static INIT: OnceLock<bool> = OnceLock::new();
    INIT.get_or_init(|| ort::init().commit());
}

/// Decide whether an `ort::Error` from `Session::builder()` looks like a
/// missing dynamic library (`load-dynamic` failure) or an honest
/// model-load failure, and wrap it accordingly.
fn map_backend_or_load_error(message: &str) -> InferenceError {
    let lower = message.to_ascii_lowercase();
    if lower.contains("libonnxruntime")
        || lower.contains("ort_dylib_path")
        || lower.contains("dylib")
        || lower.contains("library")
    {
        InferenceError::BackendUnavailable {
            reason: message.to_string(),
            hint: "set ORT_DYLIB_PATH to the libonnxruntime.so path \
                   (e.g. /gnu/store/.../lib/libonnxruntime.so)"
                .into(),
        }
    } else {
        InferenceError::ModelLoadFailed {
            reason: format!("session builder: {message}"),
        }
    }
}

/// ONNX Runtime-backed inference engine.
///
/// One instance owns one [`ort::session::Session`] plus the
/// [`ModelConfig`] that describes its tensor IO.  The engine is `Send`
/// (via `InferenceEngine`) but, like the underlying `ort::Session`, not
/// `Sync` — the runtime pins it to a single worker thread.
pub struct OnnxEngine {
    config: ModelConfig,
    session: Session,
    info: ModelInfo,
}

impl OnnxEngine {
    /// Load a model from disk and pair it with its [`ModelConfig`].
    ///
    /// Initialises the ONNX Runtime on first call (idempotent across the
    /// process via a [`OnceLock`]), opens the model file as a session
    /// and snapshots the static [`ModelInfo`] so the hot path does not
    /// touch `ort` types again.
    ///
    /// # Errors
    ///
    /// * [`InferenceError::BackendUnavailable`] — `libonnxruntime` could
    ///   not be loaded.  Carries a hint pointing at `ORT_DYLIB_PATH`.
    /// * [`InferenceError::ModelNotFound`] — the path does not exist.
    /// * [`InferenceError::ModelLoadFailed`] — the path exists but the
    ///   session builder rejected it (corrupt model, unsupported opset,
    ///   etc.).
    pub fn load(model_path: &Path, config: ModelConfig) -> Result<Self, InferenceError> {
        ensure_runtime_initialised();
        if !model_path.exists() {
            return Err(InferenceError::ModelNotFound {
                path: model_path.display().to_string(),
            });
        }
        info!(model = %model_path.display(), name = %config.name, "loading ONNX session");
        // `Session::builder()` is where `load-dynamic` actually tries to
        // dlopen libonnxruntime.  Failures there are remapped to
        // `BackendUnavailable` (with the ORT_DYLIB_PATH hint) so the
        // CLI can render the §27 "install/configure ORT" guidance.
        let mut builder =
            Session::builder().map_err(|e| map_backend_or_load_error(&e.to_string()))?;
        let session =
            builder
                .commit_from_file(model_path)
                .map_err(|e| InferenceError::ModelLoadFailed {
                    reason: format!("commit_from_file({}): {e}", model_path.display()),
                })?;

        let info = ModelInfo {
            name: Arc::<str>::from(config.name.as_str()),
            input_width: config.input_width,
            input_height: config.input_height,
            input_format: config.input_color,
        };

        debug!(
            inputs = ?session
                .inputs()
                .iter()
                .map(|i| i.name().to_string())
                .collect::<Vec<_>>(),
            outputs = ?session
                .outputs()
                .iter()
                .map(|o| o.name().to_string())
                .collect::<Vec<_>>(),
            "ONNX session ready"
        );

        Ok(Self {
            config,
            session,
            info,
        })
    }

    /// Number of inputs the underlying model exposes.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.session.inputs().len()
    }

    /// Number of outputs the underlying model exposes.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.session.outputs().len()
    }

    /// Borrow the configuration this engine was loaded with.
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl InferenceEngine for OnnxEngine {
    fn model_info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn infer(&mut self, input: InferenceInput<'_>) -> Result<InferenceOutput, InferenceError> {
        // 1. Validate that the flat data length matches the declared shape.
        let expected_elems: usize = input.shape.iter().copied().product();
        if input.data.len() != expected_elems {
            return Err(InferenceError::InferenceFailed {
                reason: format!(
                    "input length {} mismatches shape {:?} (= {})",
                    input.data.len(),
                    input.shape,
                    expected_elems
                ),
            });
        }

        // 2. Build an ndarray-backed tensor.  `ort::value::Tensor::from_array`
        //    accepts `(shape, Vec<T>)` directly in 2.0-rc.x.
        let tensor =
            Tensor::from_array((input.shape.to_vec(), input.data.to_vec())).map_err(|e| {
                InferenceError::InferenceFailed {
                    reason: format!("tensor build: {e}"),
                }
            })?;

        // 3. Snapshot the input/output names from session metadata
        //    *before* taking the mutable borrow `session.run` needs.
        let input_name = self
            .session
            .inputs()
            .first()
            .ok_or_else(|| InferenceError::InferenceFailed {
                reason: "model has no inputs".into(),
            })?
            .name()
            .to_string();

        let output_count = self.session.outputs().len();
        if self.config.output_index >= output_count {
            return Err(InferenceError::InferenceFailed {
                reason: format!(
                    "output_index {} out of range (model has {} outputs)",
                    self.config.output_index, output_count
                ),
            });
        }
        let output_name = self.session.outputs()[self.config.output_index]
            .name()
            .to_string();

        // 4. Run.  `ort::inputs![name => tensor]` builds the input map.
        let outputs = self
            .session
            .run(ort::inputs![input_name => tensor])
            .map_err(|e| InferenceError::InferenceFailed {
                reason: format!("session.run: {e}"),
            })?;

        // 5. Pick the configured output by name.
        let value =
            outputs
                .get(output_name.as_str())
                .ok_or_else(|| InferenceError::InferenceFailed {
                    reason: format!("output '{output_name}' missing from session.run result"),
                })?;
        let (shape, data) =
            value
                .try_extract_tensor::<f32>()
                .map_err(|e| InferenceError::InferenceFailed {
                    reason: format!("extract f32 tensor: {e}"),
                })?;
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        Ok(InferenceOutput {
            data: data.to_vec(),
            shape: shape_usize,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Loading a model from a path that does not exist must surface a
    /// structured error.  When `libonnxruntime` is not installed in the
    /// test environment the call short-circuits with
    /// `BackendUnavailable` instead — both are acceptable, the point
    /// is that the code does not panic and the error variant is
    /// machine-readable.
    #[test]
    fn rejects_missing_model_file() {
        let config = ModelConfig::from_toml_str(
            r#"
name = "x"
input_width = 1
input_height = 1
"#,
        )
        .expect("config ok");
        let path = Path::new("/nonexistent/model.onnx");
        match OnnxEngine::load(path, config) {
            Ok(_) => panic!("expected error for missing model"),
            Err(
                InferenceError::ModelNotFound { .. } | InferenceError::BackendUnavailable { .. },
            ) => {}
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }
}
