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
use ort::value::{DynValue, Tensor};
use tracing::{debug, info};

use super::model_config::ModelConfig;

/// Ensure the ONNX Runtime library is initialised exactly once per process.
///
/// Marks `ort::init` as attempted; the actual dlopen happens lazily in
/// `Session::builder`, so each subsequent `load` reports a fresh dylib
/// error (no cached failure).
fn ensure_runtime_initialised() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        // `ort::init().commit()` returns `bool`: `true` if this call
        // installed the env config, `false` if another caller did so
        // first.  Either outcome means "ORT is configured"; we surface
        // the bit at trace level to aid debugging double-init races.
        let committed = ort::init().commit();
        tracing::trace!(committed, "ort runtime initialisation attempted");
    });
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
/// [`ModelConfig`] that describes its tensor IO.
///
/// Send-only by API contract through `InferenceEngine: Send`.  The
/// underlying `ort::Session` is `Sync` upstream, but our
/// `infer(&mut self)` enforces exclusive access — wrap in
/// `Arc<Mutex<dyn InferenceEngine>>` for sharing.
pub struct OnnxEngine {
    config: ModelConfig,
    session: Session,
    info: ModelInfo,
    // Cached at load to avoid per-frame metadata reads.
    input_name: String,
    output_name: String,
    output_index: usize,
    input_count: usize,
    output_count: usize,
}

impl OnnxEngine {
    /// Load a model from disk and pair it with its [`ModelConfig`].
    ///
    /// Initialises the ONNX Runtime on first call (idempotent across the
    /// process via a [`OnceLock`]), opens the model file as a session
    /// and snapshots the static [`ModelInfo`] so the hot path does not
    /// touch `ort` types again.  The model path is canonicalised so log
    /// messages and error reports show a stable absolute path.
    ///
    /// # Errors
    ///
    /// * [`InferenceError::BackendUnavailable`] — `libonnxruntime` could
    ///   not be loaded.  Carries a hint pointing at `ORT_DYLIB_PATH`.
    /// * [`InferenceError::ModelNotFound`] — the path does not exist.
    /// * [`InferenceError::ModelLoadFailed`] — the path exists but the
    ///   session builder rejected it (corrupt model, unsupported opset,
    ///   etc.), or the model exposes no inputs.
    /// * [`InferenceError::InvalidModelConfig`] — `config.output_index`
    ///   refers to an output the model does not expose.
    pub fn load(model_path: &Path, config: ModelConfig) -> Result<Self, InferenceError> {
        ensure_runtime_initialised();
        let canonical = std::fs::canonicalize(model_path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => InferenceError::ModelNotFound {
                path: model_path.display().to_string(),
            },
            _ => InferenceError::ModelLoadFailed {
                reason: format!("cannot canonicalize {}: {e}", model_path.display()),
            },
        })?;
        info!(model = %canonical.display(), name = %config.name, "loading ONNX session");
        // `Session::builder()` is where `load-dynamic` actually tries to
        // dlopen libonnxruntime.  Failures there are remapped to
        // `BackendUnavailable` (with the ORT_DYLIB_PATH hint) so the
        // CLI can render the §27 "install/configure ORT" guidance.
        let session = Session::builder()
            .map_err(|e| map_backend_or_load_error(&e.to_string()))?
            .commit_from_file(&canonical)
            .map_err(|e| InferenceError::ModelLoadFailed {
                reason: format!("commit_from_file({}): {e}", canonical.display()),
            })?;

        // Resolve and cache input/output names + counts up-front;
        // per-frame metadata reads were a measurable overhead in the
        // Stage 3 benchmark.  We also validate `output_index` here so
        // misconfigs surface at load instead of on the first `infer`
        // call.
        let inputs = session.inputs();
        let input_count = inputs.len();
        let input_name = inputs
            .first()
            .ok_or_else(|| InferenceError::ModelLoadFailed {
                reason: "model has no inputs".into(),
            })?
            .name()
            .to_string();

        let outputs = session.outputs();
        let output_count = outputs.len();
        let output_index = config.output_index;
        let output_name = outputs
            .get(output_index)
            .ok_or_else(|| InferenceError::InvalidModelConfig {
                reason: format!(
                    "output_index {output_index} out of range; model has {output_count} outputs",
                ),
            })?
            .name()
            .to_string();

        let info = ModelInfo {
            name: Arc::<str>::from(config.name.as_str()),
            input_width: config.input_width,
            input_height: config.input_height,
            input_format: config.input_color,
        };

        if tracing::enabled!(tracing::Level::DEBUG) {
            // Closures over `i.name()` keep us robust against minor `ort`
            // bumps that might reshuffle the `Outlet` path — we just
            // need *something* that exposes `name()`.
            #[allow(
                clippy::redundant_closure_for_method_calls,
                reason = "closure form survives ort minor version bumps that may move the Outlet path"
            )]
            let input_names: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
            #[allow(
                clippy::redundant_closure_for_method_calls,
                reason = "closure form survives ort minor version bumps that may move the Outlet path"
            )]
            let output_names: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
            debug!(
                model = %canonical.display(),
                inputs = ?input_names,
                outputs = ?output_names,
                input_count,
                output_count,
                "ONNX session ready"
            );
        }

        Ok(Self {
            config,
            session,
            info,
            input_name,
            output_name,
            output_index,
            input_count,
            output_count,
        })
    }

    /// Number of inputs the underlying model exposes.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// Number of outputs the underlying model exposes.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.output_count
    }

    /// Borrow the configuration this engine was loaded with.
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Validate the caller's shape against `data` and build the input
    /// tensor.  Pulled out of [`Self::infer`] for readability.
    ///
    /// This is intentionally an associated function — it does not yet
    /// need any state from `self`.  If a future stage adds shape-vs-config
    /// checks (Stage 4 candidate), re-introduce a `&self` parameter then.
    fn build_tensor(input: &InferenceInput<'_>) -> Result<Tensor<f32>, InferenceError> {
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
        Tensor::from_array((input.shape.to_vec(), input.data.to_vec())).map_err(|e| {
            InferenceError::InferenceFailed {
                reason: format!("tensor build: {e}"),
            }
        })
    }

    /// Extract a flat `f32` tensor + shape from an `ort` output value.
    fn extract_f32_output(value: &DynValue) -> Result<InferenceOutput, InferenceError> {
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

impl InferenceEngine for OnnxEngine {
    fn model_info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn infer(&mut self, input: InferenceInput<'_>) -> Result<InferenceOutput, InferenceError> {
        let tensor = Self::build_tensor(&input)?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| InferenceError::InferenceFailed {
                reason: format!("session.run: {e}"),
            })?;
        // `SessionOutputs` implements `Index<usize>` — selecting by
        // `output_index` skips the linear scan in `get(&str)` and keeps
        // the engine in sync with `config.output_index` even if a future
        // ort release reshuffles output ordering.  We bound-check
        // manually because `Index<usize>` panics on overflow.
        if self.output_index >= outputs.len() {
            return Err(InferenceError::InferenceFailed {
                reason: format!(
                    "output_index {} out of range at infer time; session returned {} outputs",
                    self.output_index,
                    outputs.len()
                ),
            });
        }
        let value: &DynValue = &outputs[self.output_index];
        let _ = &self.output_name; // retained for diagnostics / future name-based lookups.
        Self::extract_f32_output(value)
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
