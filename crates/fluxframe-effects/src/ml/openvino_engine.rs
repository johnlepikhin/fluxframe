//! OpenVINO-backed [`InferenceEngine`] (Stage 8).
//!
//! Loads an ONNX model directly via OpenVINO's ONNX frontend
//! (`Core::read_model_from_file` accepts `.onnx` paths with the weights
//! parameter empty), compiles for the requested device — `"CPU"`,
//! `"GPU"`, `"NPU"`, or any OpenVINO device string including
//! `"AUTO:NPU,GPU,CPU"` — and creates a reusable `InferRequest` for
//! the per-frame hot path.
//!
//! The module file is deliberately named `openvino_engine.rs`, not
//! `openvino.rs`, to avoid shadowing the upstream `openvino` crate
//! name inside `use openvino::…` paths.
//!
//! Runtime setup: this backend needs a working OpenVINO runtime
//! on the host.  In the recommended Guix Home setup the
//! `openvino-full` package wires `OPENVINO_INSTALL_DIR` so the
//! `openvino-finder` crate locates `libopenvino_c.so` without any
//! `LD_LIBRARY_PATH` gymnastics.  See `README.md` "OpenVINO
//! inference (experimental)" and
//! `doc/plan/stage-8-openvino-inference.md` for the operator
//! command-lines.

use std::path::Path;
use std::sync::Arc;

use fluxframe_core::error::InferenceError;
use fluxframe_core::traits::{InferenceEngine, InferenceInput, InferenceOutput, ModelInfo};
use openvino::{
    CompiledModel, Core, DeviceType, ElementType, InferRequest, PropertyKey, SetupError, Shape,
    Tensor,
};
use tracing::{debug, info};

use crate::ml::ModelConfig;

/// Hint surfaced when `libopenvino_c.so` cannot be dlopened.
///
/// On Guix Home the `openvino-full` package wires
/// `OPENVINO_INSTALL_DIR` via `/etc/profile`, so the canonical
/// failure is "operator forgot to `guix home reconfigure`" — the
/// hint covers that and the legacy pip-install fallback.
const LIBRARY_HINT: &str = "ensure the OpenVINO runtime is on the loader path. \
     With Guix Home: run `guix home reconfigure` so `OPENVINO_INSTALL_DIR` \
     and `libopenvino_c.so` are exported by the `openvino-full` package. \
     Without Guix: set `LD_LIBRARY_PATH` to a directory containing \
     `libopenvino_c.so` (e.g. a pip-installed openvino's `libs/` dir, \
     after creating unversioned symlinks)";

/// OpenVINO-backed inference engine.
///
/// One instance owns one compiled model + infer request pair.  The
/// type is `Send` (matching the [`InferenceEngine`] trait bound) but
/// not `Sync` — wrap in `Arc<Mutex<…>>` for sharing across threads.
pub struct OpenVinoEngine {
    /// Cached `ModelInfo` snapshot — built from the supplied
    /// [`ModelConfig`] up front so the hot path does not touch any
    /// OpenVINO API for static metadata.
    info: ModelInfo,
    /// OpenVINO core handle.  Held for the lifetime of the engine
    /// so the dynamically-loaded library stays mapped.  We don't
    /// touch the core after `load` in the production hot path; the
    /// reference is kept so future stages can call
    /// `core.set_property` for performance tuning without rebuilding
    /// the engine.
    #[allow(
        dead_code,
        reason = "Kept alive so libopenvino.so stays mapped for the InferRequest's lifetime; \
                  future stages may call core.set_property for runtime tuning."
    )]
    core: Core,
    /// Compiled model — owns the device-side pipeline that the
    /// `InferRequest` indexes into.  MUST outlive `request`; rustc
    /// enforces this via the struct field drop order (declaration
    /// order = drop order, so `request` drops before `compiled`).
    #[allow(
        dead_code,
        reason = "Owned to keep the device pipeline alive for the lifetime of the InferRequest \
                  field; no methods need to be called on it after construction."
    )]
    compiled: CompiledModel,
    /// Per-engine reusable infer request.  The input tensor
    /// ([`Self::input_tensor`]) is bound to it once and reused; each
    /// frame overwrites the tensor's data buffer in place.
    request: InferRequest,
    /// Reused input tensor, (re)allocated and (re)bound to `request`
    /// only when the input shape changes (so a dynamic model still
    /// works; the static selfie model builds it exactly once). This
    /// replaces the per-frame `Tensor::new` that drove ~768 KB/frame of
    /// heap churn under the AUTO/CPU plugin (multi-GB RSS over a run).
    /// `Some` is the invariant "allocated AND successfully bound".
    input_tensor: Option<CachedInputTensor>,
    /// Tensor name expected by the loaded model on its single input
    /// port (e.g. `"input_1:0"` for SelfieSegmentation).  Captured
    /// once at load so the hot path does not call OpenVINO metadata
    /// APIs.
    input_name: String,
    /// Same as [`Self::input_name`] but for the model's chosen
    /// output port (`config.output_index`).
    output_name: String,
    /// Target device string for diagnostics — `"CPU"`, `"GPU"`,
    /// `"NPU"`, `"AUTO:NPU,GPU,CPU"`, etc.
    #[allow(
        dead_code,
        reason = "Reserved for future telemetry — Stage 9/10 surface the active device in the \
                  periodic metrics reporter."
    )]
    device: String,
    /// User-supplied model config.  Held so `infer()` can validate
    /// shape against expectations and so diagnostics can include
    /// the operator-visible model name.
    #[allow(
        dead_code,
        reason = "Held for diagnostic + future validation paths; the hot path consumes only \
                  `input_name` / `output_name`."
    )]
    config: ModelConfig,
}

/// The reused input tensor plus the shape it was allocated for.
///
/// `shape` is the OpenVINO `i64` form (matching the `shape_i64` built in
/// [`OpenVinoEngine::infer`]) so the per-frame "do we need to
/// reallocate?" check is a plain slice compare with no conversion.
struct CachedInputTensor {
    shape: Vec<i64>,
    tensor: Tensor,
}

/// Whether the cached input tensor must be (re)allocated for `shape`.
///
/// Pure decision extracted from the hot path so it is unit-testable
/// without an OpenVINO runtime. `None` (no tensor yet) or a shape
/// mismatch both force a reallocation; an exact match reuses.
fn needs_realloc(cached_shape: Option<&[i64]>, shape: &[i64]) -> bool {
    cached_shape != Some(shape)
}

impl OpenVinoEngine {
    /// Load a model from disk and prepare an inference request.
    ///
    /// `device` is the OpenVINO device string — passed verbatim to
    /// `Core::compile_model` (`"CPU"` for Stage 8, anything in
    /// `"GPU"`, `"NPU"`, `"AUTO:NPU,GPU,CPU"` for later stages).
    ///
    /// # Errors
    ///
    /// * [`InferenceError::BackendUnavailable`] — `libopenvino_c.so`
    ///   cannot be loaded.  Carries [`LIBRARY_HINT`] for the §27-style
    ///   CLI diagnostics.
    /// * [`InferenceError::ModelNotFound`] — `model_path` does not
    ///   exist.
    /// * [`InferenceError::ModelLoadFailed`] — `read_model_from_file`
    ///   accepts the file but `compile_model` / metadata extraction
    ///   subsequently fails (unsupported op, device not supported,
    ///   shape mismatch).
    pub fn load(
        model_path: &Path,
        config: ModelConfig,
        device: &str,
    ) -> Result<Self, InferenceError> {
        // Fail fast with a clear ModelNotFound rather than letting
        // OpenVINO surface a generic "cannot read model" with the
        // path embedded in a C-API error string.
        if !model_path.exists() {
            return Err(InferenceError::ModelNotFound {
                path: model_path.display().to_string(),
            });
        }
        let model_path_str =
            model_path
                .to_str()
                .ok_or_else(|| InferenceError::InvalidModelConfig {
                    reason: format!("model path is not valid UTF-8: {}", model_path.display()),
                })?;
        let mut core = Core::new().map_err(|e| map_setup_error(&e))?;
        // Second argument is the weights file path; for `.onnx`
        // files OpenVINO ignores it (everything is in the model
        // file itself), so passing `""` is canonical.
        let model = core.read_model_from_file(model_path_str, "").map_err(|e| {
            InferenceError::ModelLoadFailed {
                reason: format!("read_model_from_file({}): {e}", model_path.display()),
            }
        })?;
        let input_name = model
            .get_input_by_index(0)
            .and_then(|node| node.get_name())
            .map_err(|e| InferenceError::ModelLoadFailed {
                reason: format!("cannot read input port name: {e}"),
            })?;
        let output_index = config.output_index;
        let output_name = model
            .get_output_by_index(output_index)
            .and_then(|node| node.get_name())
            .map_err(|e| InferenceError::InvalidModelConfig {
                reason: format!("cannot read output port name at index {output_index}: {e}"),
            })?;
        let device_type: DeviceType = device.into();
        let mut compiled = core.compile_model(&model, device_type).map_err(|e| {
            InferenceError::ModelLoadFailed {
                reason: format!("compile_model(device={device}): {e}"),
            }
        })?;
        let request =
            compiled
                .create_infer_request()
                .map_err(|e| InferenceError::ModelLoadFailed {
                    reason: format!("create_infer_request: {e}"),
                })?;
        let info = ModelInfo {
            name: Arc::<str>::from(config.name.as_str()),
            input_width: config.input_width,
            input_height: config.input_height,
            input_format: config.input_color,
        };
        // Diagnose what AUTO actually picked.  When `device` is
        // a concrete leaf (`"CPU"` / `"GPU"` / `"NPU"`) the
        // execution-device property usually echoes it back; for
        // `"AUTO:NPU,GPU,CPU"` it surfaces the device OpenVINO
        // ranked first that successfully loaded the model.  We
        // also enumerate the available-devices set so a missing
        // accelerator (no NPU driver, no Intel Compute Runtime,
        // …) is visible in the same log line.
        let available_devices = core.available_devices().map_or_else(
            |e| format!("<query failed: {e}>"),
            |devs| {
                devs.iter()
                    .map(|d| d.as_ref().to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            },
        );
        let execution_devices = compiled
            .get_property(&PropertyKey::Other("EXECUTION_DEVICES".into()))
            .map_or_else(
                |e| format!("<query failed: {e}>"),
                std::borrow::Cow::into_owned,
            );
        info!(
            backend = "openvino",
            device = device,
            execution_devices = %execution_devices,
            available_devices = %available_devices,
            model_name = %config.name,
            model_path = %model_path.display(),
            input_port = %input_name,
            output_port = %output_name,
            "openvino backend ready",
        );
        Ok(Self {
            info,
            core,
            compiled,
            request,
            input_tensor: None,
            input_name,
            output_name,
            device: device.to_string(),
            config,
        })
    }
}

/// Map an `openvino::SetupError` (returned by [`Core::new`]) into
/// the FluxFrame error vocabulary.  Both branches surface as
/// [`InferenceError::BackendUnavailable`] — a malformed C-API
/// response from `ov_core_create` is operationally identical to a
/// missing shared library from the operator's perspective.
fn map_setup_error(err: &SetupError) -> InferenceError {
    InferenceError::BackendUnavailable {
        reason: format!("openvino Core::new failed: {err}"),
        hint: LIBRARY_HINT.into(),
    }
}

impl InferenceEngine for OpenVinoEngine {
    fn model_info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn infer(&mut self, input: InferenceInput<'_>) -> Result<InferenceOutput, InferenceError> {
        let expected: usize = input.shape.iter().product();
        if input.data.len() != expected {
            return Err(InferenceError::InferenceFailed {
                reason: format!(
                    "input length {} mismatches shape {:?} (= {expected})",
                    input.data.len(),
                    input.shape
                ),
            });
        }
        // OpenVINO's `Shape` constructor wants `&[i64]` (signed
        // because OpenVINO supports dynamic shapes with `-1`).
        // Tensor shapes never exceed `i64::MAX`; `i64::try_from` is
        // the lint-safe cast.
        let shape_i64: Vec<i64> = input
            .shape
            .iter()
            .map(|&d| i64::try_from(d))
            .collect::<Result<_, _>>()
            .map_err(|e| InferenceError::InferenceFailed {
                reason: format!("shape dimension overflow: {e}"),
            })?;
        // (Re)allocate + bind the input tensor only when the shape
        // changes (once, for the static selfie model). The hot path
        // below just overwrites its buffer — no per-frame allocation.
        if needs_realloc(
            self.input_tensor.as_ref().map(|c| c.shape.as_slice()),
            &shape_i64,
        ) {
            let shape = Shape::new(&shape_i64).map_err(|e| InferenceError::InferenceFailed {
                reason: format!("Shape::new({:?}): {e}", input.shape),
            })?;
            let tensor = Tensor::new(ElementType::F32, &shape).map_err(|e| {
                InferenceError::InferenceFailed {
                    reason: format!("Tensor::new(F32, {:?}): {e}", input.shape),
                }
            })?;
            // Bind once. OpenVINO reads the tensor lazily at `infer()`
            // time, so in-place overwrites of its buffer are picked up
            // without re-calling `set_tensor` per frame.
            self.request
                .set_tensor(&self.input_name, &tensor)
                .map_err(|e| InferenceError::InferenceFailed {
                    reason: format!("set_tensor({}): {e}", self.input_name),
                })?;
            // Only now is the cache valid: allocated AND bound. If either
            // step above failed we left `input_tensor` untouched (`None`
            // or the prior tensor) so the next call retries cleanly.
            self.input_tensor = Some(CachedInputTensor {
                shape: shape_i64,
                tensor,
            });
            debug!(
                input_shape = ?input.shape,
                "openvino input tensor (re)allocated and bound",
            );
        }
        // Overwrite the reused tensor's buffer in place with this frame.
        self.input_tensor
            .as_mut()
            .expect("input_tensor is Some after the (re)alloc block")
            .tensor
            .get_data_mut::<f32>()
            .map_err(|e| InferenceError::InferenceFailed {
                reason: format!("tensor.get_data_mut: {e}"),
            })?
            .copy_from_slice(input.data);
        self.request
            .infer()
            .map_err(|e| InferenceError::InferenceFailed {
                reason: format!("infer: {e}"),
            })?;
        let out_tensor = self.request.get_tensor(&self.output_name).map_err(|e| {
            InferenceError::InferenceFailed {
                reason: format!("get_tensor({}): {e}", self.output_name),
            }
        })?;
        let out_shape = out_tensor
            .get_shape()
            .map_err(|e| InferenceError::InferenceFailed {
                reason: format!("output tensor get_shape: {e}"),
            })?;
        let out_shape_usize: Vec<usize> = out_shape
            .get_dimensions()
            .iter()
            .map(|&d| d.max(0) as usize)
            .collect();
        let out_slice =
            out_tensor
                .get_data::<f32>()
                .map_err(|e| InferenceError::InferenceFailed {
                    reason: format!("output tensor get_data: {e}"),
                })?;
        debug!(
            input_shape = ?input.shape,
            output_shape = ?out_shape_usize,
            output_elems = out_slice.len(),
            "openvino infer ok"
        );
        Ok(InferenceOutput {
            data: out_slice.to_vec(),
            shape: out_shape_usize,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Resolve the path to the workspace-shipped SelfieSegmentation
    /// model.  `cargo test` runs with the crate directory as CWD, so
    /// a plain `"models/…"` relative path misses the file — we go up
    /// two levels via `CARGO_MANIFEST_DIR`
    /// (`<workspace>/crates/fluxframe-effects/`).
    fn selfie_segmentation_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("models")
            .join("selfie_segmentation.onnx")
    }

    /// Probe the runtime by loading the production SelfieSegmentation
    /// ONNX (the only model FluxFrame currently ships).  `#[ignore]`
    /// so CI without an OpenVINO runtime does not fail.  Run locally
    /// after `guix home reconfigure`:
    /// ```sh
    /// cargo test -p fluxframe-effects --features openvino \
    ///   --lib ml::openvino_engine:: -- --ignored
    /// ```
    #[test]
    #[ignore = "requires Guix Home with openvino-full + the SelfieSegmentation ONNX file"]
    fn load_selfie_segmentation_on_cpu() {
        let model_path = selfie_segmentation_path();
        if !model_path.exists() {
            eprintln!("skipping: {} not found", model_path.display());
            return;
        }
        let cfg = ModelConfig::new("mediapipe-selfie-segmentation", 256, 256);
        let engine =
            OpenVinoEngine::load(&model_path, cfg, "CPU").expect("openvino CPU load on this host");
        assert_eq!(&*engine.model_info().name, "mediapipe-selfie-segmentation");
    }

    /// End-to-end smoke: load SelfieSegmentation, run one inference
    /// on an all-mid-grey 256×256 input, sanity-check the output
    /// shape matches what the model config promises and that the
    /// mask values are in `[0, 1]`.  No comparison against ORT here
    /// — that lives in `tests/openvino_inference_e2e.rs`.
    #[test]
    #[ignore = "requires Guix Home with openvino-full + the SelfieSegmentation ONNX file"]
    fn infer_one_frame_smoke() {
        let model_path = selfie_segmentation_path();
        if !model_path.exists() {
            eprintln!("skipping: {} not found", model_path.display());
            return;
        }
        let cfg = ModelConfig::new("mediapipe-selfie-segmentation", 256, 256);
        let mut engine = OpenVinoEngine::load(&model_path, cfg, "CPU").expect("openvino load");
        // SelfieSegmentation NHWC: [1, 256, 256, 3].  Mid-grey f32.
        let shape = [1_usize, 256, 256, 3];
        let pixels: usize = shape.iter().product();
        let data = vec![0.5_f32; pixels];
        let out = engine
            .infer(InferenceInput {
                data: &data,
                shape: &shape,
            })
            .expect("infer ok");
        // Output is the mask: H×W single-channel-ish.  Just sanity
        // check rank and value range.  Strict shape semantics are
        // exercised by the cross-backend e2e test.
        assert!(!out.data.is_empty(), "empty output");
        assert!(!out.shape.is_empty(), "empty shape");
        let min = out.data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = out.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (-0.01..=1.01).contains(&min) && (-0.01..=1.01).contains(&max),
            "mask values out of [0, 1]: min={min} max={max}"
        );
    }

    /// CI-runnable (no OpenVINO runtime needed): the pure cache-reuse
    /// decision that keeps `infer()` from reallocating the input tensor
    /// per frame. This is the regression-prone part of the leak fix
    /// (the dynamic-shape branch); the FFI reuse path is hardware-gated.
    #[test]
    fn needs_realloc_only_on_first_call_or_shape_change() {
        let shape: [i64; 4] = [1, 256, 256, 3];
        // No tensor yet → must allocate.
        assert!(needs_realloc(None, &shape));
        // Same shape → reuse (no allocation).
        assert!(!needs_realloc(Some(&shape), &shape));
        // Changed shape → reallocate.
        let other: [i64; 4] = [1, 128, 128, 3];
        assert!(needs_realloc(Some(&shape), &other));
        // Differing rank also reallocates.
        assert!(needs_realloc(Some(&shape), &[1, 256, 256]));
    }

    /// FFI reuse smoke: run `infer()` several times on one engine and
    /// confirm each call still produces a valid mask (exercises the
    /// "bind once, overwrite in place" path). Memory non-growth is a
    /// hardware metric verified out-of-band (sample `/proc/<pid>/status`
    /// `VmRSS` under `fluxframe run` with the OpenVINO backend).
    #[test]
    #[ignore = "requires Guix Home with openvino-full + the SelfieSegmentation ONNX file"]
    fn infer_reuses_input_tensor_across_calls() {
        let model_path = selfie_segmentation_path();
        if !model_path.exists() {
            eprintln!("skipping: {} not found", model_path.display());
            return;
        }
        let cfg = ModelConfig::new("mediapipe-selfie-segmentation", 256, 256);
        let mut engine = OpenVinoEngine::load(&model_path, cfg, "CPU").expect("openvino load");
        let shape = [1_usize, 256, 256, 3];
        let pixels: usize = shape.iter().product();
        for i in 0..5 {
            let data = vec![(i as f32) * 0.1; pixels];
            let out = engine
                .infer(InferenceInput {
                    data: &data,
                    shape: &shape,
                })
                .expect("infer ok on reuse");
            assert!(!out.data.is_empty(), "empty output on call {i}");
        }
        // After the first call the tensor must be cached and reused.
        assert!(
            engine.input_tensor.is_some(),
            "input tensor should be cached after inference"
        );
    }
}
