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
    CompiledModel, Core, DeviceType, ElementType, InferRequest, Model, PartialShape, PropertyKey,
    RwPropertyKey, SetupError, Shape, Tensor,
};
use tracing::{debug, info, warn};

use crate::ml::{InputLayout, InputSize, ModelConfig};

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
///
/// Every pass runs through `infer_async` + `wait`, so [`Self::start`] /
/// [`Self::wait`] can overlap a pass with other work; [`Self::infer`]
/// is the two back to back.
pub struct OpenVinoEngine {
    /// Cached `ModelInfo` snapshot — built from the supplied
    /// [`ModelConfig`] up front so the hot path does not touch any
    /// OpenVINO API for static metadata.
    info: ModelInfo,
    /// OpenVINO core handle.  Held for the lifetime of the engine
    /// so the dynamically-loaded library stays mapped.
    #[allow(
        dead_code,
        reason = "Kept alive so libopenvino.so stays mapped for the InferRequest's lifetime."
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
    /// Tensor name of the frame input (model input 0).  Captured once
    /// at load so the hot path does not call OpenVINO metadata APIs.
    input_name: String,
    /// Same as [`Self::input_name`] but for the model's chosen
    /// output port (`config.output_index`).
    output_name: String,
    /// State carried between passes of a recurrent model; empty for
    /// stateless models.
    state: RecurrentState,
    /// A pass was started and not yet collected by [`Self::wait`].
    in_flight: bool,
}

/// The reused input tensor plus the shape it was allocated for.
///
/// `shape` is the OpenVINO `i64` form (matching the `shape_i64` built in
/// [`OpenVinoEngine::start`]) so the per-frame "do we need to
/// reallocate?" check is a plain slice compare with no conversion.
struct CachedInputTensor {
    shape: Vec<i64>,
    tensor: Tensor,
}

/// Double-buffered recurrent state.
///
/// Each [`StatePair`] owns two tensors of the state's shape.  One is
/// bound to the pair's model input, the other to its model output; after
/// every pass the roles swap, so the output of pass N becomes the input
/// of pass N+1 without copying.  `current` is the index of the buffers
/// bound to the inputs (shared by all pairs — they always swap
/// together).
struct RecurrentState {
    slots: Vec<StateSlot>,
    current: usize,
}

struct StateSlot {
    input: String,
    output: String,
    buffers: [Tensor; 2],
}

/// Upper bound for one pass.  A pass that takes longer means the device
/// hung; failing it lets the segmentation fallback take over instead of
/// freezing the worker.
const WAIT_TIMEOUT_MS: i64 = 2_000;

/// Whether the cached input tensor must be (re)allocated for `shape`.
///
/// Pure decision extracted from the hot path so it is unit-testable
/// without an OpenVINO runtime. `None` (no tensor yet) or a shape
/// mismatch both force a reallocation; an exact match reuses.
fn needs_realloc(cached_shape: Option<&[i64]>, shape: &[i64]) -> bool {
    cached_shape != Some(shape)
}

/// Static shape of the frame input for a frame-sized model.
fn frame_input_shape(config: &ModelConfig) -> [i64; 4] {
    let (w, h) = (
        i64::from(config.input_width),
        i64::from(config.input_height),
    );
    match config.input_layout {
        InputLayout::Nchw => [1, 3, h, w],
        // `InputLayout` is non-exhaustive; NHWC is the schema default.
        _ => [1, h, w, 3],
    }
}

/// Static dimensions of a state tensor from the partial shape OpenVINO
/// infers for its output once the frame input is static.  A dynamic
/// leading (batch) dimension becomes 1; any other dynamic dimension
/// means the state shape does not follow from the frame size.
fn static_state_dims(name: &str, shape: &PartialShape) -> Result<Vec<i64>, InferenceError> {
    shape
        .get_dimensions()
        .iter()
        .enumerate()
        .map(|(i, d)| match (d.is_dynamic(), i) {
            (false, _) => Ok(d.get_min()),
            (true, 0) => Ok(1),
            (true, _) => Err(InferenceError::ModelLoadFailed {
                reason: format!(
                    "state output `{name}` has a dynamic dimension {i} even with a static \
                     frame input"
                ),
            }),
        })
        .collect()
}

fn load_err(what: &str) -> impl FnOnce(openvino::InferenceError) -> InferenceError + '_ {
    move |e| InferenceError::ModelLoadFailed {
        reason: format!("{what}: {e}"),
    }
}

fn infer_err(what: &str) -> impl FnOnce(openvino::InferenceError) -> InferenceError + '_ {
    move |e| InferenceError::InferenceFailed {
        reason: format!("{what}: {e}"),
    }
}

impl OpenVinoEngine {
    /// Load a model from disk and prepare an inference request.
    ///
    /// `device` is the OpenVINO device string — passed verbatim to
    /// `Core::compile_model` (`"CPU"`, `"GPU"`, `"NPU"`,
    /// `"AUTO:NPU,GPU,CPU"`, …).
    ///
    /// # Errors
    ///
    /// See [`Self::load_with_cache`].
    pub fn load(
        model_path: &Path,
        config: &ModelConfig,
        device: &str,
    ) -> Result<Self, InferenceError> {
        Self::load_with_cache(model_path, config, device, None)
    }

    /// [`Self::load`] with an optional compiled-model cache directory
    /// (OpenVINO `CACHE_DIR`).  A cache hit skips device compilation,
    /// which takes seconds on the NPU.  `device` must then be a single
    /// device such as `"NPU"`; a cache directory that cannot be set is
    /// logged and ignored.
    ///
    /// Frame-sized models ([`InputSize::Frame`]) are reshaped to the
    /// resolved input size, and the inputs of recurrent models' state
    /// to the shapes their outputs take at that size, so the model is
    /// fully static — the NPU compiles nothing else.
    ///
    /// # Errors
    ///
    /// * [`InferenceError::BackendUnavailable`] — `libopenvino_c.so`
    ///   cannot be loaded.  Carries [`LIBRARY_HINT`].
    /// * [`InferenceError::ModelNotFound`] — `model_path` does not
    ///   exist.
    /// * [`InferenceError::ModelLoadFailed`] — the model cannot be read,
    ///   reshaped or compiled for `device`.
    /// * [`InferenceError::InvalidModelConfig`] — the sidecar names
    ///   outputs or state the model does not have.
    pub fn load_with_cache(
        model_path: &Path,
        config: &ModelConfig,
        device: &str,
        cache_dir: Option<&Path>,
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
        let mut model = core.read_model_from_file(model_path_str, "").map_err(|e| {
            InferenceError::ModelLoadFailed {
                reason: format!("read_model_from_file({}): {e}", model_path.display()),
            }
        })?;
        let input_name = model
            .get_input_by_index(0)
            .and_then(|node| node.get_name())
            .map_err(load_err("cannot read input port name"))?;
        let output_names = (0..model.get_outputs_len().map_err(load_err("outputs_len"))?)
            .map(|i| model.get_output_by_index(i).and_then(|n| n.get_name()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(load_err("cannot read output port names"))?;
        let output_index = config.output_index;
        let output_name = output_names.get(output_index).cloned().ok_or_else(|| {
            InferenceError::InvalidModelConfig {
                reason: format!(
                    "output_index {output_index} out of range: the model has {} outputs",
                    output_names.len()
                ),
            }
        })?;
        check_state_names(config, &input_name, &output_name, &output_names)?;

        let state_shapes = make_static(&mut model, config, &input_name)?;

        let device_type: DeviceType = device.into();
        if let Some(dir) = cache_dir {
            let dir = dir.to_string_lossy();
            if let Err(e) = core.set_property(&device_type, &RwPropertyKey::CacheDir, &dir) {
                warn!(device, cache_dir = %dir, error = %e, "openvino model cache not enabled");
            }
        }
        let mut compiled = core.compile_model(&model, device_type).map_err(|e| {
            InferenceError::ModelLoadFailed {
                reason: format!("compile_model(device={device}): {e}"),
            }
        })?;
        let mut request = compiled
            .create_infer_request()
            .map_err(load_err("create_infer_request"))?;
        let state = RecurrentState::bind(config, &state_shapes, &mut request)?;
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
            input_size = ?(config.input_width, config.input_height),
            state_tensors = state.slots.len(),
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
            state,
            in_flight: false,
        })
    }
}

impl Drop for OpenVinoEngine {
    /// Wait out a pass still in flight: it writes into tensors this
    /// engine owns.
    fn drop(&mut self) {
        if self.in_flight
            && let Err(e) = self.request.wait(WAIT_TIMEOUT_MS)
        {
            warn!(error = %e, "in-flight pass did not finish before the engine was dropped");
        }
    }
}

/// Check that the sidecar's state pairs name real ports and do not
/// collide with the frame input or the consumed output.
fn check_state_names(
    config: &ModelConfig,
    input_name: &str,
    output_name: &str,
    output_names: &[String],
) -> Result<(), InferenceError> {
    for pair in &config.state {
        let reason = if pair.input == input_name {
            format!(
                "state input `{}` is the frame input (model input 0)",
                pair.input
            )
        } else if pair.output == output_name {
            format!(
                "state output `{}` is the consumed output (output_index)",
                pair.output
            )
        } else if !output_names.contains(&pair.output) {
            format!("state output `{}` is not a model output", pair.output)
        } else {
            continue;
        };
        return Err(InferenceError::InvalidModelConfig { reason });
    }
    Ok(())
}

/// Reshape frame-sized and recurrent models to static shapes.  Returns
/// the static shape of every state input, in `config.state` order.
fn make_static(
    model: &mut Model,
    config: &ModelConfig,
    input_name: &str,
) -> Result<Vec<Vec<i64>>, InferenceError> {
    let mut shapes: Vec<(String, PartialShape)> = Vec::new();
    if config.input_size == InputSize::Frame {
        let dims = frame_input_shape(config);
        let shape = PartialShape::new_static(4, &dims).map_err(load_err("frame input shape"))?;
        model
            .reshape_input_by_name(input_name, &shape)
            .map_err(load_err("reshape frame input"))?;
        shapes.push((input_name.to_owned(), shape));
    }
    if !config.is_recurrent() {
        return Ok(Vec::new());
    }
    // With the frame input static OpenVINO propagates the state outputs'
    // shapes; the matching inputs take the same shapes.
    let outputs_len = model.get_outputs_len().map_err(load_err("outputs_len"))?;
    let mut state_dims = Vec::with_capacity(config.state.len());
    for pair in &config.state {
        let mut found = None;
        for i in 0..outputs_len {
            let node = model.get_output_by_index(i).map_err(load_err("output"))?;
            if node.get_name().map_err(load_err("output name"))? == pair.output {
                found = Some(node.get_partial_shape().map_err(load_err("state shape"))?);
                break;
            }
        }
        let partial = found.ok_or_else(|| InferenceError::InvalidModelConfig {
            reason: format!("state output `{}` is not a model output", pair.output),
        })?;
        let dims = static_state_dims(&pair.output, &partial)?;
        let rank = i64::try_from(dims.len()).unwrap_or(i64::MAX);
        let shape = PartialShape::new_static(rank, &dims).map_err(load_err("state shape"))?;
        shapes.push((pair.input.clone(), shape));
        state_dims.push(dims);
    }
    let refs: Vec<(&str, &PartialShape)> = shapes.iter().map(|(n, s)| (n.as_str(), s)).collect();
    model
        .reshape(&refs)
        .map_err(load_err("reshape state inputs"))?;
    Ok(state_dims)
}

impl RecurrentState {
    /// Allocate zeroed double buffers for every state pair and bind
    /// buffer 0 to the inputs, buffer 1 to the outputs.
    fn bind(
        config: &ModelConfig,
        shapes: &[Vec<i64>],
        request: &mut InferRequest,
    ) -> Result<Self, InferenceError> {
        let mut slots = Vec::with_capacity(config.state.len());
        for (pair, dims) in config.state.iter().zip(shapes) {
            let shape = Shape::new(dims).map_err(load_err("state tensor shape"))?;
            let make = || -> Result<Tensor, InferenceError> {
                let mut t =
                    Tensor::new(ElementType::F32, &shape).map_err(load_err("state tensor"))?;
                t.get_data_mut::<f32>()
                    .map_err(load_err("state tensor data"))?
                    .fill(0.0);
                Ok(t)
            };
            slots.push(StateSlot {
                input: pair.input.clone(),
                output: pair.output.clone(),
                buffers: [make()?, make()?],
            });
        }
        let state = Self { slots, current: 0 };
        state
            .rebind(request)
            .map_err(|e| InferenceError::ModelLoadFailed {
                reason: e.to_string(),
            })?;
        Ok(state)
    }

    /// Bind `buffers[current]` to the state inputs and the other buffer
    /// to the state outputs.
    fn rebind(&self, request: &mut InferRequest) -> Result<(), InferenceError> {
        let next = 1 - self.current;
        for slot in &self.slots {
            request
                .set_tensor(&slot.input, &slot.buffers[self.current])
                .map_err(infer_err("bind state input"))?;
            request
                .set_tensor(&slot.output, &slot.buffers[next])
                .map_err(infer_err("bind state output"))?;
        }
        Ok(())
    }

    /// After a pass: what was written to the outputs becomes the next
    /// pass's input.
    fn advance(&mut self, request: &mut InferRequest) -> Result<(), InferenceError> {
        if self.slots.is_empty() {
            return Ok(());
        }
        self.current = 1 - self.current;
        self.rebind(request)
    }

    /// Zero the buffers the next pass reads.
    fn clear(&mut self) {
        let current = self.current;
        for slot in &mut self.slots {
            match slot.buffers[current].get_data_mut::<f32>() {
                Ok(data) => data.fill(0.0),
                Err(e) => warn!(state = %slot.input, error = %e, "cannot clear model state"),
            }
        }
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
        self.start(input)?;
        self.wait()
    }

    fn supports_async(&self) -> bool {
        true
    }

    fn start(&mut self, input: InferenceInput<'_>) -> Result<(), InferenceError> {
        if self.in_flight {
            return Err(InferenceError::InferenceFailed {
                reason: "a pass is already in flight; call wait() first".into(),
            });
        }
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
        // changes (once, for a static model). The hot path below just
        // overwrites its buffer — no per-frame allocation.
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
            // Bind once. OpenVINO reads the tensor lazily at inference
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
            .map_err(infer_err("tensor.get_data_mut"))?
            .copy_from_slice(input.data);
        self.request
            .infer_async()
            .map_err(infer_err("infer_async"))?;
        self.in_flight = true;
        Ok(())
    }

    fn wait(&mut self) -> Result<InferenceOutput, InferenceError> {
        if !self.in_flight {
            return Err(InferenceError::InferenceFailed {
                reason: "wait() without a pass in flight".into(),
            });
        }
        self.in_flight = false;
        self.request
            .wait(WAIT_TIMEOUT_MS)
            .map_err(infer_err("wait"))?;
        let out_tensor = self.request.get_tensor(&self.output_name).map_err(|e| {
            InferenceError::InferenceFailed {
                reason: format!("get_tensor({}): {e}", self.output_name),
            }
        })?;
        let out_shape = out_tensor
            .get_shape()
            .map_err(infer_err("output tensor get_shape"))?;
        let out_shape_usize: Vec<usize> = out_shape
            .get_dimensions()
            .iter()
            .map(|&d| d.max(0) as usize)
            .collect();
        let data = out_tensor
            .get_data::<f32>()
            .map_err(infer_err("output tensor get_data"))?
            .to_vec();
        self.state.advance(&mut self.request)?;
        Ok(InferenceOutput {
            data,
            shape: out_shape_usize,
        })
    }

    fn reset_state(&mut self) {
        if self.in_flight {
            self.in_flight = false;
            if let Err(e) = self.request.wait(WAIT_TIMEOUT_MS) {
                warn!(error = %e, "dropping in-flight pass on state reset failed");
            }
        }
        self.state.clear();
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
            OpenVinoEngine::load(&model_path, &cfg, "CPU").expect("openvino CPU load on this host");
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
        let mut engine = OpenVinoEngine::load(&model_path, &cfg, "CPU").expect("openvino load");
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
        let mut engine = OpenVinoEngine::load(&model_path, &cfg, "CPU").expect("openvino load");
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

    /// Recurrent state round trip on a real Robust Video Matting model:
    /// the second pass on the same frame differs from the first (state
    /// fed back through the swapped buffers), and after `reset_state`
    /// a pass reproduces the first one.
    ///
    /// ```sh
    /// FLUXFRAME_RVM_MODEL=/path/rvm.onnx [FLUXFRAME_RVM_DEVICE=NPU] \
    ///   cargo test -p fluxframe-effects --lib rvm_state -- --ignored
    /// ```
    #[test]
    #[ignore = "requires OpenVINO and a prepared RVM model in FLUXFRAME_RVM_MODEL"]
    fn rvm_state_carries_between_passes_and_resets() {
        let Some(model_path) = std::env::var_os("FLUXFRAME_RVM_MODEL").map(PathBuf::from) else {
            eprintln!("skipping: FLUXFRAME_RVM_MODEL not set");
            return;
        };
        let device = std::env::var("FLUXFRAME_RVM_DEVICE").unwrap_or_else(|_| "CPU".into());
        let mut cfg = ModelConfig::from_toml_str(
            r#"
name = "rvm"
input_size = "frame"
input_layout = "NCHW"
output_layout = "NCHW"
state = [
  { output = "r1o", input = "r1i" },
  { output = "r2o", input = "r2i" },
  { output = "r3o", input = "r3i" },
  { output = "r4o", input = "r4i" },
]
"#,
        )
        .expect("sidecar");
        let (w, h) = (320_usize, 192_usize);
        cfg.resolve_input_size(w as u32, h as u32);
        let mut engine =
            OpenVinoEngine::load(&model_path, &cfg, &device).expect("rvm loads with static shapes");
        let shape = [1, 3, h, w];
        // Deterministic noise: plain shapes segment to all-zero alpha,
        // on which the state has no visible effect; noise gets a
        // partial alpha that the fed-back state measurably shifts.
        let mut seed = 0x2545_f491_u32;
        let frame: Vec<f32> = (0..3 * w * h)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed >> 8) as f32 / (1 << 24) as f32
            })
            .collect();
        let input = InferenceInput {
            data: &frame,
            shape: &shape,
        };
        let first = engine.infer(input.clone()).expect("pass 1");
        assert_eq!(first.shape, vec![1, 1, h, w], "alpha at frame size");
        engine.start(input.clone()).expect("async start");
        let second = engine.wait().expect("pass 2");
        let diff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
        };
        assert!(
            diff(&first.data, &second.data) > 1e-5,
            "state must influence the second pass"
        );
        engine.reset_state();
        let after_reset = engine.infer(input).expect("pass after reset");
        assert!(
            diff(&first.data, &after_reset.data) < 1e-4,
            "reset must reproduce the first pass"
        );
    }
}
