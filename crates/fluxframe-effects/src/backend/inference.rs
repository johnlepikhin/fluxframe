//! Inference-engine factory + sticky-fallback decorator.
//!
//! Stage 6 introduced the seam (single candidate `OnnxEngine`);
//! Stage 8 wires OpenVINO behind the `openvino` Cargo feature and
//! makes it the **default** path so a stock binary picks the best
//! accelerator (NPU → GPU → CPU) without operator action.
//!
//! Selection logic:
//!
//! * `InferenceBackendChoice::Auto` (default) → try OpenVINO with
//!   device `AUTO:NPU,GPU,CPU` (or whatever `FLUXFRAME_OPENVINO_DEVICE`
//!   overrides to).  OpenVINO's AUTO meta-plugin sorts the
//!   compile-target list and falls back per-device internally —
//!   so the call succeeds as long as at least the CPU plugin is
//!   loadable, which it always is when the OpenVINO runtime is
//!   reachable.  If the OpenVINO runtime itself is not reachable
//!   (`libopenvino_c.so` not on the loader path, feature disabled
//!   at compile time), silently fall back to ORT CPU.
//! * `InferenceBackendChoice::Cpu` → force ORT CPU.  Diagnostic
//!   escape hatch when an operator wants to bypass the OpenVINO
//!   path entirely.
//! * `InferenceBackendChoice::OpenVino` → force OpenVINO with the
//!   env-configured device.  Hard error if the runtime is unreachable
//!   or the feature is disabled (no silent demotion).
//!
//! No `StickyInferenceFallback` wraps the production result: each
//! branch produces exactly one engine.  The decorator stays wired
//! for tests via mocks (see `tests/backend_fallback.rs`).  This
//! sidesteps the TBB-initialisation deadlock that would otherwise
//! occur if both ORT and OpenVINO loaded into the same process at
//! startup (both runtimes spin up their own worker pools and the
//! second `dlopen` waits on a futex the first's init holds).
//!
//! Lives behind `cfg(feature = "ml")` because `OnnxEngine` is gated
//! the same way — the slim build (`default-features = false`) drops
//! the entire ML inference layer.

use std::path::Path;
use std::sync::Arc;

use fluxframe_core::error::InferenceError;
use fluxframe_core::metrics::Counters;
use fluxframe_core::traits::{InferenceEngine, InferenceInput, InferenceOutput, ModelInfo};
use tracing::{info, warn};

use crate::backend::overrides::{BackendOverrides, InferenceBackendChoice};
#[cfg(feature = "openvino")]
use crate::ml::OpenVinoEngine;
use crate::ml::{ModelConfig, OnnxEngine};

/// Env-var read by [`build_inference_engine`] to override the
/// OpenVINO device string passed to `Core::compile_model`.
///
/// Recognised values are anything OpenVINO accepts: `"CPU"`,
/// `"GPU"`, `"GPU.0"`, `"NPU"`, `"AUTO:NPU,GPU,CPU"`, `"HETERO:CPU,GPU"`,
/// etc.  When unset, [`DEFAULT_OPENVINO_DEVICE`] is used.
#[cfg(feature = "openvino")]
pub const ENV_OPENVINO_DEVICE: &str = "FLUXFRAME_OPENVINO_DEVICE";

/// Default OpenVINO device when no env override is set.
///
/// `AUTO:NPU,GPU,CPU` lets OpenVINO's AUTO meta-plugin pick the
/// best available accelerator at compile-time: NPU first
/// (`intel_vpu` kernel module + UMD + NPU compiler), then GPU
/// (Intel Compute Runtime + OpenCL ICD), then CPU plugin (always
/// available).  All three plugins + their dependencies ship with
/// the `openvino-full`, `intel-npu-driver` and
/// `intel-compute-runtime` Guix packages from the `johnlepikhin`
/// channel.
///
/// Operators who want to override (e.g. force a specific device
/// for diagnostics) can set `FLUXFRAME_OPENVINO_DEVICE` to anything
/// OpenVINO accepts: `"NPU"`, `"GPU"`, `"CPU"`, `"HETERO:NPU,CPU"`,
/// `"AUTO:NPU,CPU"` (skip GPU), etc.
#[cfg(feature = "openvino")]
pub const DEFAULT_OPENVINO_DEVICE: &str = "AUTO:NPU,GPU,CPU";

#[cfg(feature = "openvino")]
fn openvino_device_from_env() -> String {
    std::env::var(ENV_OPENVINO_DEVICE).unwrap_or_else(|_| DEFAULT_OPENVINO_DEVICE.into())
}

/// Build an inference engine honouring the supplied overrides.
///
/// `counters` is the supervisor's shared counter bundle, threaded
/// through `ProcessingContext::counters` by the runtime.  Reserved
/// for the future sticky-fallback wiring (currently unused in
/// production paths — see module-level "no sticky" note).
///
/// # Errors
///
/// * [`InferenceError`] from the chosen runtime's loader (missing
///   model file, runtime not available, invalid config).
/// * [`InferenceError::BackendUnavailable`] when `OpenVino` is
///   forced and the runtime is not reachable (or the Cargo feature
///   is disabled at compile time).  `Auto` never surfaces this — it
///   silently falls back to ORT CPU.
#[allow(
    clippy::needless_pass_by_value,
    reason = "Counters are wired into the signature for a future sticky-fallback path \
              (see module docstring); currently the factory returns exactly one engine \
              per call and counters are unused in production."
)]
pub fn build_inference_engine(
    model_path: &Path,
    config: ModelConfig,
    overrides: BackendOverrides,
    counters: Option<Arc<Counters>>,
) -> Result<Box<dyn InferenceEngine + Send>, InferenceError> {
    let _ = &counters;
    let choice = overrides.inference;
    let (engine, backend_name) = match choice {
        InferenceBackendChoice::Cpu => {
            let engine = OnnxEngine::load(model_path, config)?;
            (
                Box::new(engine) as Box<dyn InferenceEngine + Send>,
                "cpu".to_string(),
            )
        }
        InferenceBackendChoice::OpenVino => build_openvino_forced(model_path, config)?,
        InferenceBackendChoice::Auto => build_auto(model_path, config)?,
    };
    info!(
        component = "inference",
        backend = %backend_name,
        override_ = ?choice,
        "selected inference backend",
    );
    Ok(engine)
}

/// `Auto` branch: try OpenVINO with the env-configured device (or
/// the `AUTO:NPU,GPU,CPU` default), silently fall back to ORT CPU
/// if the OpenVINO runtime is unreachable.
///
/// When the `openvino` Cargo feature is disabled this collapses to
/// "always ORT CPU", which is the Stage 6 behaviour.
#[cfg(feature = "openvino")]
fn build_auto(
    model_path: &Path,
    config: ModelConfig,
) -> Result<(Box<dyn InferenceEngine + Send>, String), InferenceError> {
    let device = openvino_device_from_env();
    match OpenVinoEngine::load(model_path, config.clone(), &device) {
        Ok(engine) => Ok((Box::new(engine), format!("openvino[{device}]"))),
        Err(ov_err) => {
            warn!(
                error = %ov_err,
                "openvino probe failed; falling back to ORT CPU",
            );
            let engine = OnnxEngine::load(model_path, config)?;
            Ok((Box::new(engine), "cpu".to_string()))
        }
    }
}

#[cfg(not(feature = "openvino"))]
fn build_auto(
    model_path: &Path,
    config: ModelConfig,
) -> Result<(Box<dyn InferenceEngine + Send>, String), InferenceError> {
    let engine = OnnxEngine::load(model_path, config)?;
    Ok((Box::new(engine), "cpu".to_string()))
}

/// Forced-OpenVINO branch: probe must succeed (no silent fallback
/// to ORT).  Lives behind `cfg(feature = "openvino")`; the !feature
/// counterpart below emits a structured "feature disabled" error.
#[cfg(feature = "openvino")]
fn build_openvino_forced(
    model_path: &Path,
    config: ModelConfig,
) -> Result<(Box<dyn InferenceEngine + Send>, String), InferenceError> {
    let device = openvino_device_from_env();
    let engine = OpenVinoEngine::load(model_path, config, &device)?;
    Ok((Box::new(engine), format!("openvino[{device}]")))
}

#[cfg(not(feature = "openvino"))]
fn build_openvino_forced(
    _model_path: &Path,
    _config: ModelConfig,
) -> Result<(Box<dyn InferenceEngine + Send>, String), InferenceError> {
    Err(InferenceError::BackendUnavailable {
        reason: "FLUXFRAME_FORCE_INFERENCE_BACKEND=openvino but the `openvino` Cargo \
                 feature is disabled in this build"
            .into(),
        hint: "rebuild with `--features fluxframe-effects/openvino` or unset the \
               override"
            .into(),
    })
}

/// Sticky one-way fallback decorator for [`InferenceEngine`].
///
/// Semantically identical to [`crate::backend::blur_factory::StickyBlurFallback`]
/// but for the inference path.  See that type's documentation for the
/// transition protocol; counters target
/// [`Counters::inc_inference_runtime_fallback_gpu_to_cpu`] here.
///
/// Not used in Stage 6 production paths — wired so the seam is
/// fully testable through mocks.
pub struct StickyInferenceFallback {
    state: InferenceState,
    primary_name: &'static str,
    secondary_name: &'static str,
    counters: Arc<Counters>,
}

/// Internal state machine.  `Poisoned` is a transient marker held
/// only across the transition; it must not be observable to a caller
/// outside that path.
enum InferenceState {
    Primary {
        primary: Box<dyn InferenceEngine + Send>,
        secondary: Box<dyn InferenceEngine + Send>,
    },
    Secondary {
        secondary: Box<dyn InferenceEngine + Send>,
    },
    Poisoned,
}

impl StickyInferenceFallback {
    /// Construct a decorator.  See [`Self::primary_name`] in the
    /// blur counterpart for naming conventions.
    #[must_use]
    pub fn new(
        primary: Box<dyn InferenceEngine + Send>,
        secondary: Box<dyn InferenceEngine + Send>,
        primary_name: &'static str,
        secondary_name: &'static str,
        counters: Arc<Counters>,
    ) -> Self {
        Self {
            state: InferenceState::Primary { primary, secondary },
            primary_name,
            secondary_name,
            counters,
        }
    }

    /// `true` iff the primary has already failed and inference is now
    /// running on the secondary permanently.
    #[must_use]
    pub fn has_fallen_back(&self) -> bool {
        matches!(self.state, InferenceState::Secondary { .. })
    }
}

impl InferenceEngine for StickyInferenceFallback {
    fn model_info(&self) -> ModelInfo {
        match &self.state {
            InferenceState::Primary { primary, .. } => primary.model_info(),
            InferenceState::Secondary { secondary } => secondary.model_info(),
            InferenceState::Poisoned => panic!(
                "sticky inference fallback observed in poisoned state — \
                 this indicates a panic in a previous infer() call",
            ),
        }
    }

    fn infer(&mut self, input: InferenceInput<'_>) -> Result<InferenceOutput, InferenceError> {
        match &mut self.state {
            InferenceState::Secondary { secondary } => secondary.infer(input),
            InferenceState::Primary { primary, .. } => {
                // `InferenceInput<'_>` is `Clone` (two slice refs +
                // PhantomData): cheap to copy for the retry path.
                let retry_input = input.clone();
                match primary.infer(input) {
                    Ok(out) => Ok(out),
                    Err(e) => {
                        warn!(
                            from = self.primary_name,
                            to = self.secondary_name,
                            error = %e,
                            "sticky inference backend fallback",
                        );
                        self.counters.inc_inference_runtime_fallback_gpu_to_cpu();
                        let InferenceState::Primary { secondary, .. } =
                            std::mem::replace(&mut self.state, InferenceState::Poisoned)
                        else {
                            unreachable!("matched Primary above")
                        };
                        self.state = InferenceState::Secondary { secondary };
                        let InferenceState::Secondary { secondary } = &mut self.state else {
                            unreachable!("just installed Secondary")
                        };
                        secondary.infer(retry_input)
                    }
                }
            }
            InferenceState::Poisoned => Err(InferenceError::InferenceFailed {
                reason: "sticky inference fallback in poisoned state (programming error)".into(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxframe_core::frame::PixelFormat;
    use std::cell::Cell;

    /// Test-only InferenceEngine that errors on the first N calls
    /// then succeeds with a known output.  Interior mutability via
    /// `Cell` keeps the trait's `&mut self` happy without polluting
    /// the production trait shape.
    struct FailingEngine {
        calls: Cell<u32>,
        fail_first_n: u32,
        sentinel: f32,
    }

    impl FailingEngine {
        fn new(fail_first_n: u32, sentinel: f32) -> Self {
            Self {
                calls: Cell::new(0),
                fail_first_n,
                sentinel,
            }
        }
    }

    impl InferenceEngine for FailingEngine {
        fn model_info(&self) -> ModelInfo {
            ModelInfo {
                name: Arc::from("failing-mock"),
                input_width: 1,
                input_height: 1,
                input_format: PixelFormat::Rgb,
            }
        }

        fn infer(&mut self, _input: InferenceInput<'_>) -> Result<InferenceOutput, InferenceError> {
            let n = self.calls.get();
            self.calls.set(n + 1);
            if n < self.fail_first_n {
                Err(InferenceError::InferenceFailed {
                    reason: format!("synthetic failure call {n}"),
                })
            } else {
                Ok(InferenceOutput {
                    data: vec![self.sentinel; 1],
                    shape: vec![1],
                })
            }
        }
    }

    /// Test-only engine that always succeeds with a distinct sentinel.
    struct SentinelEngine {
        sentinel: f32,
        name: &'static str,
    }
    impl InferenceEngine for SentinelEngine {
        fn model_info(&self) -> ModelInfo {
            ModelInfo {
                name: Arc::from(self.name),
                input_width: 1,
                input_height: 1,
                input_format: PixelFormat::Rgb,
            }
        }
        fn infer(&mut self, _input: InferenceInput<'_>) -> Result<InferenceOutput, InferenceError> {
            Ok(InferenceOutput {
                data: vec![self.sentinel; 1],
                shape: vec![1],
            })
        }
    }

    fn dummy_input() -> Vec<f32> {
        vec![0.0]
    }

    #[test]
    fn factory_propagates_invalid_model_path() {
        // Sanity check that the factory does NOT swallow load errors.
        // ORT may be entirely absent in the test env, in which case
        // `BackendUnavailable` is returned instead of `ModelNotFound`
        // — both are acceptable, the point is we get a structured
        // error rather than a panic.  Avoid `.expect_err()` here
        // because the Ok variant `Box<dyn InferenceEngine + Send>` is
        // not `Debug`, which `expect_err` would require for its
        // panic message.
        let cfg = ModelConfig::new("dummy", 1, 1);
        let result = build_inference_engine(
            Path::new("/nonexistent/model.onnx"),
            cfg,
            BackendOverrides::default(),
            None,
        );
        match result {
            Ok(_) => panic!("must fail on missing model"),
            Err(
                InferenceError::ModelNotFound { .. } | InferenceError::BackendUnavailable { .. },
            ) => {}
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn factory_accepts_counters_argument() {
        // Stage 8 step 1 wiring: counters is accepted; production
        // sticky wiring is deferred (see module docstring), but the
        // call must compile and not panic.
        let cfg = ModelConfig::new("dummy", 1, 1);
        let counters = Arc::new(Counters::new());
        let _ = build_inference_engine(
            Path::new("/nonexistent/model.onnx"),
            cfg,
            BackendOverrides::default(),
            Some(counters),
        );
    }

    /// `OpenVino` override path: with the Cargo feature **disabled**
    /// the factory MUST return a structured `BackendUnavailable`
    /// rather than silently falling back to ORT.
    #[cfg(not(feature = "openvino"))]
    #[test]
    fn factory_openvino_override_errors_when_feature_disabled() {
        use crate::backend::overrides::InferenceBackendChoice;
        let cfg = ModelConfig::new("dummy", 1, 1);
        let overrides =
            BackendOverrides::default().with_inference(InferenceBackendChoice::OpenVino);
        let result =
            build_inference_engine(Path::new("/nonexistent/model.onnx"), cfg, overrides, None);
        match result {
            Ok(_) => panic!("must hard-error without openvino feature"),
            Err(InferenceError::BackendUnavailable { hint, .. }) => {
                assert!(
                    hint.contains("--features"),
                    "hint must guide rebuild: {hint}"
                );
            }
            Err(other) => panic!("expected BackendUnavailable, got: {other:?}"),
        }
    }

    /// `OpenVino` override with the feature enabled: probes the real
    /// runtime + reads the model.  Hard-error path tested with a
    /// nonexistent model file — both `ModelNotFound` (when the file
    /// check trips first) and `BackendUnavailable` (when the
    /// libopenvino path tripped) are acceptable structured outcomes.
    #[cfg(feature = "openvino")]
    #[test]
    fn factory_openvino_override_probes_real_runtime() {
        use crate::backend::overrides::InferenceBackendChoice;
        let cfg = ModelConfig::new("dummy", 1, 1);
        let overrides =
            BackendOverrides::default().with_inference(InferenceBackendChoice::OpenVino);
        let result =
            build_inference_engine(Path::new("/nonexistent/model.onnx"), cfg, overrides, None);
        match result {
            Ok(_) => panic!("nonexistent model must not succeed"),
            Err(
                InferenceError::ModelNotFound { .. }
                | InferenceError::BackendUnavailable { .. }
                | InferenceError::ModelLoadFailed { .. },
            ) => {}
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn fallback_starts_on_primary_and_no_counter_inc() {
        let counters = Arc::new(Counters::new());
        let decorator = StickyInferenceFallback::new(
            Box::new(SentinelEngine {
                sentinel: 1.0,
                name: "primary",
            }),
            Box::new(SentinelEngine {
                sentinel: 2.0,
                name: "secondary",
            }),
            "primary",
            "secondary",
            Arc::clone(&counters),
        );
        assert!(!decorator.has_fallen_back());
        assert_eq!(counters.snapshot().inference_runtime_fallback_gpu_to_cpu, 0);
    }

    #[test]
    fn fallback_transitions_on_first_error() {
        let counters = Arc::new(Counters::new());
        let mut decorator = StickyInferenceFallback::new(
            Box::new(FailingEngine::new(1, 1.0)),
            Box::new(SentinelEngine {
                sentinel: 9.0,
                name: "cpu",
            }),
            "gpu",
            "cpu",
            Arc::clone(&counters),
        );
        let data = dummy_input();
        let shape = [1];
        let out = decorator
            .infer(InferenceInput {
                data: &data,
                shape: &shape,
            })
            .expect("infer succeeds via secondary");
        // Output came from the SECONDARY sentinel — primary erred.
        assert_eq!(out.data, vec![9.0]);
        assert!(decorator.has_fallen_back());
        assert_eq!(
            counters.snapshot().inference_runtime_fallback_gpu_to_cpu,
            1,
            "counter must increment exactly once on transition",
        );
    }

    #[test]
    fn fallback_subsequent_calls_skip_primary() {
        // Primary would fail forever; sticky behaviour means primary
        // is dropped after the first failure and never called again.
        let counters = Arc::new(Counters::new());
        let mut decorator = StickyInferenceFallback::new(
            Box::new(FailingEngine::new(1000, 0.0)),
            Box::new(SentinelEngine {
                sentinel: 7.0,
                name: "cpu",
            }),
            "gpu",
            "cpu",
            Arc::clone(&counters),
        );
        let data = dummy_input();
        let shape = [1];
        for _ in 0..5 {
            let out = decorator
                .infer(InferenceInput {
                    data: &data,
                    shape: &shape,
                })
                .expect("each call succeeds via secondary");
            assert_eq!(out.data, vec![7.0]);
        }
        assert_eq!(
            counters.snapshot().inference_runtime_fallback_gpu_to_cpu,
            1,
            "counter increments once across many subsequent calls",
        );
    }

    #[test]
    fn fallback_keeps_primary_when_it_succeeds() {
        let counters = Arc::new(Counters::new());
        let mut decorator = StickyInferenceFallback::new(
            Box::new(SentinelEngine {
                sentinel: 3.0,
                name: "primary",
            }),
            Box::new(SentinelEngine {
                sentinel: 4.0,
                name: "secondary",
            }),
            "primary",
            "secondary",
            Arc::clone(&counters),
        );
        let data = dummy_input();
        let shape = [1];
        let out = decorator
            .infer(InferenceInput {
                data: &data,
                shape: &shape,
            })
            .expect("primary ok");
        assert_eq!(out.data, vec![3.0]);
        assert!(!decorator.has_fallen_back());
        assert_eq!(counters.snapshot().inference_runtime_fallback_gpu_to_cpu, 0,);
    }

    #[test]
    fn model_info_follows_active_backend() {
        let counters = Arc::new(Counters::new());
        let mut decorator = StickyInferenceFallback::new(
            Box::new(FailingEngine::new(1, 0.0)),
            Box::new(SentinelEngine {
                sentinel: 0.0,
                name: "cpu-backend",
            }),
            "gpu",
            "cpu",
            counters,
        );
        // Before transition: primary's model_info.
        assert_eq!(&*decorator.model_info().name, "failing-mock");
        // Trigger transition.
        let data = dummy_input();
        let shape = [1];
        decorator
            .infer(InferenceInput {
                data: &data,
                shape: &shape,
            })
            .expect("ok via secondary");
        // After transition: secondary's model_info.
        assert_eq!(&*decorator.model_info().name, "cpu-backend");
    }
}
