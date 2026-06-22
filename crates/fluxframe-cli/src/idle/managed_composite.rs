// See the matching attribute in `idle/placeholder.rs` for the
// rationale: Step 4 wires the supervisor call sites; until then the
// items below are unused in the binary target but live in tests.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Stage 15 Step 4 wires ManagedComposite into the supervisor"
    )
)]

//! Lifecycle wrapper around [`CompositeEffect`].
//!
//! This wrapper exists for the (deferred) ability to drop the ONNX
//! inference engine on a long idle and rebuild it when a consumer
//! reconnects. (The `DeepIdle` state that would have triggered the
//! drop was removed in Stage 16; the unload itself is still future
//! work.) The effect trait itself stays pure — `process` is still a
//! per-frame compute call — so the lifecycle concern lives here in the
//! CLI layer rather than leaking into every
//! `VideoEffect` implementer.
//!
//! [`ManagedComposite`] owns a [`CompositeEffect`] outright and
//! exposes:
//!
//! * [`Self::unload`] — drop the currently loaded ONNX engine.
//! * [`Self::reload_engine`] — synchronously rebuild the engine
//!   via [`build_inference_engine`]. Step 4 wraps this in an
//!   off-worker reload thread.
//! * [`Self::is_ready`] — `true` when the supervisor can start
//!   pulling real frames through the wrapper again.
//!
//! The wrapper implements [`VideoEffect`] so it slots into the
//! existing [`EffectChain`] without touching chain internals; the
//! supervisor keeps a parallel handle for lifecycle control.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use fluxframe_core::Counters;
use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::frame::VideoFrame;
use fluxframe_core::plane::SubchainKind;
use fluxframe_core::traits::{RawEffectParams, VideoEffect};
use fluxframe_effects::CompositeEffect;
use fluxframe_effects::backend::{BackendOverrides, build_inference_engine};
use fluxframe_effects::ml::ModelConfig;

/// CLI-side wrapper that owns a [`CompositeEffect`] and an engine
/// readiness flag. Use [`Self::new`] to wrap a prepared composite
/// effect; the wrapper takes ownership and exposes the original
/// effect through the [`VideoEffect`] trait without changes.
pub(crate) struct ManagedComposite {
    inner: CompositeEffect,
    /// `true` whenever the composite is ready to run the full
    /// chain. Toggled by [`Self::unload`] / [`Self::reload_engine`];
    /// the supervisor's worker loop checks this flag to decide
    /// whether to call `process` or fall back to the placeholder.
    engine_ready: Arc<AtomicBool>,
    /// Optional `Arc<Counters>` snapshot used by the reload path.
    /// `None` until [`Self::set_counters`] is called by the
    /// supervisor right after `prepare()`. Without it
    /// [`build_inference_engine`] still works — counters are an
    /// optional telemetry sink.
    counters: Option<Arc<Counters>>,
}

impl ManagedComposite {
    /// Wrap an already-built [`CompositeEffect`]. The engine
    /// readiness flag starts `true` because `CompositeEffect::new`
    /// followed by `prepare()` produces a fully-loaded composite —
    /// the wrapper does not "own" the engine at construction time,
    /// it merely observes the lifecycle.
    #[must_use]
    pub(crate) fn new(inner: CompositeEffect) -> Self {
        Self {
            inner,
            engine_ready: Arc::new(AtomicBool::new(true)),
            counters: None,
        }
    }

    /// Hand over the supervisor's `Arc<Counters>`. The reload path
    /// passes this on to [`build_inference_engine`] so newly
    /// constructed engines record the same telemetry as the
    /// original.
    pub(crate) fn set_counters(&mut self, counters: Arc<Counters>) {
        self.counters = Some(counters);
    }

    /// Borrow the engine-ready flag. The supervisor's idle state
    /// machine consults this on every tick to choose between
    /// `Active` and `Placeholder` levels. Cheap `Arc` clone — the
    /// caller can stash it once outside the worker loop.
    #[must_use]
    pub(crate) fn engine_ready_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.engine_ready)
    }

    /// `true` when the inner composite has a loaded inference
    /// engine *and* the readiness flag has not been cleared by a
    /// pending reload. Both checks are required: the flag is
    /// flipped to `false` first (before `take_engine`) so the
    /// worker stops calling `process` before the engine vanishes.
    ///
    /// # Concurrency
    ///
    /// Safe to call only from the thread that owns `&ManagedComposite`
    /// (the supervisor's worker thread). Other threads must consult
    /// [`Self::engine_ready_handle`] in isolation — combining it with
    /// `engine_loaded()` from another thread would race against the
    /// `&mut self` mutations performed by `unload` / `reload_engine`.
    #[must_use]
    pub(crate) fn is_ready(&self) -> bool {
        self.engine_ready.load(Ordering::Acquire) && self.inner.segmentation().engine_loaded()
    }

    /// Drop the currently loaded ONNX engine. Idempotent: returns
    /// silently when no engine is loaded. The engine-ready flag is
    /// cleared *before* the take so a concurrent observer never
    /// sees `ready = true` without an engine.
    pub(crate) fn unload(&mut self) {
        self.engine_ready.store(false, Ordering::Release);
        let _ = self.inner.segmentation_mut().take_engine();
        tracing::info!(
            target: "fluxframe::idle",
            "managed_composite: engine unloaded; placeholder will flow until reload"
        );
    }

    /// Synchronously rebuild the ONNX engine using the
    /// model path + model-config snapshot the inner composite
    /// captured during `prepare()`. Sets the readiness flag to
    /// `true` only after the new engine is fully installed.
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::Inference`] when the engine cannot
    /// be rebuilt (model file vanished, model_config mismatch,
    /// ORT init failure).
    /// Returns [`EffectError::PrepareFailed`] when the inner
    /// composite never ran `prepare()` and has no model_config to
    /// rebuild against.
    pub(crate) fn reload_engine(&mut self) -> Result<(), EffectError> {
        let model_path: PathBuf = self.inner.segmentation().model_path().to_path_buf();
        let start = Instant::now();
        tracing::info!(
            target: "fluxframe::idle",
            model = %model_path.display(),
            "managed_composite: rebuilding engine after idle"
        );
        let model_config: ModelConfig = self
            .inner
            .segmentation()
            .model_config()
            .cloned()
            .ok_or_else(|| EffectError::PrepareFailed {
                name: CompositeEffect::NAME.to_string(),
                reason: "reload_engine called before prepare — no model_config to rebuild against"
                    .into(),
            })?;
        // Engine construction may block on disk I/O and ORT init.
        // The supervisor's Stage 15 Step 4 reload thread runs this
        // off the worker thread so placeholder flow is not stalled.
        let engine = build_inference_engine(
            &model_path,
            model_config,
            BackendOverrides::current(),
            self.counters.clone(),
        )
        .map_err(|source| {
            tracing::warn!(
                target: "fluxframe::idle",
                model = %model_path.display(),
                error = %source,
                "managed_composite: engine reload failed"
            );
            EffectError::Inference {
                name: CompositeEffect::NAME.to_string(),
                source,
            }
        })?;
        self.inner.segmentation_mut().install_engine(engine);
        tracing::info!(
            target: "fluxframe::idle",
            elapsed_ms = start.elapsed().as_millis() as u64,
            "managed_composite: engine reloaded"
        );
        self.engine_ready.store(true, Ordering::Release);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VideoEffect — straight delegation to the inner composite.
//
// The trait surface is small enough to forward each call by hand.
// The only departure from a pure passthrough is `process`: if a
// reload is in flight (`engine_ready = false`) we still let the call
// reach the inner composite, which will return a Fatal — the
// supervisor's worker loop gates on `is_ready()` before invoking
// `process` so this path should never fire in production. We keep
// the delegation honest rather than silently turning Fatal into Ok.
// ---------------------------------------------------------------------------

impl VideoEffect for ManagedComposite {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        self.inner.configure(params)
    }

    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError> {
        self.inner.prepare(context)?;
        // `prepare` builds a fresh engine internally — mark ready.
        self.engine_ready.store(true, Ordering::Release);
        debug_assert!(
            self.inner.segmentation().engine_loaded(),
            "CompositeEffect::prepare returned Ok but no engine was loaded",
        );
        Ok(())
    }

    fn process(
        &mut self,
        frame: &mut VideoFrame,
        context: &mut FrameContext,
    ) -> Result<(), EffectError> {
        self.inner.process(frame, context)
    }

    fn shutdown(&mut self) -> Result<(), EffectError> {
        self.inner.shutdown()
    }

    fn reconfigure_named_effect(
        &mut self,
        section: SubchainKind,
        name: &str,
        params: RawEffectParams,
    ) -> Result<(), EffectError> {
        self.inner.reconfigure_named_effect(section, name, params)
    }
}

// `AsAnyMut` is covered by the blanket `impl<T: Any> AsAnyMut for T`
// in fluxframe-core; no per-type impl required.

#[cfg(test)]
mod tests {
    //! Testing-coverage note (Stage 15 Step 2 review, Fix #9):
    //!
    //! The unit tests in this module exercise the wrapper's
    //! lifecycle *primitives*:
    //!
    //! * [`super::ManagedComposite::unload`] (drop semantics, idempotence),
    //! * [`super::ManagedComposite::engine_ready_handle`] (flag aliasing),
    //! * the early-return arm of
    //!   [`super::ManagedComposite::reload_engine`] when
    //!   `prepare()` has not populated `model_config`.
    //!
    //! What is *not* covered here: the success path through
    //! `reload_engine` reaches
    //! [`fluxframe_effects::backend::build_inference_engine`]
    //! directly (rather than the `SegmentationBase` FnOnce
    //! factory), so faking the factory yields no observable
    //! signal. Exercising the success path end-to-end requires
    //! either a real ONNX model on disk or a sidecar mock
    //! `build_inference_engine` shim. That coverage will land in
    //! Stage 15 Step 4 as an integration test once the supervisor
    //! wires the reload thread with a model path the test harness
    //! can supply.
    use super::*;
    use fluxframe_core::error::InferenceError;
    use fluxframe_core::frame::PixelFormat;
    use fluxframe_core::traits::{InferenceEngine, InferenceInput, InferenceOutput, ModelInfo};
    use fluxframe_effects::composite::segmentation::{SegmentationBase, SegmentationConfig};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Mock inference engine that counts how many times it has
    /// served `infer` calls. Used by the lifecycle tests below to
    /// stand in for a real ONNX engine without requiring a model
    /// on disk.
    struct MockEngine {
        infer_calls: Arc<AtomicU32>,
    }

    impl InferenceEngine for MockEngine {
        fn model_info(&self) -> ModelInfo {
            ModelInfo {
                name: "mock".into(),
                input_width: 1,
                input_height: 1,
                input_format: PixelFormat::Rgb,
            }
        }

        fn infer(&mut self, _input: InferenceInput<'_>) -> Result<InferenceOutput, InferenceError> {
            self.infer_calls.fetch_add(1, Ordering::Relaxed);
            Ok(InferenceOutput {
                data: vec![0.5],
                shape: vec![1, 1, 1, 1],
            })
        }
    }

    /// `unload` drops the engine and flips the readiness flag.
    /// `install_engine` (via direct call here; reload uses
    /// `build_inference_engine` which would touch the disk) restores
    /// both. This proves the wrapper's primitive: the supervisor's
    /// reload thread does the same dance with the real builder.
    #[test]
    fn unload_then_install_round_trips_through_the_segmentation_base() {
        let infer_calls = Arc::new(AtomicU32::new(0));
        let mock = Box::new(MockEngine {
            infer_calls: Arc::clone(&infer_calls),
        }) as Box<dyn InferenceEngine + Send>;
        let cfg = SegmentationConfig {
            model: PathBuf::from("/tmp/test_model.onnx"),
            model_config: None,
            fallback_threshold: 3,
        };
        let mut base = SegmentationBase::new(cfg);
        base.install_engine(mock);

        // Construct a minimal CompositeEffect — empty sub-chains
        // suffice because we are only exercising lifecycle, not
        // process().
        let composite = CompositeEffect::new(base, vec![], vec![], vec![], vec![]);
        let mut managed = ManagedComposite::new(composite);
        assert!(managed.is_ready(), "wrapped composite starts ready");

        managed.unload();
        assert!(!managed.is_ready(), "engine_ready must drop after unload");
        assert!(
            !managed.inner.segmentation().engine_loaded(),
            "engine slot must be empty after unload"
        );

        let mock2 = Box::new(MockEngine {
            infer_calls: Arc::clone(&infer_calls),
        }) as Box<dyn InferenceEngine + Send>;
        managed.inner.segmentation_mut().install_engine(mock2);
        managed.engine_ready.store(true, Ordering::Release);
        assert!(
            managed.is_ready(),
            "wrapper recovers readiness after install_engine + flag set"
        );
    }

    /// `set_counters` accepts an `Arc<Counters>` snapshot. The
    /// supervisor calls this immediately after `prepare()` so the
    /// reload path can pass the same counters to a freshly built
    /// engine.
    #[test]
    fn set_counters_stores_handle() {
        let cfg = SegmentationConfig {
            model: PathBuf::from("/tmp/test_model.onnx"),
            model_config: None,
            fallback_threshold: 3,
        };
        let base = SegmentationBase::new(cfg);
        let composite = CompositeEffect::new(base, vec![], vec![], vec![], vec![]);
        let mut managed = ManagedComposite::new(composite);
        let counters = Arc::new(Counters::default());
        managed.set_counters(counters);
        assert!(managed.counters.is_some());
    }

    /// `unload` is idempotent — calling it twice does not panic
    /// and leaves the wrapper in the same state.
    #[test]
    fn unload_is_idempotent() {
        let cfg = SegmentationConfig {
            model: PathBuf::from("/tmp/test_model.onnx"),
            model_config: None,
            fallback_threshold: 3,
        };
        let base = SegmentationBase::new(cfg);
        let composite = CompositeEffect::new(base, vec![], vec![], vec![], vec![]);
        let mut managed = ManagedComposite::new(composite);
        managed.unload();
        managed.unload();
        assert!(!managed.is_ready());
    }

    /// `engine_ready_handle` produces a clone that observes the
    /// same readiness state as the wrapper. The supervisor uses
    /// this to plumb the flag into the worker without holding a
    /// reference to the whole wrapper.
    #[test]
    fn engine_ready_handle_aliases_internal_flag() {
        let cfg = SegmentationConfig {
            model: PathBuf::from("/tmp/test_model.onnx"),
            model_config: None,
            fallback_threshold: 3,
        };
        let base = SegmentationBase::new(cfg);
        let composite = CompositeEffect::new(base, vec![], vec![], vec![], vec![]);
        let mut managed = ManagedComposite::new(composite);
        let handle = managed.engine_ready_handle();
        assert!(handle.load(Ordering::Acquire));
        managed.unload();
        assert!(!handle.load(Ordering::Acquire));
    }

    /// `reload_engine` without a prior `prepare()` returns
    /// PrepareFailed — model_config is the rebuild key and is
    /// only populated by prepare.
    #[test]
    fn reload_without_prepare_returns_prepare_failed() {
        let cfg = SegmentationConfig {
            model: PathBuf::from("/tmp/test_model.onnx"),
            model_config: None,
            fallback_threshold: 3,
        };
        let base = SegmentationBase::new(cfg);
        let composite = CompositeEffect::new(base, vec![], vec![], vec![], vec![]);
        let mut managed = ManagedComposite::new(composite);
        let err = managed
            .reload_engine()
            .expect_err("reload before prepare must fail");
        assert!(matches!(err, EffectError::PrepareFailed { .. }));
    }
}
