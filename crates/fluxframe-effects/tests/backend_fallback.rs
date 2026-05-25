//! Integration tests for the Stage 6 backend abstraction layer.
//!
//! Lives outside the `fluxframe-effects` crate as a normal `tests/`
//! integration target so the test exercises only the **public** API
//! of the backend module — exactly what a future GPU implementation
//! will rely on.  No internal helpers, no `#[cfg(test)]`-only types.
//!
//! Scenarios:
//!
//! 1. `StickyBlurFallback` transitions from a primary that errors to a
//!    secondary that succeeds; the counter increments exactly once and
//!    subsequent calls go straight to the secondary.
//! 2. `StickyInferenceFallback` does the same for the inference path.
//! 3. `build_blur_backend` returns a CPU backend whose `name() == "cpu"`
//!    for both `Auto` and explicit `Cpu` overrides (the only options
//!    available in Stage 6 — Stage 7 adds a GPU candidate to this
//!    factory).
//!
//! The whole file is gated on `feature = "ml"` because the inference
//! scenario depends on `StickyInferenceFallback` which is itself
//! ml-gated.  Splitting blur and inference into two files would
//! double the boilerplate for marginal value; if a future slim build
//! consumer needs the blur scenario without ml, the file can be split.

#![cfg(feature = "ml")]

use std::cell::Cell;
use std::sync::Arc;

use fluxframe_core::error::{EffectError, InferenceError};
use fluxframe_core::frame::PixelFormat;
use fluxframe_core::metrics::Counters;
use fluxframe_core::traits::{InferenceEngine, InferenceInput, InferenceOutput, ModelInfo};
use fluxframe_effects::backend::{
    BackendOverrides, BlurBackend, BlurBackendChoice, CpuBlurBackend, StickyBlurFallback,
    StickyInferenceFallback, build_blur_backend,
};

// -----------------------------------------------------------------
// Mock backends used by the integration scenarios.  They live here
// (not in `fluxframe-effects::backend::tests`) precisely so this file
// drives the public trait surface — if these compile against the
// public API, a real GPU backend will too.
// -----------------------------------------------------------------

/// `BlurBackend` mock that fails on the first call, succeeds afterwards.
struct PrimaryBlur {
    calls: Cell<u32>,
}

impl PrimaryBlur {
    fn new() -> Self {
        Self {
            calls: Cell::new(0),
        }
    }
}

impl BlurBackend for PrimaryBlur {
    fn name(&self) -> &'static str {
        "primary-blur-mock"
    }
    fn prepare(&mut self, _w: u32, _h: u32) -> Result<(), EffectError> {
        Ok(())
    }
    fn blur(
        &mut self,
        _src: &[u8],
        dst: &mut [u8],
        _w: u32,
        _h: u32,
        _r: u32,
        _p: u32,
    ) -> Result<(), EffectError> {
        let n = self.calls.get();
        self.calls.set(n + 1);
        if n == 0 {
            Err(EffectError::ProcessFailed {
                name: "primary-blur-mock".into(),
                reason: "synthetic first-call failure".into(),
            })
        } else {
            // Mark output so a test can distinguish from secondary.
            dst.fill(0x11);
            Ok(())
        }
    }
}

/// `BlurBackend` mock that always succeeds with a known sentinel byte.
struct SecondaryBlur;
impl BlurBackend for SecondaryBlur {
    fn name(&self) -> &'static str {
        "secondary-blur-mock"
    }
    fn prepare(&mut self, _w: u32, _h: u32) -> Result<(), EffectError> {
        Ok(())
    }
    fn blur(
        &mut self,
        _src: &[u8],
        dst: &mut [u8],
        _w: u32,
        _h: u32,
        _r: u32,
        _p: u32,
    ) -> Result<(), EffectError> {
        dst.fill(0x99);
        Ok(())
    }
}

/// `InferenceEngine` mock that fails on the first call, succeeds afterwards.
struct PrimaryEngine {
    calls: Cell<u32>,
}

impl PrimaryEngine {
    fn new() -> Self {
        Self {
            calls: Cell::new(0),
        }
    }
}

impl InferenceEngine for PrimaryEngine {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: Arc::from("primary"),
            input_width: 1,
            input_height: 1,
            input_format: PixelFormat::Rgb,
        }
    }
    fn infer(&mut self, _input: InferenceInput<'_>) -> Result<InferenceOutput, InferenceError> {
        let n = self.calls.get();
        self.calls.set(n + 1);
        if n == 0 {
            Err(InferenceError::InferenceFailed {
                reason: "synthetic first-call failure".into(),
            })
        } else {
            Ok(InferenceOutput {
                data: vec![1.0],
                shape: vec![1],
            })
        }
    }
}

/// `InferenceEngine` mock that always succeeds with a known sentinel value.
struct SecondaryEngine;
impl InferenceEngine for SecondaryEngine {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: Arc::from("secondary"),
            input_width: 1,
            input_height: 1,
            input_format: PixelFormat::Rgb,
        }
    }
    fn infer(&mut self, _input: InferenceInput<'_>) -> Result<InferenceOutput, InferenceError> {
        Ok(InferenceOutput {
            data: vec![42.0],
            shape: vec![1],
        })
    }
}

// -----------------------------------------------------------------
// Scenarios.
// -----------------------------------------------------------------

#[test]
fn sticky_blur_fallback_transitions_through_public_api() {
    let counters = Arc::new(Counters::new());
    let mut decorator: Box<dyn BlurBackend + Send> = Box::new(StickyBlurFallback::new(
        Box::new(PrimaryBlur::new()),
        Box::new(SecondaryBlur),
        "primary-blur-mock",
        "secondary-blur-mock",
        Arc::clone(&counters),
    ));

    decorator.prepare(4, 4).expect("prepare propagates to both");

    let src = vec![0u8; 4 * 4 * 3];
    let mut dst = vec![0u8; 4 * 4 * 3];

    // First call: primary errors → transition → secondary writes 0x99.
    decorator
        .blur(&src, &mut dst, 4, 4, 1, 1)
        .expect("retry on secondary succeeds");
    assert!(
        dst.iter().all(|&b| b == 0x99),
        "secondary sentinel must be present after transition"
    );
    assert_eq!(decorator.name(), "secondary-blur-mock");
    assert_eq!(
        counters.snapshot().blur_runtime_fallback_gpu_to_cpu,
        1,
        "counter must increment exactly once on the transition",
    );

    // Subsequent calls: stay on secondary, counter does not grow.
    for _ in 0..5 {
        decorator.blur(&src, &mut dst, 4, 4, 1, 1).expect("ok");
    }
    assert_eq!(counters.snapshot().blur_runtime_fallback_gpu_to_cpu, 1);
}

#[test]
fn sticky_inference_fallback_transitions_through_public_api() {
    let counters = Arc::new(Counters::new());
    let mut decorator: Box<dyn InferenceEngine + Send> = Box::new(StickyInferenceFallback::new(
        Box::new(PrimaryEngine::new()),
        Box::new(SecondaryEngine),
        "gpu",
        "cpu",
        Arc::clone(&counters),
    ));

    let data = vec![0.0_f32];
    let shape = [1_usize];

    // First call: primary errors → transition → secondary returns 42.0.
    let out = decorator
        .infer(InferenceInput {
            data: &data,
            shape: &shape,
        })
        .expect("retry on secondary succeeds");
    assert_eq!(out.data, vec![42.0], "secondary output expected");
    assert_eq!(
        counters.snapshot().inference_runtime_fallback_gpu_to_cpu,
        1,
        "counter must increment exactly once on the transition",
    );

    // Subsequent calls: stay on secondary, counter stays at 1.
    for _ in 0..5 {
        decorator
            .infer(InferenceInput {
                data: &data,
                shape: &shape,
            })
            .expect("ok");
    }
    assert_eq!(counters.snapshot().inference_runtime_fallback_gpu_to_cpu, 1);
}

#[test]
fn build_blur_backend_auto_resolves_to_cpu() {
    // Stage 7 hotfix: `Auto` deliberately picks CPU because the
    // initial wgpu implementation lost to the CPU box-blur on the
    // production pipeline (640×480, blur_downscale=4).  wgpu is
    // reachable only through the explicit `Wgpu` override (or env
    // knob `FLUXFRAME_FORCE_BLUR_BACKEND=wgpu`).
    let backend = build_blur_backend(BackendOverrides::default(), None).expect("factory ok");
    assert_eq!(backend.name(), "cpu");
}

#[test]
fn build_blur_backend_returns_cpu_under_explicit_cpu_override() {
    let overrides = BackendOverrides::default().with_blur(BlurBackendChoice::Cpu);
    let backend = build_blur_backend(overrides, None).expect("factory infallible");
    assert_eq!(backend.name(), "cpu");
}

#[test]
fn build_blur_backend_accepts_counters_with_cpu_override() {
    // Even with counters supplied, an explicit `Cpu` override must
    // return the bare CPU backend (no decorator wraps a
    // single-candidate result).
    let counters = Arc::new(Counters::new());
    let overrides = BackendOverrides::default().with_blur(BlurBackendChoice::Cpu);
    let backend = build_blur_backend(overrides, Some(counters)).expect("factory infallible");
    assert_eq!(backend.name(), "cpu");
}

#[test]
fn cpu_blur_backend_implements_blur_backend_publicly() {
    // Sanity smoke: the trait-object form `Box<dyn BlurBackend + Send>`
    // — the form factories return — works against the CPU backend
    // through the public re-export, no crate-internal types involved.
    let mut backend: Box<dyn BlurBackend + Send> = Box::new(CpuBlurBackend::new());
    backend.prepare(4, 4).expect("prepare");
    let src = vec![0u8; 4 * 4 * 3];
    let mut dst = vec![0u8; 4 * 4 * 3];
    backend.blur(&src, &mut dst, 4, 4, 1, 1).expect("blur");
}
