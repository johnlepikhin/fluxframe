//! Blur-backend factory + sticky-fallback decorator.
//!
//! [`build_blur_backend`] is the single point where the effect picks a
//! [`BlurBackend`] implementation.  Stage 7 introduces a `wgpu`
//! candidate behind the `wgpu` Cargo feature; on the current
//! pipeline it is **not** preferred by `Auto` because Stage 7
//! measurement showed the GPU implementation losing to the CPU
//! box-blur on the production workload (see
//! [`build_blur_backend`] docstring and
//! `doc/plan/stage-7-wgpu-blur.md`).  The wgpu path stays reachable
//! through the explicit `BlurBackendChoice::Wgpu` override (env
//! `FLUXFRAME_FORCE_BLUR_BACKEND=wgpu`).  When that branch is
//! taken with runtime-supplied counters the result is wrapped in
//! [`StickyBlurFallback`] with [`CpuBlurBackend`] as the secondary
//! so any per-frame error permanently demotes the chain to CPU
//! within the same session.
//!
//! The factory does **not** call `prepare()` on the returned backend
//! — that stays the caller's responsibility (typically inside
//! `BackgroundBlurEffect::prepare`), so probe failures and effect
//! setup failures surface at the same boundary.

use std::sync::Arc;

use fluxframe_core::error::EffectError;
use fluxframe_core::metrics::Counters;
use tracing::{info, warn};

use crate::backend::blur::{BlurBackend, CpuBlurBackend};
use crate::backend::overrides::{BackendOverrides, BlurBackendChoice};
#[cfg(feature = "wgpu")]
use crate::backend::wgpu_blur::WgpuBlurBackend;

/// Build a blur backend honouring the supplied overrides.
///
/// `counters` is the supervisor's shared counter bundle, threaded
/// through `ProcessingContext::counters` by the runtime.  When
/// `Some`, the factory wraps multi-candidate results in a
/// [`StickyBlurFallback`] that publishes transition events to
/// [`Counters::inc_blur_runtime_fallback_gpu_to_cpu`].  When `None`
/// (test paths, standalone effect-chain runs) the factory returns
/// the primary backend bare, without the decorator.
///
/// Selection logic:
///
/// * `BlurBackendChoice::Cpu` — return [`CpuBlurBackend`] directly.
/// * `BlurBackendChoice::Wgpu` — probe [`WgpuBlurBackend`]; on probe
///   failure return a hard error so the diagnostic surfaces (explicit
///   override never silently demotes).
/// * `BlurBackendChoice::Auto` — return [`CpuBlurBackend`].  See the
///   "Auto behaviour" note below for why GPU is intentionally NOT
///   probed in this branch on the current pipeline shape.
///
/// The selection is logged at `info!` exactly once.
///
/// # Auto behaviour
///
/// `Auto` deliberately picks CPU on the current
/// `BackgroundBlurEffect` pipeline.  Stage 7's first wgpu
/// implementation was measured against the CPU `box_blur_rgb` on
/// the production workload (640×480 frame, `blur_downscale=4` →
/// 160×120 GPU work area) and lost to it by `processing_p95` +50 %,
/// fps -20 %, with measurable inference-stage contention from the
/// shared iGPU.  The seam, sticky-fallback decorator and Vulkan
/// probe are still wired and reachable via
/// `FLUXFRAME_FORCE_BLUR_BACKEND=wgpu` — that is the path operators
/// or future contributors take when they have a workload (bigger
/// frames, different effects, async readback) where the GPU
/// candidate is competitive.  See
/// `doc/plan/stage-7-wgpu-blur.md` "Acceptance / Auto behaviour
/// finding" for the full measurement and upgrade-path discussion.
///
/// # Errors
///
/// * [`EffectError::PrepareFailed`] when `Wgpu` is forced and probe
///   fails, or when the `wgpu` Cargo feature is disabled at compile
///   time.
pub fn build_blur_backend(
    overrides: BackendOverrides,
    counters: Option<Arc<Counters>>,
) -> Result<Box<dyn BlurBackend + Send>, EffectError> {
    let choice = overrides.blur;
    let backend: Box<dyn BlurBackend + Send> = match choice {
        BlurBackendChoice::Cpu => Box::new(CpuBlurBackend::new()),
        BlurBackendChoice::Wgpu => build_wgpu_forced(counters)?,
        BlurBackendChoice::Auto => build_auto(counters),
    };
    info!(
        component = "blur",
        backend = backend.name(),
        override_ = ?choice,
        "selected blur backend",
    );
    Ok(backend)
}

/// Forced-wgpu branch: probe must succeed.  Failure is a hard error.
#[cfg(feature = "wgpu")]
fn build_wgpu_forced(
    counters: Option<Arc<Counters>>,
) -> Result<Box<dyn BlurBackend + Send>, EffectError> {
    let primary = WgpuBlurBackend::probe()?;
    Ok(wrap_with_fallback(primary, counters))
}

#[cfg(not(feature = "wgpu"))]
fn build_wgpu_forced(
    _counters: Option<Arc<Counters>>,
) -> Result<Box<dyn BlurBackend + Send>, EffectError> {
    Err(EffectError::PrepareFailed {
        name: "blur backend factory".into(),
        reason: "FLUXFRAME_FORCE_BLUR_BACKEND=wgpu but the `wgpu` Cargo feature is disabled \
                 in this build"
            .into(),
    })
}

/// Auto-detect branch.
///
/// **Currently always returns CPU.**  Stage 7 measurement on the
/// production pipeline (640×480 with `blur_downscale=4`) showed the
/// wgpu primary lost to the CPU box-blur — see the function-level
/// docstring and `doc/plan/stage-7-wgpu-blur.md` for numbers.  We
/// keep the wgpu candidate reachable through the explicit
/// `BlurBackendChoice::Wgpu` override and the
/// `FLUXFRAME_FORCE_BLUR_BACKEND=wgpu` env knob so the seam stays
/// exercised and a future workload that suits the GPU can flip the
/// auto-detect heuristic without another signature change.
///
/// `counters` is ignored in this branch because there is no
/// multi-candidate situation: a bare CPU backend has nothing to
/// "fall back to".
fn build_auto(_counters: Option<Arc<Counters>>) -> Box<dyn BlurBackend + Send> {
    Box::new(CpuBlurBackend::new())
}

/// Wrap the GPU primary with a sticky-fallback decorator when the
/// runtime supplies counters; otherwise return the primary bare
/// (test path).  Lives behind `cfg(feature = "wgpu")` since
/// non-wgpu builds never reach a multi-candidate state.
#[cfg(feature = "wgpu")]
fn wrap_with_fallback(
    primary: WgpuBlurBackend,
    counters: Option<Arc<Counters>>,
) -> Box<dyn BlurBackend + Send> {
    match counters {
        Some(c) => Box::new(StickyBlurFallback::new(
            Box::new(primary),
            Box::new(CpuBlurBackend::new()),
            "wgpu",
            "cpu",
            c,
        )),
        None => Box::new(primary),
    }
}

/// Sticky one-way fallback decorator for [`BlurBackend`].
///
/// Wraps a primary (GPU-style) backend and an eagerly-constructed
/// secondary (CPU).  On the **first** error from the primary, the
/// decorator:
///
/// 1. emits a single `warn!` with the original error and the names
///    of both backends, so the operator can correlate the transition
///    with the upstream failure;
/// 2. increments [`Counters::inc_blur_runtime_fallback_gpu_to_cpu`];
/// 3. drops the primary (freeing whatever GPU resources it held);
/// 4. retries the operation against the secondary in the same call,
///    so the caller never sees a dropped frame purely because of the
///    transition.
///
/// All subsequent calls go straight to the secondary.  The transition
/// is one-way by design (per Stage 6 plan): a GPU backend that fails
/// once on a desktop driver is far more likely to keep failing than
/// to recover, and the sticky behaviour avoids per-frame log flap.
///
/// This decorator is **not** used in Stage 6 production paths (there
/// is no primary candidate yet); it is wired now so the seam is
/// fully testable.  The integration test
/// `tests/backend_fallback.rs` exercises the transition on mocks.
pub struct StickyBlurFallback {
    state: BlurState,
    primary_name: &'static str,
    secondary_name: &'static str,
    counters: Arc<Counters>,
}

/// Internal state machine for [`StickyBlurFallback`].  `Poisoned` is
/// a transient marker held only while the transition is in flight;
/// it must not be observable to a caller outside that path.
enum BlurState {
    Primary {
        primary: Box<dyn BlurBackend + Send>,
        secondary: Box<dyn BlurBackend + Send>,
    },
    Secondary {
        secondary: Box<dyn BlurBackend + Send>,
    },
    Poisoned,
}

impl StickyBlurFallback {
    /// Construct a decorator.  `primary_name`/`secondary_name` are
    /// the values surfaced in the transition `warn!`.  Pass static
    /// strings (`"wgpu"`, `"cpu"`, …) — they are not derived from the
    /// backends' `name()` because that method is `&self`-bound and
    /// would force borrowing across the transition.
    #[must_use]
    pub fn new(
        primary: Box<dyn BlurBackend + Send>,
        secondary: Box<dyn BlurBackend + Send>,
        primary_name: &'static str,
        secondary_name: &'static str,
        counters: Arc<Counters>,
    ) -> Self {
        Self {
            state: BlurState::Primary { primary, secondary },
            primary_name,
            secondary_name,
            counters,
        }
    }

    /// `true` iff the primary has already failed and the decorator
    /// is permanently running on the secondary.  Useful for tests and
    /// for ad-hoc diagnostics; the production hot path does not call
    /// it.
    #[must_use]
    pub fn has_fallen_back(&self) -> bool {
        matches!(self.state, BlurState::Secondary { .. })
    }
}

impl BlurBackend for StickyBlurFallback {
    fn name(&self) -> &'static str {
        match &self.state {
            BlurState::Primary { primary, .. } => primary.name(),
            BlurState::Secondary { secondary } => secondary.name(),
            BlurState::Poisoned => "poisoned",
        }
    }

    fn prepare(&mut self, width: u32, height: u32) -> Result<(), EffectError> {
        match &mut self.state {
            // Prepare BOTH backends so the fallback path is ready
            // without an on-demand allocation when the primary fails.
            // If the secondary's prepare fails the whole effect setup
            // fails — that is the correct behaviour: a decorator that
            // cannot fall back is worse than no decorator.
            BlurState::Primary { primary, secondary } => {
                primary.prepare(width, height)?;
                secondary.prepare(width, height)?;
                Ok(())
            }
            BlurState::Secondary { secondary } => secondary.prepare(width, height),
            BlurState::Poisoned => Err(EffectError::PrepareFailed {
                name: "sticky blur fallback".into(),
                reason: "decorator in poisoned state (programming error)".into(),
            }),
        }
    }

    fn blur(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        height: u32,
        radius: u32,
        passes: u32,
    ) -> Result<(), EffectError> {
        match &mut self.state {
            BlurState::Secondary { secondary } => {
                secondary.blur(src, dst, width, height, radius, passes)
            }
            BlurState::Primary { primary, .. } => {
                // Try the primary first.
                match primary.blur(src, dst, width, height, radius, passes) {
                    Ok(()) => Ok(()),
                    Err(e) => {
                        warn!(
                            from = self.primary_name,
                            to = self.secondary_name,
                            error = %e,
                            "sticky blur backend fallback",
                        );
                        self.counters.inc_blur_runtime_fallback_gpu_to_cpu();
                        // Move out of Primary, drop primary, promote secondary.
                        let BlurState::Primary { secondary, .. } =
                            std::mem::replace(&mut self.state, BlurState::Poisoned)
                        else {
                            unreachable!("matched Primary above")
                        };
                        self.state = BlurState::Secondary { secondary };
                        // Retry on secondary — `dst` may have been
                        // partially written by the failing primary,
                        // but the secondary overwrites the entire
                        // buffer, so the partial write is harmless.
                        let BlurState::Secondary { secondary } = &mut self.state else {
                            unreachable!("just installed Secondary")
                        };
                        secondary.blur(src, dst, width, height, radius, passes)
                    }
                }
            }
            BlurState::Poisoned => Err(EffectError::ProcessFailed {
                name: "sticky blur fallback".into(),
                reason: "decorator in poisoned state (programming error)".into(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::overrides::BlurBackendChoice;
    use std::cell::Cell;

    /// Test-only blur backend that errors on the first `blur` call
    /// and would succeed on subsequent calls.  We use a `Cell` so the
    /// trait's `&mut self` signature stays clean — interior mutation
    /// keeps the mock readable.
    struct FailingBlur {
        prepared: Cell<bool>,
        calls: Cell<u32>,
        fail_first_n: u32,
    }

    impl FailingBlur {
        fn new(fail_first_n: u32) -> Self {
            Self {
                prepared: Cell::new(false),
                calls: Cell::new(0),
                fail_first_n,
            }
        }
    }

    impl BlurBackend for FailingBlur {
        fn name(&self) -> &'static str {
            "failing-mock"
        }

        fn prepare(&mut self, _w: u32, _h: u32) -> Result<(), EffectError> {
            self.prepared.set(true);
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
            if n < self.fail_first_n {
                Err(EffectError::ProcessFailed {
                    name: "failing-mock".into(),
                    reason: format!("synthetic failure call {n}"),
                })
            } else {
                // On success, write a sentinel byte so tests can
                // distinguish primary-output from secondary-output.
                dst.fill(0xAA);
                Ok(())
            }
        }
    }

    /// Test-only blur backend that always succeeds and writes a
    /// distinctive sentinel so tests can confirm WHICH backend
    /// produced the output buffer.
    struct SentinelBlur {
        sentinel: u8,
    }

    impl BlurBackend for SentinelBlur {
        fn name(&self) -> &'static str {
            "sentinel"
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
            dst.fill(self.sentinel);
            Ok(())
        }
    }

    #[test]
    fn factory_auto_resolves_to_cpu() {
        // Stage 7 hotfix: `Auto` deliberately picks CPU because the
        // initial wgpu implementation lost to the CPU box-blur on
        // the production pipeline.  wgpu is reachable only through
        // the explicit `Wgpu` override (or env knob).  See the
        // factory docstring + `doc/plan/stage-7-wgpu-blur.md`.
        let backend = build_blur_backend(BackendOverrides::default(), None).expect("factory ok");
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn factory_returns_cpu_for_explicit_cpu_override() {
        let overrides = BackendOverrides::default().with_blur(BlurBackendChoice::Cpu);
        let backend = build_blur_backend(overrides, None).expect("factory ok");
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn factory_cpu_override_ignores_counters() {
        // Explicit `Cpu` returns the bare CPU backend even when
        // counters are supplied — there is no primary to wrap, so
        // the sticky-fallback decorator is intentionally absent.
        let counters = Arc::new(Counters::new());
        let overrides = BackendOverrides::default().with_blur(BlurBackendChoice::Cpu);
        let backend = build_blur_backend(overrides, Some(counters)).expect("factory ok");
        assert_eq!(backend.name(), "cpu");
    }

    #[cfg(not(feature = "wgpu"))]
    #[test]
    fn factory_forced_wgpu_errors_without_feature() {
        // When the `wgpu` Cargo feature is disabled, an explicit
        // `Wgpu` override must surface as a hard error rather than
        // silently falling back to CPU.
        let overrides = BackendOverrides::default().with_blur(BlurBackendChoice::Wgpu);
        let Err(err) = build_blur_backend(overrides, None) else {
            panic!("must hard-error without wgpu feature");
        };
        let msg = format!("{err}").to_lowercase();
        assert!(msg.contains("wgpu") || msg.contains("feature"));
    }

    #[test]
    fn fallback_starts_on_primary_and_reports_primary_name() {
        let counters = Arc::new(Counters::new());
        let decorator = StickyBlurFallback::new(
            Box::new(FailingBlur::new(1)),
            Box::new(SentinelBlur { sentinel: 0x55 }),
            "failing-mock",
            "sentinel",
            Arc::clone(&counters),
        );
        assert!(!decorator.has_fallen_back());
        assert_eq!(decorator.name(), "failing-mock");
        assert_eq!(counters.snapshot().blur_runtime_fallback_gpu_to_cpu, 0);
    }

    #[test]
    fn fallback_transitions_on_first_error() {
        let counters = Arc::new(Counters::new());
        let mut decorator = StickyBlurFallback::new(
            Box::new(FailingBlur::new(1)),
            Box::new(SentinelBlur { sentinel: 0x55 }),
            "failing-mock",
            "sentinel",
            Arc::clone(&counters),
        );
        decorator.prepare(4, 4).expect("prepare both");
        let src = vec![0u8; 4 * 4 * 3];
        let mut dst = vec![0u8; 4 * 4 * 3];
        decorator
            .blur(&src, &mut dst, 4, 4, 1, 1)
            .expect("blur succeeds via secondary");
        // After fallback, the output came from the SENTINEL secondary.
        assert!(
            dst.iter().all(|&b| b == 0x55),
            "secondary sentinel must fill the buffer"
        );
        assert!(decorator.has_fallen_back());
        assert_eq!(decorator.name(), "sentinel");
        assert_eq!(
            counters.snapshot().blur_runtime_fallback_gpu_to_cpu,
            1,
            "counter must increment exactly once on transition"
        );
    }

    #[test]
    fn fallback_subsequent_calls_skip_primary() {
        // Primary fails N times; once we're on secondary, all later
        // calls must NOT touch primary, even though primary would
        // still fail (it would have failed all N times if asked).
        let counters = Arc::new(Counters::new());
        let mut decorator = StickyBlurFallback::new(
            Box::new(FailingBlur::new(100)), // would fail 100 times
            Box::new(SentinelBlur { sentinel: 0x55 }),
            "failing-mock",
            "sentinel",
            Arc::clone(&counters),
        );
        decorator.prepare(4, 4).expect("prepare ok");
        let src = vec![0u8; 4 * 4 * 3];
        let mut dst = vec![0u8; 4 * 4 * 3];
        for _ in 0..10 {
            decorator
                .blur(&src, &mut dst, 4, 4, 1, 1)
                .expect("all 10 succeed via secondary");
        }
        // Counter is incremented exactly once, not on every call.
        assert_eq!(counters.snapshot().blur_runtime_fallback_gpu_to_cpu, 1);
    }

    #[test]
    fn fallback_uses_primary_when_it_succeeds() {
        let counters = Arc::new(Counters::new());
        let mut decorator = StickyBlurFallback::new(
            Box::new(SentinelBlur { sentinel: 0x11 }), // primary
            Box::new(SentinelBlur { sentinel: 0x99 }), // secondary
            "primary-sentinel",
            "secondary-sentinel",
            Arc::clone(&counters),
        );
        decorator.prepare(4, 4).expect("prepare ok");
        let src = vec![0u8; 4 * 4 * 3];
        let mut dst = vec![0u8; 4 * 4 * 3];
        decorator.blur(&src, &mut dst, 4, 4, 1, 1).expect("ok");
        // Primary wrote its sentinel — no fallback.
        assert!(dst.iter().all(|&b| b == 0x11));
        assert!(!decorator.has_fallen_back());
        assert_eq!(counters.snapshot().blur_runtime_fallback_gpu_to_cpu, 0);
    }

    #[test]
    fn fallback_prepare_failure_of_secondary_propagates() {
        // If secondary fails prepare, the decorator must report an
        // error — a primary-only fallback isn't a useful contract.
        struct PrepareFailingBlur;
        impl BlurBackend for PrepareFailingBlur {
            fn name(&self) -> &'static str {
                "broken-secondary"
            }
            fn prepare(&mut self, _w: u32, _h: u32) -> Result<(), EffectError> {
                Err(EffectError::PrepareFailed {
                    name: "broken-secondary".into(),
                    reason: "synthetic".into(),
                })
            }
            fn blur(
                &mut self,
                _: &[u8],
                _: &mut [u8],
                _: u32,
                _: u32,
                _: u32,
                _: u32,
            ) -> Result<(), EffectError> {
                unreachable!()
            }
        }

        let counters = Arc::new(Counters::new());
        let mut decorator = StickyBlurFallback::new(
            Box::new(SentinelBlur { sentinel: 0x11 }),
            Box::new(PrepareFailingBlur),
            "primary",
            "broken-secondary",
            counters,
        );
        let err = decorator.prepare(4, 4).expect_err("must fail");
        assert!(format!("{err}").contains("synthetic"));
    }
}
