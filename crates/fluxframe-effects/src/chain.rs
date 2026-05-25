//! Sequential effect chain runtime.
//!
//! Stage 0 implements the happy path and a single fallback policy (`Exit`
//! — propagate the first error).  Stage 5 (realtime hardening) extends
//! this with the full policy set from §26.

use std::collections::BTreeMap;

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::frame::VideoFrame;
use fluxframe_core::traits::{RawEffectParams, VideoEffect};

/// Ordered list of effects executed sequentially per frame.
pub struct EffectChain {
    effects: Vec<Box<dyn VideoEffect>>,
}

impl EffectChain {
    /// Build a chain from an ordered list of effects.
    #[must_use]
    pub fn new(effects: Vec<Box<dyn VideoEffect>>) -> Self {
        Self { effects }
    }

    /// Returns `true` if the chain contains no effects.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// Number of effects in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    /// Names of the effects in chain order — used by `tracing` and the
    /// `check` command.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.effects.iter().map(|e| e.name()).collect()
    }

    /// Consume the chain and return its underlying boxed effects.
    ///
    /// Useful for callers that need to merge a registry-built chain
    /// with externally-constructed effects (e.g. the CLI prepending a
    /// composite effect to the user-configured chain) without rebuilding
    /// each effect through the registry a second time.
    #[must_use]
    pub fn into_effects(self) -> Vec<Box<dyn VideoEffect>> {
        self.effects
    }

    /// Run `configure` on every effect, passing the matching per-effect
    /// TOML table from `per_effect_params` (or an empty table when the
    /// effect has no entry).  Must be called before [`Self::prepare_all`];
    /// effects that require configuration surface a structured
    /// `EffectError::InvalidConfig` here rather than the more cryptic
    /// "prepare called before configure" from downstream lifecycle stages.
    /// (Note: `CompositeEffect` is pre-configured by its builder, not via
    /// `configure_all`.)
    ///
    /// # Errors
    ///
    /// Returns the first [`EffectError`] produced by `configure`;
    /// subsequent effects are not configured.
    pub fn configure_all(
        &mut self,
        per_effect_params: &BTreeMap<String, toml::Value>,
    ) -> Result<(), EffectError> {
        for effect in &mut self.effects {
            let params: RawEffectParams = per_effect_params
                .get(effect.name())
                .cloned()
                .unwrap_or_else(|| toml::Value::Table(toml::Table::new()));
            effect.configure(params)?;
        }
        Ok(())
    }

    /// Run `prepare` on every effect in order.
    ///
    /// # Errors
    ///
    /// Returns the first [`EffectError`] produced by `prepare`; subsequent
    /// effects are not prepared so the runtime can abort start-up cleanly.
    pub fn prepare_all(&mut self, context: &ProcessingContext) -> Result<(), EffectError> {
        for effect in &mut self.effects {
            effect.prepare(context)?;
        }
        Ok(())
    }

    /// Process a single frame through the chain.
    ///
    /// # Errors
    ///
    /// Returns the first [`EffectError`] raised by an effect; downstream
    /// effects are skipped (Stage 0 `Exit` policy from §26).
    pub fn process(
        &mut self,
        frame: &mut VideoFrame,
        context: &mut FrameContext,
    ) -> Result<(), EffectError> {
        for effect in &mut self.effects {
            effect.process(frame, context)?;
        }
        Ok(())
    }

    /// Best-effort shutdown.  Collects every error and returns the first,
    /// but always calls `shutdown` on every effect so partial cleanup is
    /// not skipped.
    ///
    /// # Errors
    ///
    /// Returns the first [`EffectError`] observed during shutdown.  All
    /// effects still receive a `shutdown` call regardless of earlier
    /// failures.
    pub fn shutdown_all(&mut self) -> Result<(), EffectError> {
        let mut first_err: Option<EffectError> = None;
        for effect in &mut self.effects {
            if let Err(e) = effect.shutdown() {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxframe_core::frame::{FrameBuffer, FrameMeta, PixelFormat};
    use fluxframe_core::traits::RawEffectParams;
    use std::sync::{Arc, Mutex};

    /// Test double that records every lifecycle call into a shared log
    /// and can optionally fail on `process` or `shutdown`.
    struct RecordingEffect {
        label: &'static str,
        log: Arc<Mutex<Vec<&'static str>>>,
        fail_on_process: bool,
        fail_on_shutdown: bool,
    }

    impl RecordingEffect {
        fn new(label: &'static str, log: Arc<Mutex<Vec<&'static str>>>) -> Self {
            Self {
                label,
                log,
                fail_on_process: false,
                fail_on_shutdown: false,
            }
        }
    }

    impl VideoEffect for RecordingEffect {
        fn name(&self) -> &'static str {
            self.label
        }

        fn configure(&mut self, _params: RawEffectParams) -> Result<(), EffectError> {
            Ok(())
        }

        fn prepare(&mut self, _context: &ProcessingContext) -> Result<(), EffectError> {
            self.log.lock().unwrap().push(self.label);
            Ok(())
        }

        fn process(
            &mut self,
            _frame: &mut VideoFrame,
            _context: &mut FrameContext,
        ) -> Result<(), EffectError> {
            self.log.lock().unwrap().push(self.label);
            if self.fail_on_process {
                return Err(EffectError::ProcessFailed {
                    name: self.label.to_string(),
                    reason: "forced".to_string(),
                });
            }
            Ok(())
        }

        fn shutdown(&mut self) -> Result<(), EffectError> {
            self.log.lock().unwrap().push(self.label);
            if self.fail_on_shutdown {
                return Err(EffectError::ProcessFailed {
                    name: self.label.to_string(),
                    reason: "shutdown forced".to_string(),
                });
            }
            Ok(())
        }
    }

    fn make_frame() -> VideoFrame {
        VideoFrame::new_packed(
            FrameBuffer::Owned(vec![0u8; 4 * 4 * 3]),
            4,
            4,
            PixelFormat::Rgb,
            FrameMeta::default(),
        )
        .expect("packed RGB frame builds")
    }

    fn make_ctx() -> ProcessingContext {
        ProcessingContext {
            width: 4,
            height: 4,
            format: PixelFormat::Rgb,
            fps: 30,
            counters: None,
        }
    }

    #[test]
    fn process_runs_effects_in_chain_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let chain_effects: Vec<Box<dyn VideoEffect>> = vec![
            Box::new(RecordingEffect::new("A", Arc::clone(&log))),
            Box::new(RecordingEffect::new("B", Arc::clone(&log))),
            Box::new(RecordingEffect::new("C", Arc::clone(&log))),
        ];
        let mut chain = EffectChain::new(chain_effects);

        let mut frame = make_frame();
        let mut frame_context = FrameContext::default();
        chain
            .process(&mut frame, &mut frame_context)
            .expect("process ok");

        assert_eq!(*log.lock().unwrap(), vec!["A", "B", "C"]);
    }

    #[test]
    fn process_aborts_on_first_error() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut failing_b = RecordingEffect::new("B", Arc::clone(&log));
        failing_b.fail_on_process = true;
        let chain_effects: Vec<Box<dyn VideoEffect>> = vec![
            Box::new(RecordingEffect::new("A", Arc::clone(&log))),
            Box::new(failing_b),
            Box::new(RecordingEffect::new("C", Arc::clone(&log))),
        ];
        let mut chain = EffectChain::new(chain_effects);

        let mut frame = make_frame();
        let mut frame_context = FrameContext::default();
        let res = chain.process(&mut frame, &mut frame_context);

        assert!(res.is_err(), "expected error propagation");
        assert_eq!(*log.lock().unwrap(), vec!["A", "B"]);
    }

    #[test]
    fn shutdown_calls_every_effect_even_on_error() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut failing_b = RecordingEffect::new("B", Arc::clone(&log));
        failing_b.fail_on_shutdown = true;
        let chain_effects: Vec<Box<dyn VideoEffect>> = vec![
            Box::new(RecordingEffect::new("A", Arc::clone(&log))),
            Box::new(failing_b),
            Box::new(RecordingEffect::new("C", Arc::clone(&log))),
        ];
        let mut chain = EffectChain::new(chain_effects);

        let res = chain.shutdown_all();

        assert_eq!(*log.lock().unwrap(), vec!["A", "B", "C"]);
        match res {
            Err(EffectError::ProcessFailed { name, .. }) => assert_eq!(name, "B"),
            other => panic!("expected ProcessFailed from B, got {other:?}"),
        }
    }

    #[test]
    fn prepare_runs_all_in_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let chain_effects: Vec<Box<dyn VideoEffect>> = vec![
            Box::new(RecordingEffect::new("A", Arc::clone(&log))),
            Box::new(RecordingEffect::new("B", Arc::clone(&log))),
            Box::new(RecordingEffect::new("C", Arc::clone(&log))),
        ];
        let mut chain = EffectChain::new(chain_effects);

        chain.prepare_all(&make_ctx()).expect("prepare ok");
        assert_eq!(*log.lock().unwrap(), vec!["A", "B", "C"]);
    }

    #[test]
    fn empty_chain_is_noop() {
        let mut chain = EffectChain::new(Vec::new());
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(chain.names().is_empty());

        chain.prepare_all(&make_ctx()).expect("prepare ok");
        let mut frame = make_frame();
        let mut frame_context = FrameContext::default();
        chain
            .process(&mut frame, &mut frame_context)
            .expect("process ok");
        chain.shutdown_all().expect("shutdown ok");
    }

    #[test]
    fn names_returns_chain_in_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let chain_effects: Vec<Box<dyn VideoEffect>> = vec![
            Box::new(RecordingEffect::new("A", Arc::clone(&log))),
            Box::new(RecordingEffect::new("B", Arc::clone(&log))),
            Box::new(RecordingEffect::new("C", Arc::clone(&log))),
        ];
        let chain = EffectChain::new(chain_effects);
        assert_eq!(chain.names(), vec!["A", "B", "C"]);
    }
}
