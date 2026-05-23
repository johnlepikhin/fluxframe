//! Per-run metrics bundle used by [`crate::runtime`].
//!
//! Owns the [`Counters`] and the per-stage [`LatencyHistogram`]s the
//! supervisor's hot loop populates.  Behind `Arc`s so the (future)
//! periodic CLI reporter (5.D.3) can hold its own handle without
//! blocking the writer.
//!
//! Stage coverage as of 5.D.3:
//! * `processing` — wall-clock around `EffectChain::process`.
//! * `output`     — wall-clock around `OutputPipeline::push_frame`.
//! * `end_to_end` — supervisor-side latency: `recv` → post-`push`.
//! * `inference`  — written by ML effects via
//!   [`fluxframe_core::EffectTelemetry`] on
//!   [`fluxframe_core::FrameContext::telemetry`].  The supervisor wires
//!   the bundle's `inference` histogram into the per-frame context
//!   before calling [`fluxframe_effects::EffectChain::process`].
//!
//! Not wired yet:
//! * `capture` — needs the GStreamer pipeline clock to subtract from
//!   `frame.meta.timestamp` (a monotonic nanos value, not a
//!   wall-clock `Instant`).  Delegated to a future task.
//!
//! `frames_dropped` is synced from the input slot's running counter
//! (see [`RuntimeMetrics::sync_dropped_from_slot`]) rather than
//! recorded per-event: the slot owns the source of truth and we
//! reconcile in the supervisor loop.

use std::sync::Arc;

use fluxframe_core::metrics::{
    Counters, EffectTelemetry, LatencyHistogram, MetricsSnapshot,
};

/// Number of samples retained per per-stage histogram.  Sized for the
/// periodic reporter cadence (≈5 s) and a 30 fps producer: 1024
/// samples covers ~34 s, plenty of headroom against bursty reporter
/// ticks.  Capacity is fixed at construction; runtime tuning would
/// require a config plumb-through that nothing yet consumes.
const PER_STAGE_HISTOGRAM_CAPACITY: usize = 1024;

/// Metrics bundle owned by a single [`crate::runtime`] run.
///
/// `Clone` is a cheap `Arc` bump — the supervisor, the periodic
/// reporter thread, and any effect writing through
/// [`EffectTelemetry`] all share the same underlying
/// counters/histograms.
#[derive(Clone)]
pub(crate) struct RuntimeMetrics {
    /// Process-wide event counters.
    pub(crate) counters: Arc<Counters>,
    /// Effect-chain processing latency.
    pub(crate) processing: Arc<LatencyHistogram>,
    /// Output-side latency (`push_frame`).
    pub(crate) output: Arc<LatencyHistogram>,
    /// End-to-end supervisor latency: `recv` → post-`push`.
    pub(crate) end_to_end: Arc<LatencyHistogram>,
    /// ML inference latency, populated by effects via
    /// [`EffectTelemetry::record_inference`].  Empty for non-ML
    /// pipelines (passthrough leaves this untouched).
    pub(crate) inference: Arc<LatencyHistogram>,
}

impl RuntimeMetrics {
    /// Build a fresh bundle with empty histograms and zero counters.
    pub(crate) fn new() -> Self {
        let hist = || Arc::new(LatencyHistogram::with_capacity(PER_STAGE_HISTOGRAM_CAPACITY));
        Self {
            counters: Arc::new(Counters::new()),
            processing: hist(),
            output: hist(),
            end_to_end: hist(),
            inference: hist(),
        }
    }

    /// Build the [`EffectTelemetry`] sink the supervisor hands to
    /// effects via [`fluxframe_core::FrameContext::telemetry`].
    /// Cheap `Arc` clone.
    pub(crate) fn effect_telemetry(&self) -> EffectTelemetry {
        EffectTelemetry::with_inference(Arc::clone(&self.inference))
    }

    /// Take a unified [`MetricsSnapshot`] across counters and the
    /// stages we actually populate.  `capture` remains empty until
    /// its plumbing lands (see module header).
    pub(crate) fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot::new()
            .with_counters(self.counters.snapshot())
            .with_processing(self.processing.snapshot())
            .with_output(self.output.snapshot())
            .with_end_to_end(self.end_to_end.snapshot())
            .with_inference(self.inference.snapshot())
    }

    /// Reconcile [`Counters::add_frames_dropped`] against the input
    /// slot's running drop counter (which the slot atomically owns).
    ///
    /// The supervisor calls this opportunistically — there is no
    /// per-event hook into the slot.  The "already counted" baseline
    /// is read from the counter itself, so callers carry no state.
    /// Saturating subtraction defends against a hypothetical slot
    /// impl that resets its counter mid-run.
    ///
    /// **Contract:** this is the *only* path that writes
    /// `frames_dropped`; any other writer would desync the baseline.
    pub(crate) fn sync_dropped(&self, current_total: u64) {
        let already = self.counters.snapshot().frames_dropped;
        let delta = current_total.saturating_sub(already);
        if delta > 0 {
            self.counters.add_frames_dropped(delta);
        }
    }
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_bundle_is_empty() {
        let m = RuntimeMetrics::new();
        let s = m.snapshot();
        assert_eq!(s.counters.frames_in, 0);
        assert!(s.processing.is_empty());
        assert!(s.output.is_empty());
        assert!(s.end_to_end.is_empty());
        assert!(s.inference.is_empty());
        assert!(s.capture.is_empty(), "capture stays unwired in 5.D.3");
    }

    #[test]
    fn effect_telemetry_writes_to_shared_inference_histogram() {
        let m = RuntimeMetrics::new();
        let t = m.effect_telemetry();
        t.record_inference(std::time::Duration::from_micros(8_000));
        t.record_inference(std::time::Duration::from_micros(12_500));
        let s = m.snapshot();
        assert_eq!(s.inference.len(), 2);
        assert_eq!(s.inference.percentile_us(0.5), 8_000);
        assert_eq!(s.inference.percentile_us(1.0), 12_500);
    }

    #[test]
    fn clone_shares_counters() {
        let a = RuntimeMetrics::new();
        let b = a.clone();
        a.counters.inc_frames_in();
        b.counters.inc_frames_in();
        assert_eq!(a.snapshot().counters.frames_in, 2);
        assert_eq!(b.snapshot().counters.frames_in, 2);
    }

    #[test]
    fn snapshot_carries_recorded_samples() {
        let m = RuntimeMetrics::new();
        m.processing.record_us(10_000);
        m.processing.record_us(20_000);
        m.output.record_us(5_000);
        m.end_to_end.record_us(15_000);
        m.counters.inc_frames_in();
        m.counters.inc_frames_out();
        let s = m.snapshot();
        assert_eq!(s.counters.frames_in, 1);
        assert_eq!(s.counters.frames_out, 1);
        assert_eq!(s.processing.percentile_us(1.0), 20_000);
        assert_eq!(s.output.percentile_us(0.5), 5_000);
        assert_eq!(s.end_to_end.percentile_us(0.5), 15_000);
    }

    #[test]
    fn sync_dropped_accumulates_positive_delta() {
        let m = RuntimeMetrics::new();
        m.sync_dropped(5);
        assert_eq!(m.snapshot().counters.frames_dropped, 5);
        m.sync_dropped(8);
        assert_eq!(m.snapshot().counters.frames_dropped, 8);
    }

    #[test]
    fn sync_dropped_zero_total_is_noop() {
        let m = RuntimeMetrics::new();
        m.sync_dropped(0);
        assert_eq!(m.snapshot().counters.frames_dropped, 0);
        // Idempotent at the same total.
        m.sync_dropped(5);
        m.sync_dropped(5);
        assert_eq!(
            m.snapshot().counters.frames_dropped,
            5,
            "no double-count when total is unchanged"
        );
    }

    #[test]
    fn sync_dropped_handles_regressing_total() {
        // Defensive: if a future slot impl resets, we must not
        // underflow the u64 delta.
        let m = RuntimeMetrics::new();
        m.sync_dropped(10);
        m.sync_dropped(3);
        // Negative delta absorbed; counter stays at the high-water
        // mark — no double-count, no underflow.
        assert_eq!(m.snapshot().counters.frames_dropped, 10);
    }

}
