//! Realtime metrics: counters and latency histograms.
//!
//! Wire-compatible with §29 of the spec (periodic
//! `fps=… dropped=… latency_p50=…ms latency_p95=…ms inference_p50=…ms`
//! reporter).  This module defines the pure data types only; the
//! supervisor wiring (incrementing counters, recording latencies on
//! each frame) and the periodic CLI reporter ship in follow-on
//! commits (5.D.2 and 5.D.3 of the Stage 5 plan).
//!
//! Design:
//! * [`Counters`] is `Arc`-sharable; the supervisor's hot loop and the
//!   reporter thread access the same counter struct via independent
//!   atomic loads/stores — no lock contention.
//! * [`LatencyHistogram`] is a bounded ring of microsecond samples
//!   behind a `Mutex`.  Lock hold time is one `push_back` (or
//!   `pop_front`+`push_back`) per record; snapshot drains into a
//!   sorted `Vec` and releases the lock before percentile queries.
//! * Snapshot types ([`CounterValues`], [`LatencySnapshot`],
//!   [`MetricsSnapshot`]) are plain owned data, safe to clone and
//!   pass across thread boundaries.
//!
//! `Relaxed` ordering is fine for the counters: the reporter does not
//! reason about happens-before between distinct counters, and the
//! `Arc` itself provides the synchronisation needed to observe the
//! struct at all.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Process-wide event counters.
///
/// Shared between the supervisor (writer) and any metrics readers via
/// `Arc<Counters>`.  Each counter is independent; readers MUST NOT
/// assume cross-counter consistency in a single snapshot.
///
/// The atomic fields are private to keep the memory-ordering choice
/// (`Ordering::Relaxed`, see module header) encapsulated in one place.
/// Writers go through the typed `inc_*` / `add_*` helpers; readers
/// take a [`CounterValues`] via [`Counters::snapshot`].
#[derive(Debug)]
pub struct Counters {
    frames_in: AtomicU64,
    frames_out: AtomicU64,
    frames_dropped: AtomicU64,
    fallback_count: AtomicU64,
    effect_error_count: AtomicU64,
}

impl Counters {
    /// Construct counters all initialised to zero.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frames_in: AtomicU64::new(0),
            frames_out: AtomicU64::new(0),
            frames_dropped: AtomicU64::new(0),
            fallback_count: AtomicU64::new(0),
            effect_error_count: AtomicU64::new(0),
        }
    }

    /// Increment `frames_in` by one.  Frame counters use `Relaxed` —
    /// the reporter snapshots independently of any other counter.
    #[inline]
    pub fn inc_frames_in(&self) {
        self.frames_in.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `frames_out` by one.
    #[inline]
    pub fn inc_frames_out(&self) {
        self.frames_out.fetch_add(1, Ordering::Relaxed);
    }

    /// Add `n` to `frames_dropped` — supervisor reports its slot-drop
    /// delta in bulk, so the helper takes `u64` rather than a single
    /// increment.
    #[inline]
    pub fn add_frames_dropped(&self, n: u64) {
        if n != 0 {
            self.frames_dropped.fetch_add(n, Ordering::Relaxed);
        }
    }

    /// Increment `fallback_count` by one.
    #[inline]
    pub fn inc_fallback(&self) {
        self.fallback_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `effect_error_count` by one.
    #[inline]
    pub fn inc_effect_error(&self) {
        self.effect_error_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot the counter values.  Each load is independent — there
    /// is no cross-counter atomicity guarantee.
    #[must_use]
    pub fn snapshot(&self) -> CounterValues {
        CounterValues {
            frames_in: self.frames_in.load(Ordering::Relaxed),
            frames_out: self.frames_out.load(Ordering::Relaxed),
            frames_dropped: self.frames_dropped.load(Ordering::Relaxed),
            fallback_count: self.fallback_count.load(Ordering::Relaxed),
            effect_error_count: self.effect_error_count.load(Ordering::Relaxed),
        }
    }
}

impl Default for Counters {
    fn default() -> Self {
        Self::new()
    }
}

/// Plain-data snapshot of [`Counters`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct CounterValues {
    /// See [`Counters::frames_in`].
    pub frames_in: u64,
    /// See [`Counters::frames_out`].
    pub frames_out: u64,
    /// See [`Counters::frames_dropped`].
    pub frames_dropped: u64,
    /// See [`Counters::fallback_count`].
    pub fallback_count: u64,
    /// See [`Counters::effect_error_count`].
    pub effect_error_count: u64,
}

/// Bounded ring of latency samples in microseconds.
///
/// Once `capacity` samples are stored, recording a new sample evicts
/// the oldest one.  Percentile queries operate on a sorted snapshot
/// (see [`LatencyHistogram::snapshot`]).
///
/// Sized for a few seconds of frame-rate data: 1024 samples at 30 fps
/// is ~34 s of history, plenty for a periodic reporter ticking every
/// 5 s.
#[derive(Debug)]
pub struct LatencyHistogram {
    samples: Mutex<VecDeque<u64>>,
    capacity: usize,
}

impl LatencyHistogram {
    /// Construct a histogram retaining at most `capacity` samples.
    ///
    /// A zero `capacity` is clamped to 1 so callers cannot accidentally
    /// create a sink that silently discards every input.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            samples: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    /// Record one latency sample given as a [`Duration`].  Saturates at
    /// `u64::MAX` µs (≈584 942 years) so no real latency ever truncates;
    /// the cap exists only to keep the API total in the face of an
    /// arithmetic bug elsewhere.  Thin wrapper around
    /// [`LatencyHistogram::record_us`].
    #[inline]
    pub fn record_duration(&self, d: Duration) {
        let us = u64::try_from(d.as_micros()).unwrap_or(u64::MAX);
        self.record_us(us);
    }

    /// Record one latency sample in microseconds.  Evicts the oldest
    /// sample if the ring is already full.
    ///
    /// Recovers from mutex poisoning (`PoisonError::into_inner`):
    /// observability must never escalate an unrelated thread's panic
    /// into a kill of the producer.  The contents stay valid because
    /// every critical section is one `pop_front` + `push_back` with no
    /// invariants that span operations.
    #[inline]
    pub fn record_us(&self, value: u64) {
        let mut buf = self
            .samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if buf.len() == self.capacity {
            buf.pop_front();
        }
        buf.push_back(value);
    }

    /// Drain the histogram's current contents into a sorted snapshot
    /// suitable for percentile queries.  The histogram itself is NOT
    /// emptied; recording continues against the live ring.
    ///
    /// Poison-tolerant for the same reason as
    /// [`LatencyHistogram::record_us`].
    #[must_use]
    pub fn snapshot(&self) -> LatencySnapshot {
        let buf = self
            .samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut owned: Vec<u64> = Vec::with_capacity(buf.len());
        owned.extend(buf.iter().copied());
        // Release the lock before sorting — sort is O(n log n) and we
        // do not want to block other recorders for that long.
        drop(buf);
        owned.sort_unstable();
        LatencySnapshot { sorted: owned }
    }

    /// Capacity supplied at construction (post-clamp).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Sorted snapshot of a [`LatencyHistogram`]'s samples.  Percentile
/// queries are O(1) lookups into the pre-sorted vector.
#[derive(Debug, Clone, Default)]
pub struct LatencySnapshot {
    sorted: Vec<u64>,
}

impl LatencySnapshot {
    /// Number of samples in the snapshot.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sorted.len()
    }

    /// `true` when no samples have been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }

    /// Nearest-rank percentile in microseconds.  `p` is a fraction in
    /// `[0.0, 1.0]`; values outside that range are clamped, and `NaN`
    /// is treated as `0` (returns the minimum).  Returns `0` when the
    /// snapshot is empty.
    ///
    /// Convention: `percentile_us(0.5)` is the median (lower of two
    /// middle samples for even-sized inputs); `percentile_us(0.95)`
    /// is the 95th-percentile sample.  Matches what monitoring tools
    /// like Prometheus' `histogram_quantile` report for small N.
    #[must_use]
    pub fn percentile_us(&self, p: f32) -> u64 {
        if self.sorted.is_empty() {
            return 0;
        }
        // `f32::clamp` propagates NaN unchanged; handle it explicitly
        // so the caller cannot accidentally read `sorted[0]` (via the
        // `saturating_sub(1)` fallback below) when they passed NaN.
        let p = if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) };
        let n = self.sorted.len();
        let last = n - 1;
        // Nearest-rank: position = ceil(p * n), index = position - 1.
        // p == 0 returns the minimum, p == 1 returns the maximum.
        // `as usize` is safe here: post-clamp `p ∈ [0, 1]` and `n` fits
        // in usize by construction, so the f64 result is in `[0, n]` ⊂
        // `[0, usize::MAX]` — saturating-cast semantics never trigger.
        let position = (f64::from(p) * n as f64).ceil() as usize;
        let idx = position.saturating_sub(1).min(last);
        self.sorted[idx]
    }
}

/// Combined snapshot of all metrics — what the periodic reporter
/// consumes for one `info!` line.
///
/// `#[non_exhaustive]` so adding a per-stage histogram (e.g. a
/// future GPU-encode stage) does not break downstream consumers.
/// Construct via [`MetricsSnapshot::new`] / `with_*` builders rather
/// than the record literal syntax (which is forbidden cross-crate by
/// the `#[non_exhaustive]` attribute).
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct MetricsSnapshot {
    /// Counter snapshot taken at the same moment as the histograms.
    pub counters: CounterValues,
    /// End-to-end latency: capture timestamp → output push.  Dominates
    /// what an operator perceives as "delay".
    pub end_to_end: LatencySnapshot,
    /// Capture-side latency: capture timestamp → effect chain entry.
    pub capture: LatencySnapshot,
    /// Effect chain processing time (mask + composite).
    pub processing: LatencySnapshot,
    /// ML inference time only (subset of `processing`).
    pub inference: LatencySnapshot,
    /// Output-side latency: effect chain exit → output push.
    pub output: LatencySnapshot,
}

/// Telemetry sink handed to an effect via [`crate::context::FrameContext`].
///
/// An effect that owns per-stage timings (e.g. ML inference inside a
/// composite effect) calls [`EffectTelemetry::record_inference`] on
/// each frame; the supervisor reads the underlying histogram via
/// [`crate::metrics::MetricsSnapshot::inference`].
///
/// `Default` returns a no-op sink: tests and downstream callers that
/// do not wire metrics can still hand a [`crate::context::FrameContext`]
/// to an effect without fabricating a real [`LatencyHistogram`].
#[derive(Debug, Clone, Default)]
pub struct EffectTelemetry {
    inference: Option<Arc<LatencyHistogram>>,
}

impl EffectTelemetry {
    /// Build a telemetry sink that publishes inference timings into
    /// the given histogram.
    #[must_use]
    pub fn with_inference(inference: Arc<LatencyHistogram>) -> Self {
        Self {
            inference: Some(inference),
        }
    }

    /// Record one inference latency sample.  No-op when the sink was
    /// built without an inference histogram (test/default path).
    #[inline]
    pub fn record_inference(&self, d: Duration) {
        if let Some(h) = &self.inference {
            h.record_duration(d);
        }
    }
}

impl MetricsSnapshot {
    /// Construct an empty snapshot.  Every stage histogram defaults to
    /// empty; counters default to zero.  Combine with the `with_*`
    /// builders below to populate stages a producer actually tracks.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the counter snapshot.
    #[must_use]
    pub fn with_counters(mut self, counters: CounterValues) -> Self {
        self.counters = counters;
        self
    }

    /// Replace the end-to-end latency snapshot.
    #[must_use]
    pub fn with_end_to_end(mut self, snap: LatencySnapshot) -> Self {
        self.end_to_end = snap;
        self
    }

    /// Replace the capture-side latency snapshot.
    #[must_use]
    pub fn with_capture(mut self, snap: LatencySnapshot) -> Self {
        self.capture = snap;
        self
    }

    /// Replace the effect-chain processing latency snapshot.
    #[must_use]
    pub fn with_processing(mut self, snap: LatencySnapshot) -> Self {
        self.processing = snap;
        self
    }

    /// Replace the inference-only latency snapshot (subset of
    /// processing).
    #[must_use]
    pub fn with_inference(mut self, snap: LatencySnapshot) -> Self {
        self.inference = snap;
        self
    }

    /// Replace the output-side latency snapshot.
    #[must_use]
    pub fn with_output(mut self, snap: LatencySnapshot) -> Self {
        self.output = snap;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_default_zero() {
        let c = Counters::new();
        let snap = c.snapshot();
        assert_eq!(snap.frames_in, 0);
        assert_eq!(snap.frames_out, 0);
        assert_eq!(snap.frames_dropped, 0);
        assert_eq!(snap.fallback_count, 0);
        assert_eq!(snap.effect_error_count, 0);
    }

    #[test]
    fn counters_increment_visible_in_snapshot() {
        let c = Counters::new();
        for _ in 0..5 {
            c.inc_frames_in();
        }
        for _ in 0..3 {
            c.inc_frames_out();
        }
        c.add_frames_dropped(1);
        for _ in 0..7 {
            c.inc_fallback();
        }
        for _ in 0..2 {
            c.inc_effect_error();
        }
        let snap = c.snapshot();
        assert_eq!(snap.frames_in, 5);
        assert_eq!(snap.frames_out, 3);
        assert_eq!(snap.frames_dropped, 1);
        assert_eq!(snap.fallback_count, 7);
        assert_eq!(snap.effect_error_count, 2);
    }

    #[test]
    fn counters_add_frames_dropped_zero_is_noop() {
        let c = Counters::new();
        c.add_frames_dropped(0);
        assert_eq!(c.snapshot().frames_dropped, 0);
        c.add_frames_dropped(42);
        c.add_frames_dropped(0);
        assert_eq!(c.snapshot().frames_dropped, 42);
    }

    #[test]
    fn histogram_zero_capacity_clamped_to_one() {
        let h = LatencyHistogram::with_capacity(0);
        assert_eq!(h.capacity(), 1);
        h.record_us(42);
        h.record_us(99);
        let snap = h.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap.percentile_us(0.5), 99);
    }

    #[test]
    fn histogram_evicts_oldest_when_full() {
        let h = LatencyHistogram::with_capacity(3);
        for v in [10, 20, 30, 40, 50] {
            h.record_us(v);
        }
        // 10, 20 evicted; 30, 40, 50 remain.
        let snap = h.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap.percentile_us(0.0), 30);
        assert_eq!(snap.percentile_us(1.0), 50);
    }

    #[test]
    fn empty_snapshot_percentile_is_zero() {
        let snap = LatencySnapshot::default();
        assert!(snap.is_empty());
        assert_eq!(snap.percentile_us(0.5), 0);
        assert_eq!(snap.percentile_us(0.95), 0);
    }

    #[test]
    fn percentile_clamps_out_of_range_p() {
        let h = LatencyHistogram::with_capacity(5);
        for v in [10, 20, 30, 40, 50] {
            h.record_us(v);
        }
        let snap = h.snapshot();
        assert_eq!(snap.percentile_us(-1.0), 10, "p<0 clamps to min");
        assert_eq!(snap.percentile_us(2.0), 50, "p>1 clamps to max");
    }

    #[test]
    fn percentile_nan_treated_as_zero() {
        let h = LatencyHistogram::with_capacity(5);
        for v in [10, 20, 30, 40, 50] {
            h.record_us(v);
        }
        let snap = h.snapshot();
        assert_eq!(
            snap.percentile_us(f32::NAN),
            10,
            "NaN must behave as p=0 (min)"
        );
    }

    #[test]
    fn snapshot_sorts_unordered_input() {
        let h = LatencyHistogram::with_capacity(8);
        for v in [50, 10, 30, 20, 40] {
            h.record_us(v);
        }
        let snap = h.snapshot();
        // Percentile-based assertions exercise the sorted invariant
        // end-to-end without an internal accessor: min, median, max
        // can only line up if the underlying vector is sorted.
        assert_eq!(snap.percentile_us(0.0), 10);
        assert_eq!(snap.percentile_us(0.5), 30);
        assert_eq!(snap.percentile_us(1.0), 50);
    }

    #[test]
    fn percentile_median_and_p95() {
        let h = LatencyHistogram::with_capacity(100);
        for v in 1u64..=100 {
            h.record_us(v);
        }
        let snap = h.snapshot();
        assert_eq!(snap.len(), 100);
        // Nearest-rank: p50 → ceil(50)=50 → idx 49 → 50.
        assert_eq!(snap.percentile_us(0.5), 50);
        // p95 → ceil(95)=95 → idx 94 → 95.
        assert_eq!(snap.percentile_us(0.95), 95);
        // p0 → idx 0 → 1.  p1 → idx 99 → 100.
        assert_eq!(snap.percentile_us(0.0), 1);
        assert_eq!(snap.percentile_us(1.0), 100);
    }

    #[test]
    fn percentile_single_sample() {
        let h = LatencyHistogram::with_capacity(8);
        h.record_us(42);
        let snap = h.snapshot();
        assert_eq!(snap.percentile_us(0.5), 42);
        assert_eq!(snap.percentile_us(0.95), 42);
        assert_eq!(snap.percentile_us(0.0), 42);
    }

    #[test]
    fn snapshot_does_not_drain_histogram() {
        let h = LatencyHistogram::with_capacity(8);
        h.record_us(10);
        h.record_us(20);
        let snap1 = h.snapshot();
        let snap2 = h.snapshot();
        assert_eq!(snap1.len(), 2);
        assert_eq!(snap2.len(), 2, "second snapshot sees same data");
    }

    #[test]
    fn histogram_concurrent_record_does_not_lose_samples() {
        use std::sync::Arc;
        use std::thread;
        let h = Arc::new(LatencyHistogram::with_capacity(10_000));
        let mut handles = Vec::new();
        for t in 0u64..4 {
            let h = Arc::clone(&h);
            handles.push(thread::spawn(move || {
                for i in 0u64..250 {
                    h.record_us(t * 1000 + i);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let snap = h.snapshot();
        assert_eq!(snap.len(), 1000, "all 4 * 250 samples retained");
    }

    #[test]
    fn histogram_concurrent_eviction_keeps_ring_bounded() {
        use std::sync::Arc;
        use std::thread;
        // Small capacity forces interleaved pop_front + push_back
        // under contention — the actual non-trivial concurrency path.
        let cap = 100usize;
        let h = Arc::new(LatencyHistogram::with_capacity(cap));
        let mut handles = Vec::new();
        for t in 0u64..4 {
            let h = Arc::clone(&h);
            handles.push(thread::spawn(move || {
                for i in 0u64..1000 {
                    h.record_us(t * 10_000 + i);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let snap = h.snapshot();
        assert_eq!(snap.len(), cap, "ring must be bounded under contention");
        // All retained samples fall inside the value range the
        // producers actually emitted.
        let max_val = 4 * 10_000 + 1000;
        assert!(snap.percentile_us(1.0) < max_val);
    }

    #[test]
    fn percentile_infinity_clamps() {
        let h = LatencyHistogram::with_capacity(3);
        for v in [10, 20, 30] {
            h.record_us(v);
        }
        let snap = h.snapshot();
        assert_eq!(
            snap.percentile_us(f32::INFINITY),
            30,
            "+inf clamps to max (p=1)"
        );
        assert_eq!(
            snap.percentile_us(f32::NEG_INFINITY),
            10,
            "-inf clamps to min (p=0)"
        );
    }

    #[test]
    fn metrics_snapshot_default_is_empty() {
        let s = MetricsSnapshot::default();
        assert_eq!(s.counters.frames_in, 0);
        assert!(s.end_to_end.is_empty());
        assert!(s.inference.is_empty());
    }

    #[test]
    fn record_duration_converts_to_microseconds() {
        let h = LatencyHistogram::with_capacity(4);
        h.record_duration(Duration::from_millis(2));
        h.record_duration(Duration::from_micros(500));
        h.record_duration(Duration::ZERO);
        let snap = h.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap.percentile_us(0.0), 0);
        assert_eq!(snap.percentile_us(0.5), 500);
        assert_eq!(snap.percentile_us(1.0), 2_000);
    }

    #[test]
    fn record_duration_saturates_on_overflow() {
        let h = LatencyHistogram::with_capacity(1);
        // Far above u64::MAX microseconds — saturate, don't panic.
        h.record_duration(Duration::MAX);
        let snap = h.snapshot();
        assert_eq!(snap.percentile_us(1.0), u64::MAX);
    }

    #[test]
    fn effect_telemetry_default_is_noop() {
        let t = EffectTelemetry::default();
        t.record_inference(Duration::from_millis(5));
        // No panic, no observable side effect — defaults to no histogram.
    }

    #[test]
    fn effect_telemetry_with_inference_records_into_histogram() {
        let h = Arc::new(LatencyHistogram::with_capacity(4));
        let t = EffectTelemetry::with_inference(Arc::clone(&h));
        t.record_inference(Duration::from_micros(7_500));
        t.record_inference(Duration::from_micros(12_000));
        let snap = h.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap.percentile_us(0.5), 7_500);
        assert_eq!(snap.percentile_us(1.0), 12_000);
    }

    #[test]
    fn effect_telemetry_clone_shares_underlying_histogram() {
        let h = Arc::new(LatencyHistogram::with_capacity(4));
        let a = EffectTelemetry::with_inference(Arc::clone(&h));
        let b = a.clone();
        a.record_inference(Duration::from_micros(1));
        b.record_inference(Duration::from_micros(2));
        let snap = h.snapshot();
        assert_eq!(snap.len(), 2);
    }

    #[test]
    fn metrics_snapshot_builders_compose() {
        let c = Counters::new();
        c.inc_frames_in();
        c.inc_frames_in();
        let h = LatencyHistogram::with_capacity(8);
        h.record_us(100);
        h.record_us(200);
        let snap = MetricsSnapshot::new()
            .with_counters(c.snapshot())
            .with_processing(h.snapshot())
            .with_end_to_end(h.snapshot());
        assert_eq!(snap.counters.frames_in, 2);
        assert_eq!(snap.processing.len(), 2);
        assert_eq!(snap.processing.percentile_us(1.0), 200);
        assert_eq!(snap.end_to_end.len(), 2);
        // Untouched stages remain empty.
        assert!(snap.capture.is_empty());
        assert!(snap.inference.is_empty());
        assert!(snap.output.is_empty());
    }
}
