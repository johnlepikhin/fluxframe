//! Periodic metrics reporter (Stage 5.D.3).
//!
//! A dedicated thread polls the supervisor's [`RuntimeMetrics`] every
//! `interval` seconds and emits a single `info!` line in the §29
//! format (`fps=… dropped=… latency_p50=…ms inference_p50=…ms …`).
//!
//! Lifecycle:
//! * [`MetricsReporter::spawn`] starts the thread.
//! * The thread exits when EITHER the caller-provided `running` flag
//!   flips to `false` (typical Ctrl-C / EOS path) OR the
//!   [`MetricsReporter`] itself is dropped (the supervisor explicitly
//!   tearing the run down).  Both paths join cleanly.
//!
//! The reporter never observes the histograms while holding any
//! supervisor-owned lock — it goes through the same poison-tolerant
//! [`fluxframe_core::LatencyHistogram::snapshot`] path as the
//! teardown summary, so a panic in the supervisor cannot deadlock or
//! crash the reporter.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use tracing::info;

use crate::runtime_metrics::RuntimeMetrics;

/// Granularity of the reporter's shutdown poll.  Smaller is more
/// responsive to Ctrl-C, larger spends less CPU.  Sized to match
/// `crate::runtime::WORKER_POLL_TIMEOUT` so the reporter's
/// teardown latency budget cannot exceed the supervisor's — keep
/// the two in sync if either is retuned.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Handle to the reporter thread.  Dropping joins the thread; a panic
/// in the worker surfaces via the join result and is logged.
pub(crate) struct MetricsReporter {
    handle: Option<JoinHandle<()>>,
    stopped: Arc<AtomicBool>,
}

impl MetricsReporter {
    /// Spawn the reporter thread.  Returns the handle so the caller
    /// can drop it (and thereby join) at run teardown.
    ///
    /// # Errors
    ///
    /// Propagates [`std::io::Error`] from [`std::thread::Builder::spawn`].
    pub(crate) fn spawn(
        metrics: RuntimeMetrics,
        interval: Duration,
        running: Arc<AtomicBool>,
    ) -> std::io::Result<Self> {
        let stopped = Arc::new(AtomicBool::new(false));
        let stopped_for_worker = Arc::clone(&stopped);
        let handle = std::thread::Builder::new()
            .name("fluxframe-metrics".into())
            .spawn(move || reporter_loop(&metrics, interval, &running, &stopped_for_worker))?;
        Ok(Self {
            handle: Some(handle),
            stopped,
        })
    }
}

impl Drop for MetricsReporter {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Release);
        if let Some(h) = self.handle.take() {
            if let Err(e) = h.join() {
                tracing::warn!(?e, "metrics reporter thread panicked while joining");
            }
        }
    }
}

/// Compute frames-per-second from a frame delta and the wall-clock
/// window over which it was observed.  Returns `0.0` for zero-length
/// windows (defensive against a clock that did not advance between
/// ticks, which can happen on systems with coarse `Instant` resolution).
#[inline]
fn compute_fps(frames_delta: u64, dt: Duration) -> f64 {
    let secs = dt.as_secs_f64();
    if secs <= 0.0 {
        0.0
    } else {
        (frames_delta as f64) / secs
    }
}

fn reporter_loop(
    metrics: &RuntimeMetrics,
    interval: Duration,
    running: &AtomicBool,
    stopped: &AtomicBool,
) {
    let mut last_tick = Instant::now();
    let mut last_frames_out: u64 = 0;
    let mut last_frames_dropped: u64 = 0;
    while running.load(Ordering::Acquire) && !stopped.load(Ordering::Acquire) {
        // Sleep no longer than the time remaining until the next
        // scheduled tick, but cap at POLL_INTERVAL so shutdown stays
        // responsive even when `interval` is many seconds.  Without
        // this cap we would either burn 20 wake-ups/s for nothing
        // (constant POLL sleep) or hold shutdown for the full
        // `interval` (constant interval sleep).
        let pre_sleep = Instant::now();
        let until_next = interval.saturating_sub(pre_sleep.duration_since(last_tick));
        let sleep_for = std::cmp::min(POLL_INTERVAL, until_next);
        // `sleep_for == ZERO` is legal — the OS yields and we loop
        // immediately into the tick branch.
        std::thread::sleep(sleep_for);
        // Estimate "now" from the pre-sleep instant + the requested
        // sleep; saves one `clock_gettime` per iteration.  Accuracy
        // loss is bounded by the kernel's sleep slack (microseconds)
        // and irrelevant against the seconds-scale `interval`.
        let now = pre_sleep + sleep_for;
        let dt = now.duration_since(last_tick);
        if dt < interval {
            continue;
        }
        last_tick = now;

        let snap = metrics.snapshot();
        let frames_now = snap.counters.frames_out;
        let frames_delta = frames_now.saturating_sub(last_frames_out);
        last_frames_out = frames_now;
        let dropped_now = snap.counters.frames_dropped;
        let dropped_delta = dropped_now.saturating_sub(last_frames_dropped);
        last_frames_dropped = dropped_now;

        emit(&snap, frames_delta, dropped_delta, dt);
    }
}

fn emit(
    snap: &fluxframe_core::MetricsSnapshot,
    frames_delta: u64,
    dropped_delta: u64,
    window: Duration,
) {
    let fps = compute_fps(frames_delta, window);
    // Round display-only floats to two decimals so the line stays
    // readable (`fps=24.0` rather than `fps=23.965698241213406`).
    // The raw f64 is fine for downstream ingestors but the line is
    // primarily for humans tailing `journalctl`.
    let fps_rounded = round2(fps);
    let window_secs = round2(window.as_secs_f64());
    info!(
        fps = fps_rounded,
        window_secs = window_secs,
        frames_out_delta = frames_delta,
        dropped_delta = dropped_delta,
        frames_in_total = snap.counters.frames_in,
        frames_out_total = snap.counters.frames_out,
        frames_dropped_total = snap.counters.frames_dropped,
        fallback_total = snap.counters.fallback_count,
        effect_err_total = snap.counters.effect_error_count,
        inference_runtime_fallback_gpu_to_cpu = snap.counters.inference_runtime_fallback_gpu_to_cpu,
        blur_runtime_fallback_gpu_to_cpu = snap.counters.blur_runtime_fallback_gpu_to_cpu,
        inference_p50_us = snap.inference.percentile_us(0.5),
        inference_p95_us = snap.inference.percentile_us(0.95),
        processing_p50_us = snap.processing.percentile_us(0.5),
        processing_p95_us = snap.processing.percentile_us(0.95),
        output_p50_us = snap.output.percentile_us(0.5),
        output_p95_us = snap.output.percentile_us(0.95),
        end_to_end_p50_us = snap.end_to_end.percentile_us(0.5),
        end_to_end_p95_us = snap.end_to_end.percentile_us(0.95),
        "metrics tick"
    );
}

/// Round to two decimal places.  Used for display-only float fields
/// in the reporter line; not for any downstream calculation.
#[inline]
fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn compute_fps_zero_window_returns_zero() {
        assert!(approx_eq(compute_fps(100, Duration::ZERO), 0.0));
    }

    #[test]
    fn compute_fps_zero_frames_returns_zero() {
        assert!(approx_eq(compute_fps(0, Duration::from_secs(5)), 0.0));
    }

    #[test]
    fn compute_fps_normal_window() {
        // 150 frames over 5 s → 30 fps.
        let fps = compute_fps(150, Duration::from_secs(5));
        assert!((fps - 30.0).abs() < 1e-9, "expected 30 fps, got {fps}");
    }

    #[test]
    fn compute_fps_sub_second_window() {
        // 3 frames over 100 ms → 30 fps.
        let fps = compute_fps(3, Duration::from_millis(100));
        assert!((fps - 30.0).abs() < 1e-9, "expected 30 fps, got {fps}");
    }

    #[test]
    fn reporter_joins_when_running_flag_drops() {
        let running = Arc::new(AtomicBool::new(true));
        let metrics = RuntimeMetrics::new();
        let reporter =
            MetricsReporter::spawn(metrics, Duration::from_millis(40), Arc::clone(&running))
                .expect("spawn");
        // Give the worker a chance to wake at least once.
        std::thread::sleep(Duration::from_millis(120));
        running.store(false, Ordering::Release);
        // Drop joins the thread; if it does not exit promptly the
        // test will hang and CI catches it.
        drop(reporter);
    }

    #[test]
    fn reporter_joins_when_dropped_even_if_running_stays_true() {
        let running = Arc::new(AtomicBool::new(true));
        let metrics = RuntimeMetrics::new();
        let reporter =
            MetricsReporter::spawn(metrics, Duration::from_millis(40), Arc::clone(&running))
                .expect("spawn");
        // Do NOT flip `running`; the Drop-induced `stopped` flag
        // must be sufficient on its own.
        drop(reporter);
        // `running` still observable here — the supervisor would
        // typically flip it on its own teardown path.
        assert!(running.load(Ordering::Acquire));
    }

    #[test]
    fn round2_basic() {
        assert!(approx_eq(round2(23.965_698_241), 23.97));
        assert!(approx_eq(round2(0.0), 0.0));
        assert!(approx_eq(round2(5.007_156_428), 5.01));
        assert!(approx_eq(round2(100.0), 100.0));
        // Halfway cases — Rust's f64::round is round-half-away-from-zero,
        // but 1.005 cannot be represented exactly in binary float, so the
        // observed value is one of 1.00 or 1.01 depending on the platform.
        assert!(approx_eq(round2(1.005), 1.01) || approx_eq(round2(1.005), 1.00));
    }

    #[test]
    fn reporter_emits_at_least_once_when_window_elapses() {
        // Smoke test: exercise the snapshot + emit path inside the
        // loop without asserting on log output (capturing tracing
        // from a unit test is more trouble than it is worth here —
        // we just confirm the loop runs through `emit` without
        // panicking on an empty snapshot).
        let running = Arc::new(AtomicBool::new(true));
        let metrics = RuntimeMetrics::new();
        // Populate a little so the snapshot has non-zero content.
        metrics.counters.inc_frames_in();
        metrics.counters.inc_frames_out();
        metrics.processing.record_us(1000);
        let reporter =
            MetricsReporter::spawn(metrics, Duration::from_millis(40), Arc::clone(&running))
                .expect("spawn");
        std::thread::sleep(Duration::from_millis(150));
        running.store(false, Ordering::Release);
        drop(reporter);
    }
}
