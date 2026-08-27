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

use fluxframe_core::{CONSUMER_EVENT_AGE_UNSET, CounterValues, PeriodicExtras, emit_metrics_line};

use crate::runtime_metrics::RuntimeMetrics;

/// Granularity of the reporter's shutdown poll.  Smaller is more
/// responsive to Ctrl-C, larger spends less CPU.  Sized to match
/// `crate::runtime::WORKER_POLL_TIMEOUT` so the reporter's
/// teardown latency budget cannot exceed the supervisor's — keep
/// the two in sync if either is retuned.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// How long a run of identical ticks may stay silent before one is
/// emitted anyway.
///
/// Suppression exists because an idle daemon otherwise writes the same
/// forty-field line every `interval` forever — that is what turned the
/// operator's log into hundreds of megabytes in which the handful of
/// meaningful lines are unfindable. The heartbeat is the other half of
/// the bargain: "nothing is happening" must still be *visible*, or a
/// dead reporter looks exactly like a quiet one.
const QUIET_HEARTBEAT: Duration = Duration::from_secs(600);

/// Silence threshold for the client-usage gauge, above which a tick is
/// never suppressed.
///
/// The gauge grows every tick by construction, so it cannot take part in
/// the equality test — but a *large* value is the one thing an idle
/// daemon can report that is genuinely alarming (the presence
/// subscription has gone deaf), and hiding it behind a ten-minute
/// heartbeat would make the next incident harder to read than the last.
const EVENT_SILENCE_ALERT: Duration = Duration::from_secs(300);

/// Whether a tick is worth a log line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TickVerdict {
    Emit,
    Suppress,
}

/// Decide whether this tick says anything the previous one did not.
///
/// Pure so it can be tested without threads, a clock or a log capture:
/// the reporter's own tests deliberately do not assert on `tracing`
/// output, so logic left inside the loop would have shipped untested.
///
/// Three counters are excluded from the comparison because they advance
/// on their own schedule and would make every tick look eventful: the
/// idle placeholder frame count, the resync tick count, and the age of
/// the last client-usage event.
fn tick_verdict(
    prev: &CounterValues,
    cur: &CounterValues,
    frames_out_delta: u64,
    dropped_delta: u64,
    since_last_emit: Duration,
    heartbeat: Duration,
) -> TickVerdict {
    if frames_out_delta != 0 || dropped_delta != 0 {
        return TickVerdict::Emit;
    }
    if since_last_emit >= heartbeat {
        return TickVerdict::Emit;
    }
    if cur.consumer_last_external_event_age_secs != CONSUMER_EVENT_AGE_UNSET
        && cur.consumer_last_external_event_age_secs >= EVENT_SILENCE_ALERT.as_secs()
    {
        return TickVerdict::Emit;
    }
    let mut baseline = *prev;
    baseline.idle_frames_pushed_total = cur.idle_frames_pushed_total;
    baseline.consumer_resync_total = cur.consumer_resync_total;
    baseline.consumer_last_external_event_age_secs = cur.consumer_last_external_event_age_secs;
    if baseline == *cur {
        TickVerdict::Suppress
    } else {
        TickVerdict::Emit
    }
}

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

fn reporter_loop(
    metrics: &RuntimeMetrics,
    interval: Duration,
    running: &AtomicBool,
    stopped: &AtomicBool,
) {
    let mut last_tick = Instant::now();
    let mut last_frames_out: u64 = 0;
    let mut last_frames_dropped: u64 = 0;
    let mut last_emit = Instant::now();
    let mut last_emitted: Option<CounterValues> = None;
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

        // The frame deltas are consumed by the verdict, so they are
        // measured every tick even when the line is suppressed —
        // otherwise a suppressed tick would fold its frames into the
        // next emitted window and misreport fps.
        let verdict = last_emitted.as_ref().map_or(TickVerdict::Emit, |prev| {
            tick_verdict(
                prev,
                &snap.counters,
                frames_delta,
                dropped_delta,
                now.duration_since(last_emit),
                QUIET_HEARTBEAT,
            )
        });
        if verdict == TickVerdict::Suppress {
            continue;
        }
        last_emit = now;
        last_emitted = Some(snap.counters);

        emit_metrics_line(
            &snap,
            Some(PeriodicExtras {
                frames_out_delta: frames_delta,
                dropped_delta,
                window: dt,
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A steady-state snapshot: a run that has published a verdict and
    /// is sitting idle.
    fn quiet() -> CounterValues {
        let mut c = CounterValues::default();
        c.frames_in = 100;
        c.frames_out = 100;
        c.consumer_status = 0;
        c
    }

    fn verdict(prev: &CounterValues, cur: &CounterValues) -> TickVerdict {
        tick_verdict(
            prev,
            cur,
            0,
            0,
            Duration::from_secs(30),
            Duration::from_secs(600),
        )
    }

    #[test]
    fn an_identical_tick_is_suppressed() {
        assert_eq!(verdict(&quiet(), &quiet()), TickVerdict::Suppress);
    }

    #[test]
    fn the_three_self_advancing_counters_do_not_make_a_tick_eventful() {
        // Placeholder frames, resync ticks and the event-age gauge move
        // on their own every interval; if they counted, nothing would
        // ever be suppressed and the file would grow exactly as before.
        let mut cur = quiet();
        cur.idle_frames_pushed_total += 300;
        cur.consumer_resync_total += 1;
        cur.consumer_last_external_event_age_secs = 42;
        assert_eq!(verdict(&quiet(), &cur), TickVerdict::Suppress);
    }

    #[test]
    fn a_changed_consumer_verdict_is_always_emitted() {
        // The single line that says the camera woke up must never be
        // the one that gets swallowed.
        let mut cur = quiet();
        cur.consumer_status = 1;
        cur.consumer_transitions_total += 1;
        assert_eq!(verdict(&quiet(), &cur), TickVerdict::Emit);
    }

    #[test]
    fn a_resync_correction_is_always_emitted() {
        let mut cur = quiet();
        cur.consumer_resync_corrections_total += 1;
        assert_eq!(verdict(&quiet(), &cur), TickVerdict::Emit);
    }

    #[test]
    fn frames_flowing_is_always_eventful() {
        assert_eq!(
            tick_verdict(
                &quiet(),
                &quiet(),
                750,
                0,
                Duration::from_secs(30),
                Duration::from_secs(600)
            ),
            TickVerdict::Emit
        );
    }

    #[test]
    fn the_heartbeat_breaks_a_long_silence() {
        assert_eq!(
            tick_verdict(
                &quiet(),
                &quiet(),
                0,
                0,
                Duration::from_secs(601),
                Duration::from_secs(600)
            ),
            TickVerdict::Emit
        );
    }

    #[test]
    fn a_long_event_silence_defeats_suppression() {
        // A deaf presence subscription is the one thing an otherwise
        // idle daemon can report that is worth reading promptly.
        let mut cur = quiet();
        cur.consumer_last_external_event_age_secs = EVENT_SILENCE_ALERT.as_secs();
        assert_eq!(verdict(&quiet(), &cur), TickVerdict::Emit);
    }

    #[test]
    fn an_unmeasured_event_age_does_not_trip_the_silence_alert() {
        // The sentinel is `u64::MAX`; read as seconds it would trip the
        // threshold on every single tick and suppress nothing.
        let mut cur = quiet();
        cur.consumer_last_external_event_age_secs = CONSUMER_EVENT_AGE_UNSET;
        assert_eq!(verdict(&quiet(), &cur), TickVerdict::Suppress);
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
