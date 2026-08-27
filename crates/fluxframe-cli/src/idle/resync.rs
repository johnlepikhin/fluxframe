//! Level-triggered repair for the consumer-presence verdict.
//!
//! The detector's primary signal — v4l2loopback's
//! `V4L2_EVENT_PRI_CLIENT_USAGE` — is edge-triggered, and that makes it
//! a single point of failure. An event the driver never queues, or
//! never delivers, leaves the last verdict standing for the rest of the
//! run; twice in production that verdict was `Absent`, and the camera
//! stayed down until the daemon was restarted (49 minutes once, 19
//! hours the next time).
//!
//! This module is the answer: on a timer it re-reads the *absolute*
//! value from the driver through a fresh descriptor
//! ([`fluxframe_gst::probe_client_usage`]) and corrects the verdict when
//! the two disagree. Everything here hangs off [`resync_tick`], which
//! the detector's client-usage loop calls when [`ResyncState::due`]
//! says so.
//!
//! ## Why a fresh descriptor, and what it costs
//!
//! The kernel's `v4l2_event_subscribe()` is a no-op for an
//! already-subscribed `(type, id)` pair on the same file handle, so
//! re-arming `SEND_INITIAL` in place would never replay the value. A
//! new descriptor also means the driver answers by queueing an event to
//! *every* subscriber on the node, including the detector's own watch —
//! see [`drain_probe_echo`] for why that echo has to be consumed
//! explicitly rather than left to look like consumer traffic.
//!
//! ## Two side readings
//!
//! The same tick answers two questions nothing else in the daemon does:
//! whether *we* still hold the loopback's OUTPUT stream
//! ([`check_output_stream`] — if we do not, every consumer's `STREAMON`
//! fails with `EIO` while our own frame counters keep climbing), and
//! whether somebody holds the node open without streaming
//! ([`check_external_holder`] — the fingerprint of a client that lost
//! the race for v4l2loopback's single capture slot).
//!
//! ## Scope
//!
//! Only the kernel-event presence source uses any of this. The
//! `inotify` heuristic cannot: the probe's own `open`/`close` would
//! land in that path's open-balance and read as a consumer attaching.

use std::io;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use fluxframe_core::{CONSUMER_EVENT_AGE_UNSET, EXTERNAL_OPENERS_UNSET, OUTPUT_STREAM_UNKNOWN};
use fluxframe_gst::{
    ClientUsage, LoopbackState, ProbeFailure, classify_probe_error, probe_client_usage,
    read_loopback_state,
};
use tracing::{debug, info, warn};

use super::detector::{
    DetectorContext, DetectorState, UsageSource, WalkOutcome, apply_usage,
    count_external_consumers, read_comm, shutdown_bounded_timeout_ms,
};
use super::state::ConsumerStatus;

/// On-demand, authoritative readings about the loopback node — the
/// level-triggered counterpart to the edge-triggered [`UsageSource`].
///
/// Both methods answer "what is true right now", which is what makes a
/// latched verdict recoverable: the event stream can go silent, but a
/// fresh read cannot.
///
/// A trait for the same reason as [`UsageSource`] — the production
/// implementation needs a real v4l2loopback node, which no unit test has.
pub(super) trait NodeProbe {
    /// Re-read capture usage from the driver.
    ///
    /// Total by design: "the kernel did not answer" is an `Err`, never a
    /// reading, so it can never be mistaken for "nobody is streaming".
    ///
    /// # Errors
    ///
    /// See [`fluxframe_gst::probe_client_usage`]; callers must treat
    /// `ENOTTY`/`EINVAL`/`EACCES` as permanent and everything else as
    /// worth retrying.
    fn client_usage(&self) -> io::Result<ClientUsage>;

    /// Does the producer still hold the node's OUTPUT stream?
    ///
    /// This is a self-check, not an observation about consumers — see
    /// [`LoopbackState`].
    fn output_state(&self) -> LoopbackState;
}

/// [`NodeProbe`] against a real `/dev/videoN`.
pub(super) struct DeviceProbe {
    device: PathBuf,
}

impl DeviceProbe {
    pub(super) fn new(device: PathBuf) -> Self {
        Self { device }
    }
}

impl NodeProbe for DeviceProbe {
    fn client_usage(&self) -> io::Result<ClientUsage> {
        // The driver answers `SEND_INITIAL` synchronously, so the wait is
        // a formality — but it still runs on the thread `Drop` joins, so
        // it gets the same budget as every other blocking call here.
        // Note the budget covers the `poll`, not the `open` before it:
        // that one takes an interruptible driver mutex and has no
        // timeout to give it. See `probe_client_usage`.
        probe_client_usage(&self.device, shutdown_bounded_timeout_ms())
    }

    fn output_state(&self) -> LoopbackState {
        read_loopback_state(&self.device)
    }
}

/// How often the client-usage loop takes an authoritative re-read.
#[derive(Debug, Clone, Copy)]
pub(super) enum ResyncSchedule {
    /// Never — `idle.resync_interval_secs = 0`. The verdict is then only
    /// as good as the event stream, which is what the incidents this
    /// mechanism exists for looked like.
    Off,
    /// Wall-clock cadence; what the supervisor builds from config.
    Every(Duration),
    /// Fire every `n`-th loop iteration.
    ///
    /// Test-only seam: the scripted [`UsageSource`] fakes drain in a few
    /// iterations and stop the loop, so a wall-clock cadence would never
    /// fire and the resync path would go untested.
    #[cfg(test)]
    EveryIterations(u32),
}

impl ResyncSchedule {
    /// Build a wall-clock schedule, mapping a zero interval onto
    /// [`Self::Off`].
    ///
    /// A zero-length `Every` would re-open the device on every loop
    /// iteration — 20 opens a second against a node with ten opener
    /// slots. The config layer already rejects it
    /// (`MIN_IDLE_RESYNC_INTERVAL_SECS`), but the invariant belongs to
    /// the type as well: a second construction site must not be able to
    /// reintroduce it.
    pub(super) fn every(interval: Duration) -> Self {
        if interval.is_zero() {
            Self::Off
        } else {
            Self::Every(interval)
        }
    }

    /// The configured cadence, or zero for schedules that have none.
    fn interval(self) -> Duration {
        match self {
            Self::Every(d) => d,
            Self::Off => Duration::ZERO,
            #[cfg(test)]
            Self::EveryIterations(_) => Duration::ZERO,
        }
    }
}

/// Backoff ceiling after `EBUSY` from the probe's `open`.
///
/// `EBUSY` means the node is at `max_openers` — the one moment when one
/// more opener is actively harmful, because a real consumer trying to
/// attach right then gets the same error. So the diagnostic backs off
/// instead of competing for the last slot.
const RESYNC_BACKOFF_MAX: Duration = Duration::from_secs(300);

/// First backoff step after an `EBUSY`, before doubling.
const RESYNC_BACKOFF_START: Duration = Duration::from_secs(1);

/// How many consecutive re-reads must agree on "nobody is streaming"
/// before the verdict is downgraded.
///
/// One is enough in the other direction. This side releases the camera
/// `idle.teardown_secs` later, under whoever is using it, so it is worth
/// one extra interval of delay. The value is part of the documented
/// contract (README, changelog) — changing it changes user-visible
/// behaviour.
const RESYNC_ABSENT_CONFIRMATIONS: u8 = 2;

/// How often the `/proc` census behind `external_openers` may run.
///
/// The census answers "is somebody holding the node without streaming",
/// which changes on human timescales, and it costs a full `/proc` walk.
/// Running it on every resync tick would put that walk in the idle
/// steady state — where the daemon spends most of its life — for a gauge
/// that almost never changes. A transition still forces a fresh census
/// (see [`ResyncState::census_due`]), so the answer is never stale
/// across a state change.
const EXTERNAL_CENSUS_INTERVAL: Duration = Duration::from_secs(300);

/// Everything the resync tick has to remember between firings.
///
/// All of it is edge state: the counters answer "how often", this
/// answers "has anything changed since last time" — which is what keeps
/// a persistent fault from filling the log with one line per tick.
pub(super) struct ResyncState {
    schedule: ResyncSchedule,
    last_tick: Instant,
    iterations: u32,
    /// Set once a probe error proves permanent; stops the ticking for
    /// the rest of the run rather than retrying 120 times an hour.
    disabled: bool,
    /// Deadline set after `EBUSY`, the current backoff step, and the
    /// step to return to once a probe succeeds again. Without the last
    /// one the escalation is permanent: isolated `EBUSY`s minutes apart
    /// would keep doubling from wherever the previous one left off.
    backoff_until: Option<Instant>,
    backoff: Duration,
    backoff_base: Duration,
    /// Consecutive re-reads that said "nobody" while the verdict says
    /// somebody. [`RESYNC_ABSENT_CONFIRMATIONS`] are required before
    /// acting, and any contradicting event resets the run — two
    /// re-reads separated by "a client is streaming" are not two
    /// consecutive re-reads.
    absent_streak: u8,
    /// Edge trackers for the three conditions worth one log line each.
    probe_failing: bool,
    output_down: Option<bool>,
    external_holder: Option<bool>,
    /// When the census behind `external_holder` last ran.
    last_census: Option<Instant>,
    /// When the last event we did NOT cause ourselves arrived, and the
    /// baseline to measure against before any has.
    last_external_event: Option<Instant>,
    started_at: Instant,
    /// How long the node may stay event-silent before saying so, and
    /// whether that has already been said.
    stale_after: Duration,
    stale_warned: bool,
}

impl ResyncState {
    pub(super) fn new(schedule: ResyncSchedule, teardown_grace: Duration) -> Self {
        let now = Instant::now();
        let interval = schedule.interval();
        Self {
            schedule,
            last_tick: now,
            iterations: 0,
            disabled: false,
            backoff_until: None,
            backoff: interval,
            backoff_base: interval,
            absent_streak: 0,
            probe_failing: false,
            output_down: None,
            external_holder: None,
            last_census: None,
            last_external_event: None,
            started_at: now,
            // Three intervals of silence is not yet suspicious on a
            // healthy but unused node; below the idle grace period the
            // warning would race the hysteresis it is describing.
            stale_after: (interval * 3).max(teardown_grace),
            stale_warned: false,
        }
    }

    /// Announce at startup which of the two states the safety net is in.
    ///
    /// Deliberately not part of [`Self::new`]: a constructor that logs
    /// cannot be called from a test or a diagnostic path without
    /// polluting the output.
    pub(super) fn announce(&self) {
        if matches!(self.schedule, ResyncSchedule::Off) {
            warn!(
                target: "fluxframe::idle",
                "consumer resync disabled by config — a lost kernel event will latch \
                 the verdict until the daemon restarts"
            );
        } else {
            info!(
                target: "fluxframe::idle",
                resync_interval_secs = self.schedule.interval().as_secs(),
                "consumer resync armed"
            );
        }
    }

    /// An event arrived that we did not induce ourselves.
    pub(super) fn note_external_event(&mut self) {
        self.last_external_event = Some(Instant::now());
        self.stale_warned = false;
        // A re-read run before this event and one run after it are not
        // "consecutive": the event is evidence against the downgrade
        // they were accumulating towards.
        self.absent_streak = 0;
    }

    /// Count one iteration of the detector loop.
    ///
    /// Separate from [`Self::due`] so the predicate stays a predicate —
    /// and so the test schedule counts loop iterations rather than
    /// "iterations on which somebody happened to ask".
    pub(super) fn note_iteration(&mut self) {
        self.iterations = self.iterations.saturating_add(1);
    }

    pub(super) fn due(&self) -> bool {
        if self.disabled {
            return false;
        }
        if self
            .backoff_until
            .is_some_and(|until| Instant::now() < until)
        {
            return false;
        }
        match self.schedule {
            ResyncSchedule::Off => false,
            ResyncSchedule::Every(interval) => self.last_tick.elapsed() >= interval,
            #[cfg(test)]
            ResyncSchedule::EveryIterations(n) => n > 0 && self.iterations % n == 0,
        }
    }

    /// May the `/proc` census run on this tick?
    ///
    /// Always when the previous answer is unknown — that is either the
    /// first Absent tick or the tick right after a consumer detached,
    /// which is exactly when an operator asks "who is holding it" —
    /// and otherwise no more often than [`EXTERNAL_CENSUS_INTERVAL`].
    fn census_due(&self) -> bool {
        match self.last_census {
            None => true,
            Some(_) if self.external_holder.is_none() => true,
            Some(at) => at.elapsed() >= EXTERNAL_CENSUS_INTERVAL,
        }
    }
}

/// One authoritative re-read: compare it with the event-driven verdict,
/// correct the verdict when they disagree, and take the two side
/// readings that make a silent `Absent` interpretable.
///
/// **Ordering matters.** The verdict is snapshotted *before* the probe,
/// because subscribing echoes an event back onto `source` (see
/// [`fluxframe_gst::probe_client_usage`]). Comparing against a verdict
/// the echo already repaired would report zero corrections during
/// exactly the failure this exists to detect. The echo is then drained
/// explicitly so it cannot masquerade as consumer traffic in the
/// silence gauge.
///
/// This function must stay on the detector thread: it writes
/// `state.last_status`, which is plain `&mut`, and the status atomic has
/// exactly one writer by construction. Moving the probe to a thread of
/// its own needs that ownership reworked first.
pub(super) fn resync_tick<S: UsageSource, P: NodeProbe>(
    source: &mut S,
    probe: &P,
    ctx: &DetectorContext<'_>,
    state: &mut DetectorState<'_>,
    resync: &mut ResyncState,
) {
    resync.last_tick = Instant::now();
    ctx.counters.inc_consumer_resync();

    let believed = *state.last_status;
    match probe.client_usage() {
        Ok(reading) => {
            if resync.probe_failing {
                info!(target: "fluxframe::idle", "consumer resync recovered");
                resync.probe_failing = false;
            }
            resync.backoff_until = None;
            // Escalation is per-incident, not for the rest of the run:
            // without this, isolated `EBUSY`s hours apart would keep
            // doubling and silently leave the safety net at the ceiling.
            resync.backoff = resync.backoff_base;
            let echo_disagreed = drain_probe_echo(source, reading, ctx, state, resync);
            if !echo_disagreed {
                reconcile(reading, believed, ctx, state, resync);
            }
            // The census walks `/proc`, which has no upper bound on a
            // busy host — do not start it if shutdown is already asked
            // for, since `Drop` is waiting on this thread.
            if !ctx.stopped.load(Ordering::Acquire) {
                check_external_holder(reading, ctx, resync);
            }
        }
        Err(e) => on_probe_error(&e, ctx, resync),
    }

    check_output_stream(probe, ctx, resync);
    if resync.disabled {
        // The probe just proved permanently unavailable, so the age can
        // never advance again. Leaving the last measured value in place
        // would freeze the gauge at something small — "an event just
        // arrived" — which is the false-healthy reading the sentinel
        // exists to prevent.
        ctx.counters
            .set_consumer_last_external_event_age_secs(CONSUMER_EVENT_AGE_UNSET);
    } else {
        report_event_silence(ctx, resync);
    }
}

/// Consume the event our own subscription just caused, if it is still
/// queued, and report whether it disagreed with what the probe read.
///
/// A disagreement means the truth changed between the probe and the
/// drain. That reading is newer, so it is applied — and the drift
/// comparison is skipped for this tick, because a verdict that was
/// correct until a moment ago is not evidence of a deaf subscription.
///
/// Exactly one event is taken. If a genuine event was queued ahead of
/// the echo and carries the same value, it is consumed in the echo's
/// place and the echo itself reaches the main loop on the next
/// iteration — costing the silence gauge one interval of accuracy, but
/// never a state change.
fn drain_probe_echo<S: UsageSource>(
    source: &mut S,
    reading: ClientUsage,
    ctx: &DetectorContext<'_>,
    state: &mut DetectorState<'_>,
    resync: &mut ResyncState,
) -> bool {
    let echo = match source.wait(0) {
        Ok(Some(echo)) => echo,
        Ok(None) => return false,
        // The subscription is dead. Not fatal here — the next
        // `wait(timeout)` in the main loop surfaces it and the caller
        // falls back — but losing the fact entirely would hide *where*
        // it died.
        Err(e) => {
            debug!(
                target: "fluxframe::idle",
                error = %e,
                "echo drain failed; the subscription looks dead"
            );
            return false;
        }
    };
    if echo == reading {
        return false;
    }
    debug!(
        target: "fluxframe::idle",
        probed = reading.count,
        observed = echo.count,
        "client-usage changed while probing — taking the newer reading"
    );
    resync.note_external_event();
    apply_usage(echo, ctx, state);
    true
}

/// Compare an authoritative reading with the event-driven verdict and
/// act on the difference.
fn reconcile(
    reading: ClientUsage,
    believed: Option<ConsumerStatus>,
    ctx: &DetectorContext<'_>,
    state: &mut DetectorState<'_>,
    resync: &mut ResyncState,
) {
    let observed = if reading.attached() {
        ConsumerStatus::Present
    } else {
        ConsumerStatus::Absent
    };

    // Nothing to disagree with yet: no verdict published, or one that
    // means "I do not know". Publishing is right, counting it as a
    // correction is not — every run would then start at one and the
    // counter would stop meaning "events are being lost".
    if !matches!(
        believed,
        Some(ConsumerStatus::Present | ConsumerStatus::Absent)
    ) {
        debug!(
            target: "fluxframe::idle",
            ?believed,
            ?observed,
            "consumer resync — initial synchronisation"
        );
        apply_usage(reading, ctx, state);
        return;
    }

    if believed == Some(observed) {
        debug!(target: "fluxframe::idle", ?observed, "consumer resync — verdict confirmed");
        resync.absent_streak = 0;
        return;
    }

    warn!(
        target: "fluxframe::idle",
        ?believed,
        ?observed,
        count = reading.count,
        "consumer state drift — the kernel disagrees with the event-driven verdict"
    );

    if observed == ConsumerStatus::Present {
        // Waking up on a false positive costs an idle camera for one
        // interval; staying asleep on a true positive costs the user
        // their camera. Apply immediately.
        ctx.counters.inc_consumer_resync_corrections();
        resync.absent_streak = 0;
        apply_usage(reading, ctx, state);
        return;
    }

    // Downgrades are the expensive direction: `idle.teardown_secs` after
    // this the camera is released, and a client mid-call sees it vanish.
    // Require several consecutive re-reads to agree before paying that.
    resync.absent_streak = resync.absent_streak.saturating_add(1);
    if resync.absent_streak < RESYNC_ABSENT_CONFIRMATIONS {
        debug!(
            target: "fluxframe::idle",
            confirmations = resync.absent_streak,
            required = RESYNC_ABSENT_CONFIRMATIONS,
            "holding the Present verdict until another re-read agrees"
        );
        return;
    }
    ctx.counters.inc_consumer_resync_corrections();
    resync.absent_streak = 0;
    apply_usage(reading, ctx, state);
}

/// Classify a probe failure: permanent faults stop the mechanism,
/// contention backs it off, everything else retries next tick.
fn on_probe_error(e: &io::Error, ctx: &DetectorContext<'_>, resync: &mut ResyncState) {
    ctx.counters.inc_consumer_resync_failures();
    let raw = e.raw_os_error();
    match classify_probe_error(e) {
        ProbeFailure::Permanent => {
            // The node cannot answer this question at all — a
            // non-loopback sink, v4l2loopback < 0.13, or permissions.
            // Retrying every interval would turn the failure counter
            // into an uptime clock.
            warn!(
                target: "fluxframe::idle",
                error = %e,
                errno = ?raw,
                "consumer resync unavailable on this node — disabling it for this run"
            );
            resync.disabled = true;
            ctx.counters
                .set_consumer_last_external_event_age_secs(CONSUMER_EVENT_AGE_UNSET);
            return;
        }
        ProbeFailure::Busy => {
            let step = if resync.backoff.is_zero() {
                RESYNC_BACKOFF_START
            } else {
                resync.backoff
            };
            let next = (step * 2).min(RESYNC_BACKOFF_MAX);
            resync.backoff = next;
            resync.backoff_until = Some(Instant::now() + next);
        }
        // `ProbeFailure` is `#[non_exhaustive]`; anything the driver
        // starts reporting that we have not classified yet gets the
        // conservative treatment — retry next tick, change nothing.
        _ => {}
    }
    if !resync.probe_failing {
        resync.probe_failing = true;
        warn!(
            target: "fluxframe::idle",
            error = %e,
            errno = ?raw,
            backoff_secs = resync.backoff_until.map(|_| resync.backoff.as_secs()),
            "consumer resync failed — the verdict is unverified until it recovers"
        );
    }
}

/// Verify that *we* are still streaming into the loopback.
///
/// Nothing else notices this: `idle_frames_pushed_total` counts frames
/// handed to the pipeline, so it keeps climbing while the driver rejects
/// every consumer with `EIO`.
fn check_output_stream<P: NodeProbe>(
    probe: &P,
    ctx: &DetectorContext<'_>,
    resync: &mut ResyncState,
) {
    let (gauge, down) = match probe.output_state() {
        LoopbackState::ProducerStreaming => (1, Some(false)),
        LoopbackState::ProducerStopped => (0, Some(true)),
        other => {
            debug!(
                target: "fluxframe::idle",
                state = ?other,
                "loopback OUTPUT state unavailable"
            );
            (OUTPUT_STREAM_UNKNOWN, None)
        }
    };
    ctx.counters.set_output_stream_up(gauge);
    let Some(down) = down else { return };
    if resync.output_down == Some(down) {
        return;
    }
    if down {
        ctx.counters.inc_output_stream_down();
        warn!(
            target: "fluxframe::idle",
            "loopback OUTPUT stream is not held — every capture client will get EIO \
             until the output pipeline is rebuilt"
        );
    } else if resync.output_down == Some(true) {
        info!(target: "fluxframe::idle", "loopback OUTPUT stream is held again");
    }
    resync.output_down = Some(down);
}

/// When the driver says nobody is streaming, find out whether anybody is
/// nevertheless *holding* the node.
///
/// That combination is the fingerprint of a client that opened the
/// device and never reached `STREAMON` — usually because another
/// application already owns v4l2loopback's single capture slot. Without
/// this the log cannot tell that case from "nobody wants the camera".
fn check_external_holder(
    reading: ClientUsage,
    ctx: &DetectorContext<'_>,
    resync: &mut ResyncState,
) {
    if reading.attached() {
        // Somebody is streaming; who else holds an fd is not interesting.
        // Clearing the edge tracker also forces a fresh census on the
        // first tick after they detach.
        ctx.counters.set_external_openers(EXTERNAL_OPENERS_UNSET);
        resync.external_holder = None;
        return;
    }
    if !resync.census_due() {
        return;
    }
    resync.last_census = Some(Instant::now());
    match count_external_consumers(ctx.proc_root, ctx.device_basename, ctx.my_pid) {
        WalkOutcome::External { pid } => {
            // The walk short-circuits on the first hit, so this gauge is
            // "at least one", not a count.
            ctx.counters.set_external_openers(1);
            if resync.external_holder != Some(true) {
                // `comm` is set by the other process (`PR_SET_NAME`) and
                // can carry newlines or escapes; it goes into a log line
                // an operator reads during an incident.
                let comm = read_comm(ctx.proc_root, pid)
                    .map(|c| c.escape_default().to_string())
                    .unwrap_or_default();
                warn!(
                    target: "fluxframe::idle",
                    holder_pid = pid,
                    holder_comm = %comm,
                    "loopback is held open by a process that is not streaming — a client \
                     probably failed REQBUFS/STREAMON because the capture slot is taken"
                );
            }
            resync.external_holder = Some(true);
        }
        WalkOutcome::CleanAbsent => {
            ctx.counters.set_external_openers(0);
            resync.external_holder = Some(false);
        }
        // "I may not look" is not "nobody is there" — the mistake that
        // sank the previous detector. Report it as unknown, not as zero.
        outcome => {
            ctx.counters.set_external_openers(EXTERNAL_OPENERS_UNSET);
            debug!(
                target: "fluxframe::idle",
                ?outcome,
                "cannot attribute the loopback's holders"
            );
            resync.external_holder = None;
        }
    }
}

/// Publish how long the node has been event-silent, and say so once when
/// that crosses the threshold.
///
/// A verdict that happens to be correct while the subscription is deaf
/// produces no drift and no correction — this is the only signal that
/// separates it from a healthy quiet node.
fn report_event_silence(ctx: &DetectorContext<'_>, resync: &mut ResyncState) {
    // The gauge only reports real observations, so it stays at the
    // sentinel until an event actually arrives. The *threshold* falls
    // back to the loop's start instead: a subscription that was already
    // deaf when the run began never delivers a first event, and that is
    // precisely a case worth a warning rather than an eternal sentinel.
    let age = if let Some(last) = resync.last_external_event {
        let age = last.elapsed();
        ctx.counters
            .set_consumer_last_external_event_age_secs(age.as_secs());
        age
    } else {
        ctx.counters
            .set_consumer_last_external_event_age_secs(CONSUMER_EVENT_AGE_UNSET);
        resync.started_at.elapsed()
    };
    if age >= resync.stale_after && !resync.stale_warned {
        resync.stale_warned = true;
        warn!(
            target: "fluxframe::idle",
            silent_secs = age.as_secs(),
            threshold_secs = resync.stale_after.as_secs(),
            "no client-usage events for longer than expected — if consumers are \
             attaching, the subscription has gone deaf"
        );
    }
}

/// Scripted [`NodeProbe`], shared with the detector's own tests — the
/// loop tests need it to exercise the resync path end to end.
#[cfg(test)]
pub(super) mod fixtures {
    use super::{ClientUsage, LoopbackState, NodeProbe, io};

    /// Each `client_usage` call pops one entry; once drained it keeps
    /// repeating the last answer, so a test can script the interesting
    /// tick and let the rest of the run coast.
    pub(in crate::idle) struct FakeProbe {
        usage: std::cell::RefCell<std::collections::VecDeque<io::Result<ClientUsage>>>,
        last: std::cell::Cell<Option<u32>>,
        pub(in crate::idle) output: std::cell::Cell<LoopbackState>,
        pub(in crate::idle) calls: std::cell::Cell<u32>,
    }

    impl FakeProbe {
        pub(in crate::idle) fn new(script: Vec<io::Result<ClientUsage>>) -> Self {
            Self {
                usage: std::cell::RefCell::new(script.into()),
                last: std::cell::Cell::new(None),
                output: std::cell::Cell::new(LoopbackState::ProducerStreaming),
                calls: std::cell::Cell::new(0),
            }
        }

        /// A probe that never gets asked anything — for the paths where
        /// the resync schedule is `Off`.
        pub(in crate::idle) fn silent() -> Self {
            Self::new(Vec::new())
        }
    }

    impl NodeProbe for FakeProbe {
        fn client_usage(&self) -> io::Result<ClientUsage> {
            self.calls.set(self.calls.get() + 1);
            match self.usage.borrow_mut().pop_front() {
                Some(Ok(u)) => {
                    self.last.set(Some(u.count));
                    Ok(u)
                }
                Some(Err(e)) => Err(e),
                None => match self.last.get() {
                    Some(count) => Ok(ClientUsage { count }),
                    None => Err(io::Error::from(io::ErrorKind::TimedOut)),
                },
            }
        }

        fn output_state(&self) -> LoopbackState {
            self.output.get()
        }
    }

    #[expect(
        clippy::unnecessary_wraps,
        reason = "the Result is the point: scripts mix successful readings with probe \
                  failures, mirroring `usage()` on the event side"
    )]
    pub(in crate::idle) fn probe_ok(count: u32) -> io::Result<ClientUsage> {
        Ok(ClientUsage { count })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use fluxframe_core::Counters;

    use super::fixtures::FakeProbe;
    use super::{
        ClientUsage, Duration, LoopbackState, RESYNC_BACKOFF_MAX, ResyncSchedule, ResyncState,
        check_external_holder, check_output_stream, io, on_probe_error, report_event_silence,
    };
    use crate::idle::detector::tests::{make_proc_entry, with_ctx};

    #[test]
    fn the_wall_clock_schedule_fires_only_after_its_interval() {
        // The production schedule. Every other resync test drives the
        // test-only iteration variant, so without this the dispatcher
        // branch that actually ships is never executed: "the tick never
        // fires" would pass the whole suite.
        let never = ResyncState::new(
            ResyncSchedule::every(Duration::from_secs(3600)),
            Duration::from_secs(5),
        );
        assert!(!never.due(), "a fresh state must wait out its interval");

        let now = ResyncState::new(
            ResyncSchedule::every(Duration::from_millis(1)),
            Duration::from_secs(5),
        );
        std::thread::sleep(Duration::from_millis(2));
        assert!(now.due());
    }

    #[test]
    fn a_zero_interval_is_not_a_schedule() {
        // Guards the invariant at the type: `Every(ZERO)` would re-open
        // the device on every loop iteration, twenty times a second.
        let state = ResyncState::new(
            ResyncSchedule::every(Duration::ZERO),
            Duration::from_secs(5),
        );
        assert!(matches!(state.schedule, ResyncSchedule::Off));
        assert!(!state.due());
    }

    #[test]
    fn a_disabled_resync_never_fires_again() {
        let mut state = ResyncState::new(
            ResyncSchedule::every(Duration::from_millis(1)),
            Duration::from_secs(5),
        );
        state.disabled = true;
        std::thread::sleep(Duration::from_millis(2));
        assert!(!state.due());
    }

    #[test]
    fn busy_backs_off_and_recovers_to_the_configured_interval() {
        // The regression this guards: `backoff` used to keep its
        // escalated value after a success, so isolated EBUSYs hours
        // apart doubled from wherever the previous incident left off
        // and pinned the safety net at the five-minute ceiling.
        // Spelled numerically because this crate does not depend on
        // `libc`.
        const EBUSY: i32 = 16;
        let counters = Counters::new();
        let interval = Duration::from_secs(30);
        let mut resync = ResyncState::new(ResyncSchedule::every(interval), Duration::from_secs(5));
        let busy = io::Error::from_raw_os_error(EBUSY);

        with_ctx(Path::new("/proc"), &counters, |ctx| {
            on_probe_error(&busy, ctx, &mut resync);
            let first = resync.backoff;
            assert!(resync.backoff_until.is_some(), "EBUSY must back off");
            assert!(!resync.due(), "the backoff deadline suppresses the tick");

            on_probe_error(&busy, ctx, &mut resync);
            assert!(
                resync.backoff > first,
                "consecutive EBUSY must escalate: {first:?} -> {:?}",
                resync.backoff
            );
            assert!(resync.backoff <= RESYNC_BACKOFF_MAX);
        });

        // A success ends the incident; the next one starts over.
        resync.backoff_until = None;
        resync.backoff = resync.backoff_base;
        assert_eq!(resync.backoff, interval);
        assert_eq!(counters.snapshot().consumer_resync_failures_total, 2);
    }

    #[test]
    fn a_permanent_failure_is_not_retried_and_takes_the_gauge_with_it() {
        const ENOTTY: i32 = 25;
        let counters = Counters::new();
        let mut resync = ResyncState::new(
            ResyncSchedule::every(Duration::from_millis(1)),
            Duration::from_secs(5),
        );
        with_ctx(Path::new("/proc"), &counters, |ctx| {
            on_probe_error(&io::Error::from_raw_os_error(ENOTTY), ctx, &mut resync);
        });
        assert!(resync.disabled);
        std::thread::sleep(Duration::from_millis(2));
        assert!(!resync.due());
        assert_eq!(
            counters.snapshot().consumer_last_external_event_age_secs,
            fluxframe_core::CONSUMER_EVENT_AGE_UNSET
        );
    }

    #[test]
    fn event_silence_is_measured_only_from_real_events() {
        let counters = Counters::new();
        let mut resync = ResyncState::new(ResyncSchedule::Off, Duration::from_secs(5));
        resync.stale_after = Duration::ZERO;

        with_ctx(Path::new("/proc"), &counters, |ctx| {
            // Nothing observed yet: the gauge must stay at the sentinel
            // rather than claim "an event just arrived".
            report_event_silence(ctx, &mut resync);
            assert_eq!(
                counters.snapshot().consumer_last_external_event_age_secs,
                fluxframe_core::CONSUMER_EVENT_AGE_UNSET
            );
            // ...but the threshold still runs off the loop's start, so a
            // subscription that was already deaf at startup is reported.
            assert!(resync.stale_warned);

            resync.note_external_event();
            assert!(!resync.stale_warned, "a real event re-arms the warning");
            report_event_silence(ctx, &mut resync);
            assert_ne!(
                counters.snapshot().consumer_last_external_event_age_secs,
                fluxframe_core::CONSUMER_EVENT_AGE_UNSET
            );
        });
    }

    #[test]
    fn an_external_event_breaks_a_run_of_agreeing_reads() {
        // Two re-reads separated by "a client is streaming" are not two
        // consecutive re-reads, and must not release the camera.
        let mut resync = ResyncState::new(ResyncSchedule::Off, Duration::from_secs(5));
        resync.absent_streak = 1;
        resync.note_external_event();
        assert_eq!(resync.absent_streak, 0);
    }

    #[test]
    fn the_proc_census_is_throttled_between_state_changes() {
        // The walk is unbounded on a busy host, and the question it
        // answers changes on human timescales — running it on every
        // tick would put it in the idle steady state.
        let dir = tempfile::tempdir().expect("tempdir");
        make_proc_entry(dir.path(), 4321, 7, Path::new("/dev/video10"));
        let counters = Counters::new();
        let mut resync = ResyncState::new(ResyncSchedule::Off, Duration::from_secs(5));

        with_ctx(dir.path(), &counters, |ctx| {
            assert!(resync.census_due(), "the first Absent tick must census");
            check_external_holder(ClientUsage { count: 0 }, ctx, &mut resync);
            assert_eq!(counters.snapshot().external_openers, 1);
            assert!(
                !resync.census_due(),
                "a second tick moments later must not walk /proc again"
            );

            // A consumer attaching and detaching invalidates the answer.
            check_external_holder(ClientUsage { count: 1 }, ctx, &mut resync);
            assert!(resync.census_due(), "a state change forces a fresh census");
        });
    }

    #[test]
    fn output_stream_loss_is_counted_once_per_edge() {
        let counters = Counters::new();
        let probe = FakeProbe::silent();
        probe.output.set(LoopbackState::ProducerStopped);
        let mut resync = ResyncState::new(ResyncSchedule::Off, Duration::from_secs(5));
        with_ctx(Path::new("/proc"), &counters, |ctx| {
            check_output_stream(&probe, ctx, &mut resync);
            check_output_stream(&probe, ctx, &mut resync);
            assert_eq!(counters.snapshot().output_stream_up, 0);
            assert_eq!(
                counters.snapshot().output_stream_down_total,
                1,
                "a persistent fault is one edge, not one per tick"
            );

            probe.output.set(LoopbackState::ProducerStreaming);
            check_output_stream(&probe, ctx, &mut resync);
            assert_eq!(counters.snapshot().output_stream_up, 1);
            assert_eq!(counters.snapshot().output_stream_down_total, 1);
        });
    }

    #[test]
    fn an_unreadable_output_state_is_unknown_not_down() {
        // A non-loopback sink must not look like a broken one.
        let counters = Counters::new();
        let probe = FakeProbe::silent();
        probe
            .output
            .set(LoopbackState::Unreadable(io::ErrorKind::NotFound));
        let mut resync = ResyncState::new(ResyncSchedule::Off, Duration::from_secs(5));
        with_ctx(Path::new("/proc"), &counters, |ctx| {
            check_output_stream(&probe, ctx, &mut resync);
        });
        assert_eq!(
            counters.snapshot().output_stream_up,
            fluxframe_core::OUTPUT_STREAM_UNKNOWN
        );
        assert_eq!(counters.snapshot().output_stream_down_total, 0);
    }

    #[test]
    fn a_holder_that_is_not_streaming_is_surfaced() {
        // The scenario the log could not previously distinguish from
        // "nobody wants the camera": somebody holds the node but never
        // reached STREAMON.
        let dir = tempfile::tempdir().expect("tempdir");
        make_proc_entry(dir.path(), 4321, 7, Path::new("/dev/video10"));
        let counters = Counters::new();
        let mut resync = ResyncState::new(ResyncSchedule::Off, Duration::from_secs(5));
        with_ctx(dir.path(), &counters, |ctx| {
            check_external_holder(ClientUsage { count: 0 }, ctx, &mut resync);
        });
        assert_eq!(counters.snapshot().external_openers, 1);
    }

    #[test]
    fn nobody_holding_the_node_reads_as_zero_not_unknown() {
        let dir = tempfile::tempdir().expect("tempdir");
        make_proc_entry(dir.path(), 4321, 7, Path::new("/dev/video0"));
        let counters = Counters::new();
        let mut resync = ResyncState::new(ResyncSchedule::Off, Duration::from_secs(5));
        with_ctx(dir.path(), &counters, |ctx| {
            check_external_holder(ClientUsage { count: 0 }, ctx, &mut resync);
        });
        assert_eq!(counters.snapshot().external_openers, 0);
    }

    #[test]
    fn holders_are_not_walked_while_somebody_is_streaming() {
        // With a consumer attached the question is meaningless, and the
        // gauge must say "not determined" rather than "nobody".
        let dir = tempfile::tempdir().expect("tempdir");
        make_proc_entry(dir.path(), 4321, 7, Path::new("/dev/video10"));
        let counters = Counters::new();
        let mut resync = ResyncState::new(ResyncSchedule::Off, Duration::from_secs(5));
        with_ctx(dir.path(), &counters, |ctx| {
            check_external_holder(ClientUsage { count: 1 }, ctx, &mut resync);
        });
        assert_eq!(
            counters.snapshot().external_openers,
            fluxframe_core::EXTERNAL_OPENERS_UNSET
        );
    }
}
