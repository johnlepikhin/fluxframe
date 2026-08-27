//! Realtime metrics: counters and latency histograms.
//!
//! Wire-compatible with §29 of the spec (periodic
//! `fps=… dropped=… latency_p50=…ms latency_p95=…ms inference_p50=…ms`
//! reporter).  This module owns the metric data types ([`Counters`],
//! [`CounterValues`], the histograms, [`MetricsSnapshot`]) **and** the
//! single structured-log emitter [`emit_metrics_line`] — kept here, next
//! to `CounterValues`, so the field list is exhaustively destructured in
//! one place and the compiler flags any counter added but never emitted.
//! The supervisor wiring (incrementing counters, recording latencies) and
//! the periodic CLI reporter live in `fluxframe-cli` and call
//! [`emit_metrics_line`].
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::Mutex;

/// Sentinel for `consumer_status` before the detector has published a
/// verdict. Distinct from every `ConsumerStatus` discriminant (0..=2).
pub const CONSUMER_STATUS_UNSET: u64 = u64::MAX;

/// Sentinel for `consumer_source` before a detection path has been
/// chosen — including runs with no detector at all (idle disabled, or a
/// sink that has none).
pub const CONSUMER_SOURCE_UNSET: u64 = u64::MAX;

/// Sentinel for `output_stream_up`: the producer's own OUTPUT stream has
/// not been observed. Distinct from `0` ("observed, and it is down"),
/// which is an actionable failure — the resync tick that reads it does
/// not run on every presence source, and sysfs is not always readable.
pub const OUTPUT_STREAM_UNKNOWN: u64 = u64::MAX;

/// Sentinel for `consumer_last_external_event_age_secs` before any
/// externally-attributable client-usage event has been seen. Zero would
/// read as "an event just arrived", which is the exact false-healthy
/// signal this gauge exists to remove.
pub const CONSUMER_EVENT_AGE_UNSET: u64 = u64::MAX;

/// Sentinel for `external_openers`: nobody has walked `/proc` yet, so
/// "how many other processes hold the loopback" is unknown rather than
/// zero.
pub const EXTERNAL_OPENERS_UNSET: u64 = u64::MAX;

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
    // Stage 6 (backend abstraction): one-way sticky transitions from a
    // GPU/accelerator backend to the CPU secondary, recorded by the
    // sticky-fallback decorators in `fluxframe-effects::backend`.
    // Scalar — no labels — because today there is a single `from` source
    // (any future accelerator) and a single `to` sink (CPU). When a
    // second `from` variant appears (e.g. NPU vs iGPU), revisit the
    // counter shape rather than encoding the source as a string.
    inference_runtime_fallback_gpu_to_cpu: AtomicU64,
    blur_runtime_fallback_gpu_to_cpu: AtomicU64,
    // Stage 15 idle-mode counters. `idle_entered_total` increments on
    // every Active → Idle edge; `idle_frames_pushed_total` is the count
    // of placeholder frames emitted to the output during idle steady
    // state. `deep_idle_entered_total` is retained but always 0 — the
    // `DeepIdle` state was removed in Stage 16 (see the deprecated
    // `inc_deep_idle_entered`); the field stays for wire-compat.
    //
    // Scope note on `idle_frames_pushed_total`: it counts frames handed
    // to the output pipeline, not writes that reached the device. A sink
    // that stopped holding the driver's OUTPUT stream keeps this counter
    // growing while every capture client gets `EIO` — `output_stream_up`
    // is the signal that catches that case.
    idle_entered_total: AtomicU64,
    deep_idle_entered_total: AtomicU64,
    idle_frames_pushed_total: AtomicU64,
    // Stage 16 input-supervisor counters. `input_acquire_attempts_total`
    // increments each time the supervisor spawns an acquire (build+start
    // of the camera input); `input_acquire_failures_total` on each failed
    // acquire (busy/absent camera). A growing gap between the two with a
    // flat `frames_in` means the camera is contended — the loopback keeps
    // streaming the placeholder meanwhile.
    input_acquire_attempts_total: AtomicU64,
    input_acquire_failures_total: AtomicU64,
    // Consumer-wake + hotplug counters. `resume_failures_total` increments
    // each time an idle→active resume fails to re-acquire the camera input
    // (which restarts the chain; under `device = "auto"` that re-enumerates
    // the capture device on the next pass). `unattributable_wakes_total`
    // increments each time the consumer detector wakes on an open of the
    // loopback it could not attribute via `/proc` (EACCES — e.g. a sandboxed
    // browser).
    resume_failures_total: AtomicU64,
    unattributable_wakes_total: AtomicU64,
    // Effective size of the effect-processing rayon pool. A gauge (set
    // once at startup), not a monotonic counter — lets an operator
    // confirm the deliberate low thread cap took effect and spot
    // oversubscription against `inference_p95`.
    processing_threads: AtomicU64,
    // Consumer-presence observability. The detector's verdict used to be
    // invisible in steady state — a status latched to `Present` for a
    // whole run looked exactly like a genuinely busy camera, which is how
    // a permanently-stuck idle detector went unnoticed. These make the
    // verdict, its provenance and its liveness all readable from one
    // metrics line.
    //
    // All three of `consumer_status` / `consumer_clients` /
    // `consumer_source` are gauges:
    //
    // * `consumer_status` mirrors `ConsumerStatus` (0 = Absent,
    //   1 = Present, 2 = Unknown), or [`CONSUMER_STATUS_UNSET`]
    //   (`u64::MAX`) before the detector has published a first verdict.
    // * `consumer_source` records which detection path is live (0 =
    //   kernel client-usage events, 1 = inotify + `/proc`, 2 = `/proc`
    //   polling, 3 = detection disabled), or [`CONSUMER_SOURCE_UNSET`]
    //   (`u64::MAX`) before a path has been chosen — so a silent
    //   degradation to the fallback path cannot masquerade as a healthy
    //   run. See `set_consumer_source`.
    // * `consumer_clients` is the absolute count of capture clients for
    //   the paths that can supply one (only the kernel-event source
    //   today); it stays at 0 for every other path. It has no sentinel:
    //   0 already reads as "nobody, or not counted here", and
    //   `consumer_source` says which of the two it is.
    //
    // The two sentinels exist because zero is a meaningful value for
    // both fields (Absent / the authoritative kernel source), so a run
    // that has observed nothing yet — or has no detector at all — would
    // otherwise print "no consumer, kernel events live".
    // `consumer_transitions_total` is the liveness signal: a flat counter
    // with a non-zero uptime means the detector stopped observing.
    consumer_status: AtomicU64,
    consumer_clients: AtomicU64,
    consumer_source: AtomicU64,
    consumer_transitions_total: AtomicU64,
    // Frames composited and published while the detector reported
    // `Absent`. Every one of those ran ML segmentation and the effect
    // chain for nobody.
    //
    // Scope note: this catches the state machine failing to act on an
    // `Absent` verdict, not a detector that never produces one. A
    // detector latched to `Present` keeps this at zero — the signal for
    // that failure is a flat `consumer_transitions_total` across a long
    // uptime. Steady-state expectation is a small non-zero value (the
    // `idle.teardown_secs` grace window) that stops growing.
    frames_out_while_no_consumer_total: AtomicU64,
    // Mid-run input recovery. `input_reacquire_total` counts attempts to
    // re-open the camera without tearing the loopback down;
    // `input_reacquire_failures_total` those that failed. `input_down_ms_total`
    // accumulates wall time with no valid input inside a run.
    //
    // Note this is deliberately NOT the same quantity as "time the
    // loopback advertised no CAPTURE caps": that window lives *between*
    // runs, outlives any per-run metrics bundle, and is tracked
    // separately by the auto-input supervisor.
    input_reacquire_total: AtomicU64,
    input_reacquire_failures_total: AtomicU64,
    input_down_ms_total: AtomicU64,
    // Consumer-presence resync. The kernel client-usage subscription is
    // edge-driven: one event the driver never queues (or never delivers)
    // latches the verdict for the rest of the run, and the camera stays
    // down until the daemon is restarted. The detector therefore re-reads
    // the absolute value from the driver on a timer, and these three make
    // that mechanism auditable:
    //
    // * `consumer_resync_total` is the liveness proof — a flat counter
    //   over a long uptime means the tick itself stopped running, which
    //   is otherwise indistinguishable from "nothing ever drifted".
    // * `consumer_resync_corrections_total` counts re-reads that
    //   disagreed with the event-driven verdict. Non-zero means events
    //   are being lost; the initial sync out of `Unknown` deliberately
    //   does NOT count here.
    // * `consumer_resync_failures_total` counts re-reads that could not
    //   be taken at all (no free opener slot, permissions, a driver
    //   without the event).
    consumer_resync_total: AtomicU64,
    consumer_resync_corrections_total: AtomicU64,
    consumer_resync_failures_total: AtomicU64,
    // Age of the most recent client-usage event NOT attributable to our
    // own resync probe, or [`CONSUMER_EVENT_AGE_UNSET`].
    //
    // The probe's subscription makes the driver queue an event to every
    // subscribed file handle on the node — including the detector's own
    // long-lived watch. Counting that echo as traffic would pin this
    // gauge below the resync interval forever and erase the one symptom
    // that identifies a deaf subscription ("no events for 49 minutes").
    // The detector drains its own echo before updating this.
    consumer_last_external_event_age_secs: AtomicU64,
    // Does the producer still hold the loopback's OUTPUT stream? `1` yes,
    // `0` no (every capture client gets `EIO` — see the scope note on
    // `idle_frames_pushed_total`), [`OUTPUT_STREAM_UNKNOWN`] when it has
    // not been observed. `output_stream_down_total` counts the down
    // edges, so a fault that flapped between two metrics lines still
    // leaves a trace.
    output_stream_up: AtomicU64,
    output_stream_down_total: AtomicU64,
    // Whether anybody other than us holds the loopback node open — `1`
    // for "at least one" (the walk stops at the first hit, so it is not
    // a count), `0` for an authoritative nobody, or
    // [`EXTERNAL_OPENERS_UNSET`] when the walk could not tell. `1` while
    // `consumer_status` reads Absent is the fingerprint of a client that
    // opened the device but never reached `STREAMON` — typically because
    // another application already owns the single capture slot.
    external_openers: AtomicU64,
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
            inference_runtime_fallback_gpu_to_cpu: AtomicU64::new(0),
            blur_runtime_fallback_gpu_to_cpu: AtomicU64::new(0),
            idle_entered_total: AtomicU64::new(0),
            deep_idle_entered_total: AtomicU64::new(0),
            idle_frames_pushed_total: AtomicU64::new(0),
            input_acquire_attempts_total: AtomicU64::new(0),
            input_acquire_failures_total: AtomicU64::new(0),
            resume_failures_total: AtomicU64::new(0),
            unattributable_wakes_total: AtomicU64::new(0),
            processing_threads: AtomicU64::new(0),
            consumer_status: AtomicU64::new(CONSUMER_STATUS_UNSET),
            consumer_clients: AtomicU64::new(0),
            consumer_source: AtomicU64::new(CONSUMER_SOURCE_UNSET),
            consumer_transitions_total: AtomicU64::new(0),
            frames_out_while_no_consumer_total: AtomicU64::new(0),
            input_reacquire_total: AtomicU64::new(0),
            input_reacquire_failures_total: AtomicU64::new(0),
            input_down_ms_total: AtomicU64::new(0),
            consumer_resync_total: AtomicU64::new(0),
            consumer_resync_corrections_total: AtomicU64::new(0),
            consumer_resync_failures_total: AtomicU64::new(0),
            consumer_last_external_event_age_secs: AtomicU64::new(CONSUMER_EVENT_AGE_UNSET),
            output_stream_up: AtomicU64::new(OUTPUT_STREAM_UNKNOWN),
            output_stream_down_total: AtomicU64::new(0),
            external_openers: AtomicU64::new(EXTERNAL_OPENERS_UNSET),
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

    /// Increment `inference_runtime_fallback_gpu_to_cpu` by one.
    ///
    /// Called by the inference sticky-fallback decorator the first
    /// (and only) time a GPU-side inference engine errors out and the
    /// effect transitions to the CPU secondary for the remainder of the
    /// session.
    #[inline]
    pub fn inc_inference_runtime_fallback_gpu_to_cpu(&self) {
        self.inference_runtime_fallback_gpu_to_cpu
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `blur_runtime_fallback_gpu_to_cpu` by one.  Counterpart
    /// of [`Counters::inc_inference_runtime_fallback_gpu_to_cpu`] for the
    /// blur backend.
    #[inline]
    pub fn inc_blur_runtime_fallback_gpu_to_cpu(&self) {
        self.blur_runtime_fallback_gpu_to_cpu
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `idle_entered_total` — fires on the supervisor's
    /// Active → Idle transition.
    #[inline]
    pub fn inc_idle_entered(&self) {
        self.idle_entered_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `deep_idle_entered_total`.
    ///
    /// Deprecated since Stage 16: the `DeepIdle` state was removed, so
    /// nothing calls this anymore and `deep_idle_entered_total` stays 0.
    /// The counter and this method are retained so existing dashboards
    /// that read the field keep working (they now read a constant 0).
    #[inline]
    #[deprecated(note = "DeepIdle state removed in Stage 16; counter is always 0")]
    pub fn inc_deep_idle_entered(&self) {
        self.deep_idle_entered_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `idle_frames_pushed_total` — fires every placeholder
    /// frame the worker emits while in Idle.
    #[inline]
    pub fn inc_idle_frames_pushed(&self) {
        self.idle_frames_pushed_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `input_acquire_attempts_total` — fires each time the
    /// supervisor spawns an input-acquire (build+start of the camera).
    #[inline]
    pub fn inc_input_acquire_attempts(&self) {
        self.input_acquire_attempts_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `input_acquire_failures_total` — fires on each failed
    /// acquire (camera busy/absent), before the supervisor backs off.
    #[inline]
    pub fn inc_input_acquire_failures(&self) {
        self.input_acquire_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `resume_failures_total` — fires each time an idle→active
    /// resume fails to re-acquire the camera input.
    #[inline]
    pub fn inc_resume_failures(&self) {
        self.resume_failures_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `unattributable_wakes_total` — fires when the consumer
    /// detector wakes on a loopback open it could not attribute via
    /// `/proc` (EACCES: sandboxed/privileged consumer).
    #[inline]
    pub fn inc_unattributable_wakes(&self) {
        self.unattributable_wakes_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record the effective effect-processing rayon pool size.
    ///
    /// A gauge, not a counter: set once at startup from
    /// `rayon::current_num_threads()` so an operator can confirm the
    /// deliberate low cap (see `cap_rayon_pool`) took effect.
    #[inline]
    pub fn set_processing_threads(&self, threads: u64) {
        self.processing_threads.store(threads, Ordering::Relaxed);
    }

    /// Record the consumer detector's current verdict and bump the
    /// transition counter.
    ///
    /// `status` uses the detector's own encoding (0 = Absent,
    /// 1 = Present, 2 = Unknown) so the two cannot drift apart; callers
    /// pass the same byte they publish on the shared atomic.  Until the
    /// first call the gauge reads [`CONSUMER_STATUS_UNSET`], which is
    /// outside that range by construction.
    ///
    /// Call this only on an actual change: the transition counter is the
    /// liveness signal, and bumping it on every poll would destroy its
    /// meaning.
    #[inline]
    pub fn set_consumer_status(&self, status: u8) {
        self.consumer_status
            .store(u64::from(status), Ordering::Relaxed);
        self.consumer_transitions_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record the absolute number of capture clients, for detection
    /// paths that can supply one (the kernel client-usage source today).
    /// A gauge; paths that only know "someone / nobody" leave it at
    /// zero, as does a run before any observation — there is no
    /// sentinel here, read `consumer_source` to tell the cases apart.
    #[inline]
    pub fn set_consumer_clients(&self, clients: u64) {
        self.consumer_clients.store(clients, Ordering::Relaxed);
    }

    /// Record which consumer-detection path is live.
    ///
    /// The encoding is owned by the detector; the four values in use
    /// are `0` = kernel client-usage events, `1` = inotify + `/proc`
    /// walk, `2` = `/proc` polling fallback, `3` = detection disabled by
    /// config.  Until one is chosen — including for the whole of a run
    /// with no detector at all — the gauge reads
    /// [`CONSUMER_SOURCE_UNSET`].  Kept as a bare integer here so
    /// `fluxframe-core` does not grow a dependency on the supervisor's
    /// detector types.
    #[inline]
    pub fn set_consumer_source(&self, source: u64) {
        self.consumer_source.store(source, Ordering::Relaxed);
    }

    /// Increment `consumer_resync_total` — one authoritative re-read of
    /// the driver's capture-usage value was taken.
    ///
    /// Bump this on every tick, successful or not: it is the liveness
    /// proof for the resync mechanism itself, and a mechanism that only
    /// counts its successes cannot report that it stopped running.
    #[inline]
    pub fn inc_consumer_resync(&self) {
        self.consumer_resync_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `consumer_resync_corrections_total` — a re-read
    /// disagreed with the event-driven verdict.
    ///
    /// Only for a genuine disagreement between two verdicts. The first
    /// sync out of `Unknown` (or out of "nothing published yet") is not
    /// a correction: counting it would put a floor of one on every run
    /// and destroy the counter's meaning as "events are being lost".
    #[inline]
    pub fn inc_consumer_resync_corrections(&self) {
        self.consumer_resync_corrections_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `consumer_resync_failures_total` — a re-read could not
    /// be taken (no free opener slot, permissions, or a driver without
    /// the client-usage event).
    #[inline]
    pub fn inc_consumer_resync_failures(&self) {
        self.consumer_resync_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record how long ago the last *externally-caused* client-usage
    /// event arrived, or [`CONSUMER_EVENT_AGE_UNSET`] if none has.
    ///
    /// The caller is responsible for excluding the echo of its own
    /// resync probe — see the field comment. A value that keeps growing
    /// while consumers come and go is the signature of a subscription
    /// that has gone deaf.
    #[inline]
    pub fn set_consumer_last_external_event_age_secs(&self, age_secs: u64) {
        self.consumer_last_external_event_age_secs
            .store(age_secs, Ordering::Relaxed);
    }

    /// Record whether the producer still holds the loopback's OUTPUT
    /// stream: `1` up, `0` down, [`OUTPUT_STREAM_UNKNOWN`] unobserved.
    ///
    /// `0` is actionable on its own: with no OUTPUT stream the driver
    /// fails every capture `STREAMON` with `EIO`, so clients see a
    /// broken camera while this daemon still reports healthy frame
    /// counts.
    #[inline]
    pub fn set_output_stream_up(&self, state: u64) {
        self.output_stream_up.store(state, Ordering::Relaxed);
    }

    /// Increment `output_stream_down_total` — one transition from
    /// "producer is streaming" to "producer is not".
    ///
    /// Call on the edge, not on every observation: the gauge says what
    /// is true now, this says how often it broke, and a per-observation
    /// bump would conflate the two.
    #[inline]
    pub fn inc_output_stream_down(&self) {
        self.output_stream_down_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record how many processes other than us hold the loopback node
    /// open, or [`EXTERNAL_OPENERS_UNSET`] when that was not determined.
    #[inline]
    pub fn set_external_openers(&self, openers: u64) {
        self.external_openers.store(openers, Ordering::Relaxed);
    }

    /// Increment `frames_out_while_no_consumer_total` — one composited
    /// frame published while the detector reported `Absent`.
    #[inline]
    pub fn inc_frames_out_while_no_consumer(&self) {
        self.frames_out_while_no_consumer_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `input_reacquire_total` — one attempt to recover the
    /// camera in-place, without tearing the loopback output down.
    #[inline]
    pub fn inc_input_reacquire(&self) {
        self.input_reacquire_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment `input_reacquire_failures_total`.
    #[inline]
    pub fn inc_input_reacquire_failures(&self) {
        self.input_reacquire_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Add a completed no-input interval, in milliseconds.
    #[inline]
    pub fn add_input_down_ms(&self, ms: u64) {
        if ms != 0 {
            self.input_down_ms_total.fetch_add(ms, Ordering::Relaxed);
        }
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
            inference_runtime_fallback_gpu_to_cpu: self
                .inference_runtime_fallback_gpu_to_cpu
                .load(Ordering::Relaxed),
            blur_runtime_fallback_gpu_to_cpu: self
                .blur_runtime_fallback_gpu_to_cpu
                .load(Ordering::Relaxed),
            idle_entered_total: self.idle_entered_total.load(Ordering::Relaxed),
            deep_idle_entered_total: self.deep_idle_entered_total.load(Ordering::Relaxed),
            idle_frames_pushed_total: self.idle_frames_pushed_total.load(Ordering::Relaxed),
            input_acquire_attempts_total: self.input_acquire_attempts_total.load(Ordering::Relaxed),
            input_acquire_failures_total: self.input_acquire_failures_total.load(Ordering::Relaxed),
            resume_failures_total: self.resume_failures_total.load(Ordering::Relaxed),
            unattributable_wakes_total: self.unattributable_wakes_total.load(Ordering::Relaxed),
            processing_threads: self.processing_threads.load(Ordering::Relaxed),
            consumer_status: self.consumer_status.load(Ordering::Relaxed),
            consumer_clients: self.consumer_clients.load(Ordering::Relaxed),
            consumer_source: self.consumer_source.load(Ordering::Relaxed),
            consumer_transitions_total: self.consumer_transitions_total.load(Ordering::Relaxed),
            frames_out_while_no_consumer_total: self
                .frames_out_while_no_consumer_total
                .load(Ordering::Relaxed),
            input_reacquire_total: self.input_reacquire_total.load(Ordering::Relaxed),
            input_reacquire_failures_total: self
                .input_reacquire_failures_total
                .load(Ordering::Relaxed),
            input_down_ms_total: self.input_down_ms_total.load(Ordering::Relaxed),
            consumer_resync_total: self.consumer_resync_total.load(Ordering::Relaxed),
            consumer_resync_corrections_total: self
                .consumer_resync_corrections_total
                .load(Ordering::Relaxed),
            consumer_resync_failures_total: self
                .consumer_resync_failures_total
                .load(Ordering::Relaxed),
            consumer_last_external_event_age_secs: self
                .consumer_last_external_event_age_secs
                .load(Ordering::Relaxed),
            output_stream_up: self.output_stream_up.load(Ordering::Relaxed),
            output_stream_down_total: self.output_stream_down_total.load(Ordering::Relaxed),
            external_openers: self.external_openers.load(Ordering::Relaxed),
        }
    }
}

impl Default for Counters {
    fn default() -> Self {
        Self::new()
    }
}

/// Plain-data snapshot of [`Counters`].
///
/// [`Default`] is hand-written rather than derived: the gauges that
/// carry a sentinel must read "not observed" in a snapshot nobody has
/// written to. A derived default would make an empty snapshot claim
/// `output_stream_up = 0` ("the producer's OUTPUT stream is down") and
/// `consumer_last_external_event_age_secs = 0` ("an event just
/// arrived") — the false-healthy/false-alarm pair these sentinels exist
/// to prevent. The values match [`Counters::new`] field for field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// See [`Counters::inc_inference_runtime_fallback_gpu_to_cpu`].
    pub inference_runtime_fallback_gpu_to_cpu: u64,
    /// See [`Counters::inc_blur_runtime_fallback_gpu_to_cpu`].
    pub blur_runtime_fallback_gpu_to_cpu: u64,
    /// See [`Counters::inc_idle_entered`].
    pub idle_entered_total: u64,
    /// See [`Counters::inc_deep_idle_entered`].
    pub deep_idle_entered_total: u64,
    /// See [`Counters::inc_idle_frames_pushed`].
    pub idle_frames_pushed_total: u64,
    /// See [`Counters::inc_input_acquire_attempts`].
    pub input_acquire_attempts_total: u64,
    /// See [`Counters::inc_input_acquire_failures`].
    pub input_acquire_failures_total: u64,
    /// See [`Counters::inc_resume_failures`].
    pub resume_failures_total: u64,
    /// See [`Counters::inc_unattributable_wakes`].
    pub unattributable_wakes_total: u64,
    /// See [`Counters::set_processing_threads`].
    pub processing_threads: u64,
    /// See [`Counters::set_consumer_status`].
    pub consumer_status: u64,
    /// See [`Counters::set_consumer_clients`].
    pub consumer_clients: u64,
    /// See [`Counters::set_consumer_source`].
    pub consumer_source: u64,
    /// See [`Counters::set_consumer_status`].
    pub consumer_transitions_total: u64,
    /// See [`Counters::inc_frames_out_while_no_consumer`].
    pub frames_out_while_no_consumer_total: u64,
    /// See [`Counters::inc_input_reacquire`].
    pub input_reacquire_total: u64,
    /// See [`Counters::inc_input_reacquire_failures`].
    pub input_reacquire_failures_total: u64,
    /// See [`Counters::add_input_down_ms`].
    pub input_down_ms_total: u64,
    /// See [`Counters::inc_consumer_resync`].
    pub consumer_resync_total: u64,
    /// See [`Counters::inc_consumer_resync_corrections`].
    pub consumer_resync_corrections_total: u64,
    /// See [`Counters::inc_consumer_resync_failures`].
    pub consumer_resync_failures_total: u64,
    /// See [`Counters::set_consumer_last_external_event_age_secs`].
    pub consumer_last_external_event_age_secs: u64,
    /// See [`Counters::set_output_stream_up`].
    pub output_stream_up: u64,
    /// See [`Counters::inc_output_stream_down`].
    pub output_stream_down_total: u64,
    /// See [`Counters::set_external_openers`].
    pub external_openers: u64,
}

impl Default for CounterValues {
    fn default() -> Self {
        Counters::new().snapshot()
    }
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
    /// `parking_lot::Mutex` has no poisoning concept, so the lock is
    /// infallible — recording can never be derailed by an unrelated
    /// thread's panic.  Each critical section is one `pop_front` +
    /// `push_back` with no cross-call invariants.
    #[inline]
    pub fn record_us(&self, value: u64) {
        let mut buf = self.samples.lock();
        if buf.len() == self.capacity {
            buf.pop_front();
        }
        buf.push_back(value);
    }

    /// Drain the histogram's current contents into a sorted snapshot
    /// suitable for percentile queries.  The histogram itself is NOT
    /// emptied; recording continues against the live ring.
    ///
    /// Infallible by virtue of `parking_lot::Mutex` (no poisoning).
    #[must_use]
    pub fn snapshot(&self) -> LatencySnapshot {
        let buf = self.samples.lock();
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

/// Rolling-window extras only the periodic reporter has: the frames/drop
/// deltas since the last tick and the window they were observed over.
///
/// The teardown summary — which reports absolute totals, not deltas —
/// passes `None` to [`emit_metrics_line`], and these fields emit as zero.
#[derive(Debug, Clone, Copy, Default)]
pub struct PeriodicExtras {
    /// `frames_out` delta since the previous tick.
    pub frames_out_delta: u64,
    /// `frames_dropped` delta since the previous tick.
    pub dropped_delta: u64,
    /// Wall-clock window the deltas were observed over.
    pub window: Duration,
}

/// Emit one structured metrics `info!` line — the SINGLE source of truth
/// for the metrics field set.
///
/// Both the periodic reporter (`fluxframe-cli::metrics_reporter`) and the
/// supervisor's teardown summary call this, so a newly added counter is
/// emitted from exactly one place with one key. The `snap.counters`
/// destructure below is intentionally exhaustive (no `..`): adding a field
/// to [`CounterValues`] is a compile error here until it is wired into the
/// line, which is the guard against "counter added but never emitted".
///
/// `extras` carries the reporter's rolling-window deltas; the teardown
/// summary passes `None` (the delta/fps fields emit as zero). The line is
/// emitted under the `fluxframe::metrics` target so `RUST_LOG=fluxframe=…`
/// filters keep matching it.
pub fn emit_metrics_line(snap: &MetricsSnapshot, extras: Option<PeriodicExtras>) {
    // Exhaustive by design — see the doc comment. `CounterValues` is
    // `Copy`, so this reads out of the borrowed snapshot without moving it.
    let CounterValues {
        frames_in,
        frames_out,
        frames_dropped,
        fallback_count,
        effect_error_count,
        inference_runtime_fallback_gpu_to_cpu,
        blur_runtime_fallback_gpu_to_cpu,
        idle_entered_total,
        deep_idle_entered_total,
        idle_frames_pushed_total,
        input_acquire_attempts_total,
        input_acquire_failures_total,
        resume_failures_total,
        unattributable_wakes_total,
        processing_threads,
        consumer_status,
        consumer_clients,
        consumer_source,
        consumer_transitions_total,
        frames_out_while_no_consumer_total,
        input_reacquire_total,
        input_reacquire_failures_total,
        input_down_ms_total,
        consumer_resync_total,
        consumer_resync_corrections_total,
        consumer_resync_failures_total,
        consumer_last_external_event_age_secs,
        output_stream_up,
        output_stream_down_total,
        external_openers,
    } = snap.counters;

    let ex = extras.unwrap_or_default();
    let periodic = extras.is_some();
    let fps = round2(compute_fps(ex.frames_out_delta, ex.window));
    let window_secs = round2(ex.window.as_secs_f64());
    // Distinguish the two callers by message so existing log greps keep
    // working; the field set is identical either way.
    let msg = if periodic {
        "metrics tick"
    } else {
        "run metrics"
    };

    tracing::info!(
        target: "fluxframe::metrics",
        fps = fps,
        window_secs = window_secs,
        frames_out_delta = ex.frames_out_delta,
        dropped_delta = ex.dropped_delta,
        frames_in_total = frames_in,
        frames_out_total = frames_out,
        frames_dropped_total = frames_dropped,
        fallback_total = fallback_count,
        effect_err_total = effect_error_count,
        inference_runtime_fallback_gpu_to_cpu = inference_runtime_fallback_gpu_to_cpu,
        blur_runtime_fallback_gpu_to_cpu = blur_runtime_fallback_gpu_to_cpu,
        idle_entered_total = idle_entered_total,
        deep_idle_entered_total = deep_idle_entered_total,
        idle_frames_pushed_total = idle_frames_pushed_total,
        input_acquire_attempts_total = input_acquire_attempts_total,
        input_acquire_failures_total = input_acquire_failures_total,
        resume_failures_total = resume_failures_total,
        unattributable_wakes_total = unattributable_wakes_total,
        processing_threads = processing_threads,
        consumer_status = consumer_status,
        consumer_clients = consumer_clients,
        consumer_source = consumer_source,
        consumer_transitions_total = consumer_transitions_total,
        frames_out_while_no_consumer_total = frames_out_while_no_consumer_total,
        input_reacquire_total = input_reacquire_total,
        input_reacquire_failures_total = input_reacquire_failures_total,
        input_down_ms_total = input_down_ms_total,
        consumer_resync_total = consumer_resync_total,
        consumer_resync_corrections_total = consumer_resync_corrections_total,
        consumer_resync_failures_total = consumer_resync_failures_total,
        consumer_last_external_event_age_secs = consumer_last_external_event_age_secs,
        output_stream_up = output_stream_up,
        output_stream_down_total = output_stream_down_total,
        external_openers = external_openers,
        capture_p50_us = snap.capture.percentile_us(0.5),
        capture_p95_us = snap.capture.percentile_us(0.95),
        inference_p50_us = snap.inference.percentile_us(0.5),
        inference_p95_us = snap.inference.percentile_us(0.95),
        processing_p50_us = snap.processing.percentile_us(0.5),
        processing_p95_us = snap.processing.percentile_us(0.95),
        output_p50_us = snap.output.percentile_us(0.5),
        output_p95_us = snap.output.percentile_us(0.95),
        end_to_end_p50_us = snap.end_to_end.percentile_us(0.5),
        end_to_end_p95_us = snap.end_to_end.percentile_us(0.95),
        "{msg}"
    );
}

/// Frames-per-second from a frame delta and its wall-clock window.
/// Returns `0.0` for zero-length windows (defensive against a clock that
/// did not advance between ticks on coarse-`Instant` systems).
#[inline]
fn compute_fps(frames_delta: u64, dt: Duration) -> f64 {
    let secs = dt.as_secs_f64();
    if secs <= 0.0 {
        0.0
    } else {
        (frames_delta as f64) / secs
    }
}

/// Round to two decimal places. Display-only for the reporter line; never
/// used for a downstream calculation.
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
    fn round2_basic() {
        assert!(approx_eq(round2(23.965_698_241), 23.97));
        assert!(approx_eq(round2(0.0), 0.0));
        assert!(approx_eq(round2(100.0), 100.0));
    }

    #[test]
    fn emit_metrics_line_smoke_both_variants() {
        // Exercises the emit path (exhaustive destructure + info!) without
        // asserting on captured output — just that neither variant panics.
        let snap = MetricsSnapshot::new();
        emit_metrics_line(&snap, None);
        emit_metrics_line(
            &snap,
            Some(PeriodicExtras {
                frames_out_delta: 30,
                dropped_delta: 1,
                window: Duration::from_secs(1),
            }),
        );
    }

    #[test]
    fn counters_default_zero() {
        let c = Counters::new();
        let snap = c.snapshot();
        assert_eq!(snap.frames_in, 0);
        assert_eq!(snap.frames_out, 0);
        assert_eq!(snap.frames_dropped, 0);
        assert_eq!(snap.fallback_count, 0);
        assert_eq!(snap.effect_error_count, 0);
        assert_eq!(snap.inference_runtime_fallback_gpu_to_cpu, 0);
        assert_eq!(snap.blur_runtime_fallback_gpu_to_cpu, 0);
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
        c.inc_inference_runtime_fallback_gpu_to_cpu();
        c.inc_blur_runtime_fallback_gpu_to_cpu();
        c.inc_blur_runtime_fallback_gpu_to_cpu();
        let snap = c.snapshot();
        assert_eq!(snap.frames_in, 5);
        assert_eq!(snap.frames_out, 3);
        assert_eq!(snap.frames_dropped, 1);
        assert_eq!(snap.fallback_count, 7);
        assert_eq!(snap.effect_error_count, 2);
        assert_eq!(snap.inference_runtime_fallback_gpu_to_cpu, 1);
        assert_eq!(snap.blur_runtime_fallback_gpu_to_cpu, 2);
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
