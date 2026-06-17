//! Process-fd consumer presence detector for v4l2loopback devices.
//!
//! Linux-only — `/proc/*/fd` is a Linux interface. On other targets
//! this module is not compiled, and the supervisor (`build_idle_runtime`
//! in `runtime.rs`) gates its wiring through the same `cfg`.
//!
//! ## Why not sysfs?
//!
//! The first Stage 15 implementation polled
//! `/sys/class/video4linux/<videoN>/state`, on the assumption that
//! `"capture"` means "a reader has called `STREAMON`". That was wrong:
//! the v4l2loopback attribute actually reflects the producer's
//! `ready_for_capture` flag — the moment fluxframe itself starts
//! writing frames, `state` reads `"capture"` regardless of whether
//! anyone is reading. The detector pinned `Present` and idle mode
//! never engaged. The clean alternative (`VIDIOC_DQEVENT` +
//! `V4L2_EVENT_PRI_CLIENT_USAGE`) requires raw ioctls, which collide
//! with the workspace `#![forbid(unsafe_code)]`.
//!
//! ## How the inotify + walk hybrid works
//!
//! The detector watches `/dev/videoN` for `IN_OPEN`,
//! `IN_CLOSE_NOWRITE` and `IN_CLOSE_WRITE` via `inotify`. In
//! steady-state idle the cost is one nonblocking `read(2)` returning
//! `EAGAIN` every ~50 ms (the [`SHUTDOWN_POLL_GRANULARITY`] window
//! used so the worker can honour `running` / `stopped`), not a
//! per-event syscall — orders of magnitude cheaper than the prior
//! per-poll `/proc` walk, and the kernel only delivers an actual
//! event when somebody opens or closes the device.
//!
//! Counting events naively is unsound — fluxframe's input pipeline
//! opens and closes the device on its own state transitions, and
//! we have no way to distinguish self-events from external ones at
//! the event level. So on every event the detector walks
//! `/proc/[0-9]+/fd/` via [`count_external_consumers`], which
//! filters out `my_pid` and decides Present/Absent:
//!
//! * one or more external pids hold the fd → `Present`
//! * nobody but fluxframe holds the fd → `Absent`
//! * `/proc` itself is unreadable → `Unknown` (fail-open)
//!
//! Per-pid walk errors — `EACCES` on `/proc/<root-pid>/fd`,
//! `ENOENT` from a process that exited mid-scan — are skipped
//! silently. Steady-state expected on any multi-user / sandboxed
//! system.
//!
//! ## Overflow and fallback
//!
//! On `IN_Q_OVERFLOW` (extremely rare unless the system is under
//! pathological load), the detector does one synchronous walk to
//! resync.
//!
//! When `inotify::init` or `watches().add` fails (e.g. user at
//! `/proc/sys/fs/inotify/max_user_watches`, hard sandbox, or the
//! device node doesn't exist), the dispatcher falls back to the
//! original polling path — a `/proc` walk every
//! `idle.poll_interval_ms`. The walk happens identically; only the
//! wake mechanism differs. Public behaviour and `ConsumerStatus`
//! semantics are unchanged.
//!
//! The dispatcher also falls back on `IN_IGNORED` (e.g. `modprobe
//! -r v4l2loopback` removes the watched node mid-run); polling
//! continues reporting whatever the walk decides until the device
//! comes back. Persistent `read_events` failures (rare — typically
//! `EINTR` storms) publish `Unknown` (fail-open) on the first error
//! and after [`INOTIFY_READ_ERROR_BUDGET`] consecutive errors
//! escalate to a polling fallback so the watch never silently dies.
//!
//! ## RESTAT semantics
//!
//! The polling path rearms the [`LogOnce`] gates every
//! `RESTAT_INTERVAL_POLLS` *poll iterations*; the inotify path
//! rearms every `RESTAT_INTERVAL_POLLS` *walks* (each walk fires on
//! an open/close event). On a busy device the cadence is similar;
//! on a quiet device the inotify path may never rearm — which is a
//! feature, not a bug, since the gates exist to surface
//! activity-time problems.
//!
//! ## Limitations
//!
//! * **Root-owned consumers are invisible.** An unprivileged
//!   fluxframe cannot read `/proc/<root-pid>/fd`, so a root-owned
//!   process consuming `/dev/video10` looks identical to "no
//!   consumer". The pragmatic workaround is to run fluxframe as the
//!   same user as the consumer. This is documented in README.
//! * **`/proc` must be mounted.** Hard containerization that hides
//!   `/proc` collapses the detector to `Unknown` (fail-open).
//!
//! ## Lifecycle
//!
//! [`ConsumerDetector::spawn`] returns a handle owning the worker
//! thread. The thread shuts down on either of two signals:
//!
//! * the caller-provided `running` flag flips to `false` (the
//!   supervisor's shared Ctrl-C atomic), OR
//! * the [`ConsumerDetector`] is dropped (sets an internal `stopped`
//!   flag that nobody else sees).
//!
//! `Drop` joins the worker. The poll loop uses a chunked sleep so
//! shutdown latency is bounded at ~50 ms regardless of how large
//! `poll_interval` is — see [`SHUTDOWN_POLL_GRANULARITY`].
//!
//! [`IdleConfig::poll_interval_ms`]: fluxframe_core::IdleConfig::poll_interval_ms

use std::ffi::{OsStr, OsString};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use fluxframe_gst::output::OutputSink;
use tracing::{debug, info, warn};

use super::state::ConsumerStatus;

/// Number of poll iterations between log-flag re-arms. With the
/// default 250 ms poll cadence this is one re-arm every 15 s —
/// enough to surface a fresh diagnostic if `/proc` access conditions
/// change mid-run without spamming the log.
const RESTAT_INTERVAL_POLLS: u32 = 60;

/// Granularity of the worker's between-poll sleep. The loop wakes at
/// least this often to re-check `running` / `stopped`, capping
/// shutdown latency at this value regardless of `poll_interval`.
const SHUTDOWN_POLL_GRANULARITY: Duration = Duration::from_millis(50);

/// Persistent `inotify::read_events` failures budget. The first
/// failure already published `Unknown` (fail-open). After this many
/// consecutive errors we surface `DetectorExit::ReadFatal` so the
/// dispatcher falls back to polling rather than burning CPU on a
/// permanently broken fd.
const INOTIFY_READ_ERROR_BUDGET: u32 = 16;

/// Stack buffer size for `inotify::read_events`. Each event is a
/// ~16-byte header plus a variable-length name; 1024 bytes holds
/// ~64 events per drain. Steady-state is 0 events; bursts come
/// during pipeline state transitions (a handful of events).
const INOTIFY_EVENT_BUFFER_BYTES: usize = 1024;

/// Variant returned from [`run_detector_inotify`] back to the
/// dispatcher. Each variant maps to a deliberate dispatcher log
/// level: init/add-watch failures are routine on slim hosts and log
/// at `info`; watch-removed and read-fatal are runtime surprises and
/// log at `warn`.
enum DetectorExit {
    /// The watch fired `IN_IGNORED` — the device node was deleted
    /// (e.g. `modprobe -r v4l2loopback`). Fall back to polling so
    /// the detector keeps producing a status until the device
    /// returns.
    WatchRemoved,
    /// `Inotify::init` failed (sandbox, ENOMEM).
    InitFailed(io::Error),
    /// `watches().add` failed (ENOSPC = at `max_user_watches`,
    /// ENOENT when the device node doesn't exist).
    AddFailed(io::Error),
    /// `read_events` produced [`INOTIFY_READ_ERROR_BUDGET`]
    /// consecutive non-recoverable errors. The first error already
    /// published `Unknown` (fail-open).
    ReadFatal(io::Error),
}

impl std::fmt::Display for DetectorExit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WatchRemoved => f.write_str("watch removed (IN_IGNORED)"),
            Self::InitFailed(e) => write!(f, "inotify init failed: {e}"),
            Self::AddFailed(e) => write!(f, "inotify watch add failed: {e}"),
            Self::ReadFatal(e) => write!(f, "inotify read failed repeatedly: {e}"),
        }
    }
}

/// Read-only context bundle threaded through dispatcher and path
/// functions. Reduces argument count and keeps the call sites
/// readable.
struct DetectorContext<'a> {
    proc_root: &'a Path,
    my_pid: u32,
    running: &'a AtomicBool,
    status: &'a AtomicU8,
    stopped: &'a AtomicBool,
    restat_interval_polls: u32,
    device_basename: &'a OsStr,
    /// Per-detector test counter incremented in the inotify path on
    /// successful init + add_watch. See [`ConsumerDetector::inotify_activations_handle`].
    #[cfg(test)]
    inotify_activations: &'a AtomicU64,
}

/// Mutable per-run state threaded between dispatcher and path
/// functions. Pulled out of arg lists so neither path needs
/// `clippy::too_many_arguments`.
struct DetectorState<'a> {
    log_once: &'a mut LogOnce,
    last_status: &'a mut Option<ConsumerStatus>,
}

/// Resolve the `/dev/videoN` device path for a V4L2 output sink.
/// Returns `None` for non-V4L2 sinks (`Fake` / `Auto` / `Pipewire`) —
/// the supervisor uses the `Option` to gate detector spawn.
#[must_use]
pub(crate) fn device_path(sink: &OutputSink) -> Option<PathBuf> {
    let OutputSink::V4l2Loopback { device } = sink else {
        return None;
    };
    Some(device.clone())
}

/// One-shot fire gate. `fire()` returns `true` once and then `false`
/// until `rearm()` flips it back on. Encapsulates the consume-on-use
/// pattern shared by every diagnostic flag in [`LogOnce`].
struct Gate(bool);

impl Gate {
    /// Construct an armed gate (will fire on the next `fire()` call).
    const fn armed() -> Self {
        Self(true)
    }

    /// Attempt to consume the gate. Returns `true` exactly once per
    /// `armed()` / `rearm()` cycle.
    fn fire(&mut self) -> bool {
        std::mem::replace(&mut self.0, false)
    }

    /// Re-arm the gate so the next `fire()` returns `true` again.
    fn rearm(&mut self) {
        self.0 = true;
    }

    /// Test/inspection helper: is the gate currently armed?
    #[cfg(test)]
    pub(super) fn is_armed(&self) -> bool {
        self.0
    }
}

/// One-shot diagnostic gates for the detector. Each gate fires at
/// most once per re-arm window. `LogOnce::rearm()` re-arms every
/// gate in lockstep.
///
/// Gates:
/// * `scan_failed` — top-level `/proc` `read_dir` failure (the only
///   realistic detector failure).
/// * `eacces_storm` — every candidate `/proc/<pid>/fd` returned
///   `EACCES`, so we found no consumers but the answer is suspect.
struct LogOnce {
    scan_failed: Gate,
    eacces_storm: Gate,
}

impl LogOnce {
    /// All gates armed — each will log on its next matching failure.
    fn armed() -> Self {
        Self {
            scan_failed: Gate::armed(),
            eacces_storm: Gate::armed(),
        }
    }

    /// Re-arm every gate (called on `RESTAT_INTERVAL_POLLS` boundary).
    fn rearm(&mut self) {
        self.scan_failed.rearm();
        self.eacces_storm.rearm();
    }
}

/// Detector handle returned from [`ConsumerDetector::spawn`]. Owns
/// the worker thread and exposes the shared status atomic.
///
/// Drop joins the worker — see the module docstring. There is no
/// public `join` method; the supervisor relies on `Drop`.
pub(crate) struct ConsumerDetector {
    handle: Option<JoinHandle<()>>,
    status: Arc<AtomicU8>,
    /// Drop-only kill signal. Flipped to `true` in `Drop::drop` so
    /// the worker can exit even when the caller's `running` flag
    /// is still `true` (test cleanup; supervisor teardown ordering).
    stopped: Arc<AtomicBool>,
    /// Per-detector test-only counter, incremented each time
    /// `run_detector_inotify` successfully reaches its main loop
    /// (init + add_watch both succeeded). Tests snapshot the
    /// per-detector Arc to assert the inotify path was actually
    /// taken (vs. silently falling back to polling) without racing
    /// against sibling tests via a global.
    #[cfg(test)]
    inotify_activations: Arc<AtomicU64>,
}

impl ConsumerDetector {
    /// Spawn the detector thread. The thread watches the device via
    /// inotify; the `poll_interval` is used only by the
    /// `/proc/*/fd/` polling fallback that fires when inotify init
    /// or watch-add fails. The thread shuts down when either
    /// `running` flips to `false` or the returned handle is dropped.
    ///
    /// `device_path` is the `/dev/videoN` path whose basename we
    /// match against fd targets. `my_pid` is the supervisor's own
    /// pid (typically `std::process::id()`); fds held by this pid
    /// are excluded from the external-consumer count.
    ///
    /// The `Unknown` initial status is reset to the first observed
    /// value on the first poll — the supervisor's state machine
    /// treats `Unknown` as `Present`, so the first tick is a no-op
    /// (Active stays Active) regardless of what the walk finds.
    pub(crate) fn spawn(
        device_path: PathBuf,
        my_pid: u32,
        poll_interval: Duration,
        running: Arc<AtomicBool>,
    ) -> Self {
        Self::spawn_with_proc_root(
            device_path,
            PathBuf::from("/proc"),
            my_pid,
            poll_interval,
            running,
            RESTAT_INTERVAL_POLLS,
            #[cfg(test)]
            Arc::new(AtomicU64::new(0)),
        )
    }

    /// Test-only constructor that lets tests inject a tempdir as
    /// `/proc`, tune the re-arm cadence, and observe how often the
    /// inotify path activated. Production code uses [`Self::spawn`]
    /// which fixes the defaults.
    pub(crate) fn spawn_with_proc_root(
        device_path: PathBuf,
        proc_root: PathBuf,
        my_pid: u32,
        poll_interval: Duration,
        running: Arc<AtomicBool>,
        restat_interval_polls: u32,
        #[cfg(test)] inotify_activations: Arc<AtomicU64>,
    ) -> Self {
        let status = Arc::new(AtomicU8::new(ConsumerStatus::Unknown.as_u8()));
        let stopped = Arc::new(AtomicBool::new(false));
        let status_for_thread = Arc::clone(&status);
        let stopped_for_thread = Arc::clone(&stopped);
        #[cfg(test)]
        let activations_for_thread = Arc::clone(&inotify_activations);
        let handle = thread::Builder::new()
            .name("fluxframe-detector".into())
            .spawn(move || {
                run_detector(
                    device_path,
                    proc_root,
                    my_pid,
                    poll_interval,
                    running,
                    status_for_thread,
                    stopped_for_thread,
                    restat_interval_polls,
                    #[cfg(test)]
                    activations_for_thread,
                );
            })
            .expect("spawn detector thread");
        Self {
            handle: Some(handle),
            status,
            stopped,
            #[cfg(test)]
            inotify_activations,
        }
    }

    /// Cheap clone of the shared status atomic. The worker reads this
    /// every loop iteration; only the detector writes to it. Ordering
    /// is `Acquire` on the worker side, `Release` on the detector
    /// side — see the comment in [`run_detector`] for the rationale.
    pub(crate) fn status_handle(&self) -> Arc<AtomicU8> {
        Arc::clone(&self.status)
    }

    /// Test-only handle on the per-detector inotify activation
    /// counter. Tests assert `> 0` after spawn to verify the inotify
    /// path was actually taken.
    #[cfg(test)]
    pub(super) fn inotify_activations_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.inotify_activations)
    }
}

impl Drop for ConsumerDetector {
    fn drop(&mut self) {
        // Signal the worker to exit at its next poll-granularity
        // wakeup (≤ SHUTDOWN_POLL_GRANULARITY away).
        self.stopped.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            if let Err(panic) = handle.join() {
                warn!(
                    target: "fluxframe::idle",
                    panic = ?panic,
                    "consumer detector thread panicked during shutdown"
                );
            }
        }
    }
}

/// Publish `new_status` to the shared atomic and log the transition,
/// but only when it differs from the previously-published value.
///
/// `Release` store pairs with the worker's `Acquire` load in
/// `IdleStateMachine::tick` — guarantees the timer restart on the
/// worker side is ordered after the observed transition.
fn publish_if_changed(
    new_status: ConsumerStatus,
    status: &AtomicU8,
    last_status: &mut Option<ConsumerStatus>,
) {
    if Some(new_status) == *last_status {
        return;
    }
    status.store(new_status.as_u8(), Ordering::Release);
    if matches!(new_status, ConsumerStatus::Present | ConsumerStatus::Absent) {
        // Promoted to `info` (rare event, no spam risk) so the
        // operator can see attach/detach transitions with
        // `RUST_LOG=info` without enabling debug-level globally.
        info!(
            target: "fluxframe::idle",
            from = ?last_status,
            to = ?new_status,
            "consumer status changed"
        );
    } else {
        // Keep an explicit debug! for Unknown so operators running at
        // debug-level still see the transition.
        debug!(
            target: "fluxframe::idle",
            from = ?last_status,
            to = ?new_status,
            "consumer status changed"
        );
    }
    *last_status = Some(new_status);
}

#[allow(
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    reason = "thread entry-point: values are moved into the spawned thread on construction; \
              taking them by reference would require lifetime gymnastics across the thread \
              boundary, and bundling the owned values into a struct here only adds noise"
)]
fn run_detector(
    device_path: PathBuf,
    proc_root: PathBuf,
    my_pid: u32,
    poll_interval: Duration,
    running: Arc<AtomicBool>,
    status: Arc<AtomicU8>,
    stopped: Arc<AtomicBool>,
    restat_interval_polls: u32,
    #[cfg(test)] inotify_activations: Arc<AtomicU64>,
) {
    // Resolved once outside the loop — `file_name()` allocates `OsStr`
    // borrows from the underlying PathBuf and we want to avoid the
    // walk-time path manipulation.
    let Some(basename) = device_path.file_name() else {
        warn!(
            target: "fluxframe::idle",
            path = %device_path.display(),
            "device path has no basename — consumer detector disabled"
        );
        status.store(ConsumerStatus::Unknown.as_u8(), Ordering::Release);
        return;
    };
    let device_basename: OsString = basename.to_os_string();

    info!(
        target: "fluxframe::idle",
        device = %device_path.display(),
        my_pid,
        poll_ms = poll_interval.as_millis() as u64,
        "consumer detector started"
    );

    let mut log_once = LogOnce::armed();
    let mut last_status: Option<ConsumerStatus> = None;
    let ctx = DetectorContext {
        proc_root: &proc_root,
        my_pid,
        running: &running,
        status: &status,
        stopped: &stopped,
        restat_interval_polls,
        device_basename: &device_basename,
        #[cfg(test)]
        inotify_activations: &inotify_activations,
    };
    let mut state = DetectorState {
        log_once: &mut log_once,
        last_status: &mut last_status,
    };

    // Try the inotify event-driven path first. On failure (watch
    // limit, sandbox restriction, device-node deleted mid-run, …)
    // fall back to the polling path. Both share
    // `count_external_consumers` for the actual presence test, so
    // the public contract is identical — only the wake mechanism
    // differs.
    if let Some(exit) = run_detector_inotify(&device_path, &ctx, &mut state).err() {
        match &exit {
            DetectorExit::WatchRemoved => warn!(
                target: "fluxframe::idle",
                "inotify watch removed — falling back to /proc polling"
            ),
            DetectorExit::ReadFatal(e) => warn!(
                target: "fluxframe::idle",
                error = %e,
                "inotify read repeatedly failed — falling back to /proc polling"
            ),
            DetectorExit::InitFailed(_) | DetectorExit::AddFailed(_) => info!(
                target: "fluxframe::idle",
                reason = %exit,
                "inotify path unavailable — falling back to /proc polling"
            ),
        }
        run_detector_polling(poll_interval, &ctx, &mut state);
    }

    info!(target: "fluxframe::idle", "consumer detector exiting");
}

/// Running summary of a batch of `inotify` events. Built up
/// incrementally by [`EventSummary::observe`] as the reader drains
/// the kernel queue.
#[derive(Default, Clone, Copy)]
struct EventSummary {
    needs_rescan: bool,
    overflow: bool,
    watch_removed: bool,
}

impl EventSummary {
    /// Fold a single event's mask into the running summary.
    fn observe(&mut self, mask: inotify::EventMask) {
        if mask.contains(inotify::EventMask::Q_OVERFLOW) {
            self.overflow = true;
        }
        if mask.intersects(
            inotify::EventMask::OPEN
                | inotify::EventMask::CLOSE_WRITE
                | inotify::EventMask::CLOSE_NOWRITE,
        ) {
            self.needs_rescan = true;
        }
        if mask.contains(inotify::EventMask::IGNORED) {
            self.watch_removed = true;
        }
    }
}

/// Event-driven detector path. Watches the device for
/// `IN_OPEN` / `IN_CLOSE_NOWRITE` / `IN_CLOSE_WRITE` and, on every
/// event, walks `/proc` via `count_external_consumers` to decide
/// Present vs Absent. In steady-state idle the cost is one
/// nonblocking `read(2)` returning `EAGAIN` every ~50 ms (the
/// [`SHUTDOWN_POLL_GRANULARITY`] window), not a per-event syscall —
/// orders of magnitude cheaper than the prior per-poll `/proc`
/// walk, and the kernel only delivers an actual event when the
/// device is opened or closed.
///
/// # Errors
///
/// Returns:
///
/// * [`DetectorExit::InitFailed`] — `Inotify::init` failed (sandbox,
///   ENOMEM).
/// * [`DetectorExit::AddFailed`] — `watches().add` failed (ENOSPC at
///   `max_user_watches`, ENOENT for a missing device node).
/// * [`DetectorExit::WatchRemoved`] — the watch fired `IN_IGNORED`
///   mid-run (e.g. `modprobe -r v4l2loopback`).
/// * [`DetectorExit::ReadFatal`] — `read_events` returned
///   non-recoverable errors for [`INOTIFY_READ_ERROR_BUDGET`]
///   consecutive iterations; the first failure already published
///   `Unknown` (fail-open).
fn run_detector_inotify(
    device_path: &Path,
    ctx: &DetectorContext<'_>,
    state: &mut DetectorState<'_>,
) -> Result<(), DetectorExit> {
    let mut inotify = inotify::Inotify::init().map_err(DetectorExit::InitFailed)?;
    inotify
        .watches()
        .add(
            device_path,
            inotify::WatchMask::OPEN
                | inotify::WatchMask::CLOSE_NOWRITE
                | inotify::WatchMask::CLOSE_WRITE,
        )
        .map_err(DetectorExit::AddFailed)?;

    #[cfg(test)]
    ctx.inotify_activations.fetch_add(1, Ordering::Release);

    // Baseline walk — the kernel only delivers events that happen
    // AFTER `add()`, so we need one synchronous walk to publish the
    // initial state. Without this the worker would observe
    // `Unknown` until the first open/close, which on a quiet system
    // may be never.
    let baseline = count_external_consumers(
        ctx.proc_root,
        ctx.device_basename,
        ctx.my_pid,
        state.log_once,
    );
    publish_if_changed(baseline, ctx.status, state.last_status);

    let mut buffer = [0u8; INOTIFY_EVENT_BUFFER_BYTES];
    let mut walks_since_restat: u32 = 0;
    // Consecutive `read_events` errors (non-WouldBlock). Reset on
    // every successful read. The first error already published
    // Unknown (fail-open); the budget below caps how long we keep
    // retrying before escalating to a polling fallback.
    let mut consecutive_read_errors: u32 = 0;

    while ctx.running.load(Ordering::Acquire) && !ctx.stopped.load(Ordering::Acquire) {
        let summary = read_event_batch(
            &mut inotify,
            &mut buffer,
            ctx,
            state,
            &mut consecutive_read_errors,
        )?;
        apply_summary(summary, ctx, state, &mut walks_since_restat)?;
    }

    Ok(())
}

/// Drain queued `inotify` events into an [`EventSummary`]. Handles
/// the WouldBlock sleep, the error budget, and the first-error
/// `Unknown` publish. Returns the accumulated summary on success;
/// returns [`DetectorExit::ReadFatal`] when the consecutive-error
/// budget is exhausted.
fn read_event_batch(
    inotify: &mut inotify::Inotify,
    buffer: &mut [u8],
    ctx: &DetectorContext<'_>,
    state: &mut DetectorState<'_>,
    consecutive_read_errors: &mut u32,
) -> Result<EventSummary, DetectorExit> {
    let mut summary = EventSummary::default();
    match inotify.read_events(buffer) {
        Ok(events) => {
            *consecutive_read_errors = 0;
            for ev in events {
                summary.observe(ev.mask);
            }
        }
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
            *consecutive_read_errors = 0;
            // No events queued. Sleep at the shutdown granularity
            // so we honour `running` / `stopped` within ~50 ms.
            thread::sleep(SHUTDOWN_POLL_GRANULARITY);
        }
        Err(e) => {
            if *consecutive_read_errors == 0 {
                // Fail-open on the first surprise so the
                // supervisor's idle FSM treats the gap as Present
                // (Unknown is mapped to Present upstream). Better
                // to leave the input pipeline running through a
                // transient inotify hiccup than silently freeze on
                // stale Present.
                publish_if_changed(ConsumerStatus::Unknown, ctx.status, state.last_status);
            }
            *consecutive_read_errors = consecutive_read_errors.saturating_add(1);
            // Noise reduction: warn only on the very first error
            // (operator-visible) and right before we escalate to a
            // polling fallback. Intermediate retries land at debug
            // so a transient EINTR storm doesn't flood the log.
            let consecutive = *consecutive_read_errors;
            if consecutive == 1 || consecutive >= INOTIFY_READ_ERROR_BUDGET - 1 {
                warn!(
                    target: "fluxframe::idle",
                    error = %e,
                    consecutive,
                    "inotify read failed; retrying after shutdown poll window"
                );
            } else {
                debug!(
                    target: "fluxframe::idle",
                    error = %e,
                    consecutive,
                    "inotify read failed; retrying after shutdown poll window"
                );
            }
            if consecutive >= INOTIFY_READ_ERROR_BUDGET {
                return Err(DetectorExit::ReadFatal(e));
            }
            thread::sleep(SHUTDOWN_POLL_GRANULARITY);
        }
    }
    Ok(summary)
}

/// React to a drained [`EventSummary`]: log overflow, escalate
/// `IGNORED` to a `WatchRemoved` exit (publishing `Unknown` first so
/// the polling fallback's startup window doesn't leave a stale
/// status), and run the `/proc` walk + restat tick when an
/// open/close or overflow needs a rescan.
fn apply_summary(
    summary: EventSummary,
    ctx: &DetectorContext<'_>,
    state: &mut DetectorState<'_>,
    walks_since_restat: &mut u32,
) -> Result<(), DetectorExit> {
    if summary.watch_removed {
        // Publish Unknown BEFORE escalating so the polling fallback's
        // startup window (until its first walk completes) doesn't
        // leave a stale Present/Absent behind for the supervisor's
        // state machine to act on.
        publish_if_changed(ConsumerStatus::Unknown, ctx.status, state.last_status);
        // Device node disappeared. Surface to the dispatcher so
        // the polling fallback can keep producing a status
        // until the device returns.
        return Err(DetectorExit::WatchRemoved);
    }

    if summary.overflow {
        info!(
            target: "fluxframe::idle",
            "inotify queue overflow — resyncing via /proc walk"
        );
    }

    if summary.needs_rescan || summary.overflow {
        let new_status = count_external_consumers(
            ctx.proc_root,
            ctx.device_basename,
            ctx.my_pid,
            state.log_once,
        );
        publish_if_changed(new_status, ctx.status, state.last_status);
        // Rearm gate cadence shifts from wall-clock (polling path)
        // to walk-count here. The gates exist to surface
        // activity-time problems; rearm-on-activity is arguably
        // better behaviour.
        tick_restat(
            walks_since_restat,
            ctx.restat_interval_polls,
            state.log_once,
        );
    }
    Ok(())
}

/// Bump a restat counter; when it crosses `interval`, rearm the
/// log-once gates and reset the counter. Shared by both the inotify
/// path (counts walks) and the polling path (counts polls).
fn tick_restat(counter: &mut u32, interval: u32, log_once: &mut LogOnce) {
    *counter = counter.wrapping_add(1);
    if *counter >= interval {
        log_once.rearm();
        *counter = 0;
    }
}

fn run_detector_polling(
    poll_interval: Duration,
    ctx: &DetectorContext<'_>,
    state: &mut DetectorState<'_>,
) {
    let mut polls_since_restat: u32 = 0;

    while ctx.running.load(Ordering::Acquire) && !ctx.stopped.load(Ordering::Acquire) {
        let new_status = count_external_consumers(
            ctx.proc_root,
            ctx.device_basename,
            ctx.my_pid,
            state.log_once,
        );
        publish_if_changed(new_status, ctx.status, state.last_status);

        // Clear the one-shot log flag every restat_interval_polls
        // so a transient diagnostic (e.g. `/proc` briefly remount)
        // re-surfaces after the re-arm window.
        tick_restat(
            &mut polls_since_restat,
            ctx.restat_interval_polls,
            state.log_once,
        );

        // Bounded shutdown latency: chunk the sleep so we re-check
        // `running` / `stopped` at least every
        // `SHUTDOWN_POLL_GRANULARITY`. Without this cap, a multi-
        // second `poll_interval` would delay Ctrl-C / Drop for the
        // full interval.
        let mut remaining = poll_interval;
        while remaining > Duration::ZERO
            && ctx.running.load(Ordering::Acquire)
            && !ctx.stopped.load(Ordering::Acquire)
        {
            let chunk = remaining.min(SHUTDOWN_POLL_GRANULARITY);
            thread::sleep(chunk);
            remaining = remaining.saturating_sub(chunk);
        }
    }
}

/// Walk `proc_root` (typically `/proc`) and decide whether any
/// external process holds an fd whose symlink target's basename
/// equals `device_basename`. The walk short-circuits as soon as the
/// first match is found — there is no count, only Present / Absent.
///
/// # Errors handled inline
///
/// * Top-level `read_dir(proc_root)` failure fires
///   `log_once.scan_failed` (warn) and returns `Unknown`.
/// * Per-pid `read_dir(<proc>/<pid>/fd)` `EACCES` errors are counted
///   and surfaced once via `log_once.eacces_storm` (info) if the
///   scan would otherwise return `Absent` — i.e. every candidate pid
///   was unreadable. This catches the multi-user blind spot where an
///   unprivileged fluxframe cannot see a root-owned consumer.
/// * Per-pid `ENOENT` and other I/O errors skip silently — steady-
///   state expected on a multi-user / sandboxed system.
/// * `read_link` failures on individual fds skip silently.
fn count_external_consumers(
    proc_root: &Path,
    device_basename: &OsStr,
    my_pid: u32,
    log_once: &mut LogOnce,
) -> ConsumerStatus {
    let entries = match std::fs::read_dir(proc_root) {
        Ok(it) => it,
        Err(e) => {
            if log_once.scan_failed.fire() {
                warn!(
                    target: "fluxframe::idle",
                    proc_root = %proc_root.display(),
                    error = %e,
                    "/proc scan failed — consumer detection disabled"
                );
            }
            return ConsumerStatus::Unknown;
        }
    };

    let mut has_external = false;
    let mut eacces_count: u32 = 0;

    'outer: for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let Ok(pid) = name_str.parse::<u32>() else {
            // Non-numeric `/proc` entries (`self`, `cpuinfo`, …).
            continue;
        };
        if pid == my_pid {
            continue;
        }

        let fd_dir = entry.path().join("fd");
        // EACCES (root or other-user pid) is counted so we can surface
        // the "every candidate was unreadable" blind spot below.
        // ENOENT (process exited mid-scan) and any other I/O error
        // skip silently — steady-state expected on a multi-user
        // system.
        let fd_entries = match std::fs::read_dir(&fd_dir) {
            Ok(it) => it,
            Err(e) if e.kind() == io::ErrorKind::PermissionDenied => {
                eacces_count = eacces_count.saturating_add(1);
                continue;
            }
            Err(_) => continue,
        };

        for fd in fd_entries.flatten() {
            let Ok(link) = std::fs::read_link(fd.path()) else {
                continue;
            };
            if link.file_name() == Some(device_basename) {
                has_external = true;
                // One matching fd is enough to settle Present/Absent.
                // Break the inner fd loop AND the outer pid loop —
                // scanning further pids would be wasted work.
                break 'outer;
            }
        }
    }

    if !has_external && eacces_count > 0 && log_once.eacces_storm.fire() {
        info!(
            target: "fluxframe::idle",
            eacces_count,
            "scanned /proc; all candidate fd dirs returned EACCES — idle may be inaccurate if a privileged consumer is active"
        );
    }

    if has_external {
        ConsumerStatus::Present
    } else {
        ConsumerStatus::Absent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::time::Instant;

    /// Helper: wait for the detector to publish the expected status
    /// or time out. Spin-poll cadence is short enough that the test
    /// stays under a few hundred ms even on slow CI.
    fn wait_for_status(
        status: &Arc<AtomicU8>,
        expected: ConsumerStatus,
        timeout: Duration,
    ) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            let observed = ConsumerStatus::from_u8(status.load(Ordering::Acquire));
            if observed == expected {
                return true;
            }
            thread::sleep(Duration::from_millis(5));
        }
        false
    }

    /// Synthesise a `/proc/<pid>/fd/<fd>` symlink pointing at
    /// `target`. `target` does not need to exist.
    fn make_proc_entry(proc_root: &Path, pid: u32, fd: u32, target: &Path) {
        let fd_dir = proc_root.join(pid.to_string()).join("fd");
        fs::create_dir_all(&fd_dir).unwrap();
        symlink(target, fd_dir.join(fd.to_string())).unwrap();
    }

    #[test]
    fn device_path_for_v4l2_loopback() {
        let sink = OutputSink::V4l2Loopback {
            device: PathBuf::from("/dev/video10"),
        };
        let path = device_path(&sink).expect("v4l2 sink yields a device path");
        assert_eq!(path, PathBuf::from("/dev/video10"));
    }

    #[test]
    fn device_path_none_for_non_v4l2_sinks() {
        assert!(device_path(&OutputSink::Fake).is_none());
        assert!(device_path(&OutputSink::Auto).is_none());
        assert!(
            device_path(&OutputSink::Pipewire {
                node_name: Some("fluxframe".into())
            })
            .is_none()
        );
    }

    #[test]
    fn one_external_pid_with_matching_fd_is_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        make_proc_entry(dir.path(), 1234, 3, Path::new("/dev/video10"));

        let mut log = LogOnce::armed();
        let status = count_external_consumers(dir.path(), OsStr::new("video10"), 9999, &mut log);
        assert_eq!(status, ConsumerStatus::Present);
        assert!(
            log.scan_failed.is_armed(),
            "happy path must not consume scan_failed gate"
        );
    }

    #[test]
    fn multiple_fds_same_pid_count_once() {
        // Two fds from one pid both pointing at /dev/video10 should
        // still produce Present (and conceptually one consumer — the
        // public surface only exposes Present/Absent, so we verify
        // status; HashSet collapses the duplicates internally).
        let dir = tempfile::tempdir().expect("tempdir");
        make_proc_entry(dir.path(), 1234, 3, Path::new("/dev/video10"));
        make_proc_entry(dir.path(), 1234, 4, Path::new("/dev/video10"));
        make_proc_entry(dir.path(), 1234, 5, Path::new("/dev/video10"));

        let mut log = LogOnce::armed();
        let status = count_external_consumers(dir.path(), OsStr::new("video10"), 9999, &mut log);
        assert_eq!(status, ConsumerStatus::Present);
    }

    #[test]
    fn self_pid_only_is_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        // fluxframe's own fd should not count as a consumer.
        make_proc_entry(dir.path(), 9999, 3, Path::new("/dev/video10"));

        let mut log = LogOnce::armed();
        let status = count_external_consumers(dir.path(), OsStr::new("video10"), 9999, &mut log);
        assert_eq!(status, ConsumerStatus::Absent);
    }

    #[test]
    fn no_matching_fds_is_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        // External pid but holding unrelated fds.
        make_proc_entry(dir.path(), 1234, 3, Path::new("/dev/null"));
        make_proc_entry(dir.path(), 1234, 4, Path::new("/dev/video11"));
        make_proc_entry(dir.path(), 5678, 3, Path::new("/dev/zero"));

        let mut log = LogOnce::armed();
        let status = count_external_consumers(dir.path(), OsStr::new("video10"), 9999, &mut log);
        assert_eq!(status, ConsumerStatus::Absent);
    }

    /// A pid with a chmod-0o000 fd dir must be skipped silently
    /// (steady-state expected on multi-user systems), while a
    /// readable pid with a matching fd is still surfaced. The
    /// `scan_failed` gate must NOT fire — that's reserved for
    /// `/proc` root-level failure only. The `eacces_storm` gate
    /// also must NOT fire here because we DID find an external
    /// consumer — the EACCES count is irrelevant in that case.
    #[test]
    fn permission_denied_pid_skipped_silently() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Unreadable pid (simulated root-owned process).
        let unreadable_fd_dir = dir.path().join("1234").join("fd");
        fs::create_dir_all(&unreadable_fd_dir).unwrap();
        // Restore permissions before the tempdir drops — some
        // platforms refuse to remove a 0o000 directory.
        let _restore = ScopeGuard::new({
            let p = unreadable_fd_dir.clone();
            move || {
                let _ = fs::set_permissions(&p, fs::Permissions::from_mode(0o700));
            }
        });
        fs::set_permissions(&unreadable_fd_dir, fs::Permissions::from_mode(0o000)).unwrap();

        // External pid that IS readable and matches.
        make_proc_entry(dir.path(), 5678, 3, Path::new("/dev/video10"));

        let mut log = LogOnce::armed();
        let status = count_external_consumers(dir.path(), OsStr::new("video10"), 9999, &mut log);
        assert_eq!(status, ConsumerStatus::Present);
        assert!(
            log.scan_failed.is_armed(),
            "per-pid EACCES must not consume the scan_failed gate"
        );
        assert!(
            log.eacces_storm.is_armed(),
            "found a consumer => eacces_storm gate must stay armed"
        );
    }

    #[test]
    fn unreadable_proc_root_is_unknown() {
        let mut log = LogOnce::armed();
        let status = count_external_consumers(
            Path::new("/no/such/proc/path"),
            OsStr::new("video10"),
            9999,
            &mut log,
        );
        assert_eq!(status, ConsumerStatus::Unknown);
        assert!(
            !log.scan_failed.is_armed(),
            "scan_failed gate must be consumed when /proc root cannot be read"
        );
    }

    #[test]
    fn mixed_self_plus_external_is_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        make_proc_entry(dir.path(), 9999, 3, Path::new("/dev/video10"));
        make_proc_entry(dir.path(), 1234, 3, Path::new("/dev/video10"));

        let mut log = LogOnce::armed();
        let status = count_external_consumers(dir.path(), OsStr::new("video10"), 9999, &mut log);
        assert_eq!(status, ConsumerStatus::Present);
    }

    /// Walk the full transition cycle expected at runtime: absent →
    /// add an external consumer → drop it → make the proc root
    /// disappear → put it back. Verifies the **polling fallback**
    /// path picks up each transition within the poll window — the
    /// device path points at a non-existent node so the inotify
    /// path fails at `add_watch` and the dispatcher falls back to
    /// polling. The inotify path itself is exercised by
    /// `inotify_path_fires_on_external_open` below.
    #[test]
    fn detector_cycles_through_states_via_fake_proc_polling_fallback() {
        let dir = tempfile::tempdir().expect("tempdir");
        let proc_root = dir.path().to_path_buf();
        // Synthesise an inert non-numeric entry so the directory is
        // non-empty from the start (read_dir on an empty tempdir is
        // still legal but the test cycles below mutate it).
        fs::write(proc_root.join("self"), "").unwrap();

        let running = Arc::new(AtomicBool::new(true));
        // Non-existent device path forces inotify::watches().add to
        // fail with ENOENT → dispatcher falls back to polling. The
        // per-detector activations counter is checked at end-of-test
        // to confirm the inotify path never activated.
        let activations = Arc::new(AtomicU64::new(0));
        let detector = ConsumerDetector::spawn_with_proc_root(
            PathBuf::from("/dev/this-device-does-not-exist"),
            proc_root.clone(),
            9999,
            Duration::from_millis(20),
            Arc::clone(&running),
            RESTAT_INTERVAL_POLLS,
            Arc::clone(&activations),
        );
        let status = detector.status_handle();

        assert!(
            wait_for_status(&status, ConsumerStatus::Absent, Duration::from_millis(500)),
            "empty proc must surface as Absent"
        );

        // Attach an external consumer. Basename must match the
        // device_path basename used at detector spawn.
        make_proc_entry(
            &proc_root,
            1234,
            3,
            Path::new("/dev/this-device-does-not-exist"),
        );
        assert!(
            wait_for_status(&status, ConsumerStatus::Present, Duration::from_millis(500)),
            "external pid attaching must surface as Present"
        );

        // Detach.
        fs::remove_dir_all(proc_root.join("1234")).unwrap();
        assert!(
            wait_for_status(&status, ConsumerStatus::Absent, Duration::from_millis(500)),
            "external pid detaching must surface as Absent"
        );

        assert_eq!(
            activations.load(Ordering::Acquire),
            0,
            "inotify path must not activate when add_watch fails"
        );

        running.store(false, Ordering::Release);
        drop(detector);
    }

    /// Shutdown latency is bounded by `SHUTDOWN_POLL_GRANULARITY`
    /// (~50 ms), NOT by `poll_interval`. The 500 ms tolerance leaves
    /// CI plenty of slack to avoid flakiness.
    #[test]
    fn shutdown_completes_quickly_even_with_long_poll_interval() {
        let dir = tempfile::tempdir().expect("tempdir");
        let running = Arc::new(AtomicBool::new(true));
        let detector = ConsumerDetector::spawn_with_proc_root(
            PathBuf::from("/dev/video10"),
            dir.path().to_path_buf(),
            9999,
            Duration::from_secs(2), // long poll
            Arc::clone(&running),
            RESTAT_INTERVAL_POLLS,
            Arc::new(AtomicU64::new(0)),
        );

        // Wait briefly to ensure the detector entered its sleep loop.
        thread::sleep(Duration::from_millis(100));

        let start = Instant::now();
        running.store(false, Ordering::Release);
        drop(detector); // Drop joins under the new pattern.
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_millis(500),
            "shutdown took {elapsed:?} — expected < 500 ms"
        );
    }

    /// `LogOnce::rearm` resets the `scan_failed` gate so a transient
    /// `/proc` failure can re-surface in the log after the re-arm
    /// window. `eacces_storm` follows the same contract (rearmed in
    /// lockstep).
    #[test]
    fn log_once_rearm_re_arms_scan_failed_armed() {
        let mut log = LogOnce::armed();
        assert!(log.scan_failed.is_armed());
        assert!(log.eacces_storm.is_armed());

        // Consuming the gate via an unreadable proc root.
        let _ = count_external_consumers(
            Path::new("/no/such/proc/path"),
            OsStr::new("video10"),
            9999,
            &mut log,
        );
        assert!(!log.scan_failed.is_armed());

        // A second failure does NOT re-flip — gate stays consumed.
        let _ = count_external_consumers(
            Path::new("/no/such/proc/path"),
            OsStr::new("video10"),
            9999,
            &mut log,
        );
        assert!(!log.scan_failed.is_armed());

        log.rearm();
        assert!(
            log.scan_failed.is_armed(),
            "rearm restores the scan_failed gate"
        );
        assert!(
            log.eacces_storm.is_armed(),
            "rearm restores the eacces_storm gate"
        );

        // Now the gate fires again on the next failure.
        let _ = count_external_consumers(
            Path::new("/no/such/proc/path"),
            OsStr::new("video10"),
            9999,
            &mut log,
        );
        assert!(
            !log.scan_failed.is_armed(),
            "re-armed gate is consumable again"
        );
    }

    /// Synthesise a `/proc` walk where the only candidate pid has a
    /// chmod-0o000 fd dir and no other pids exist — exactly the
    /// scenario where every external candidate returns EACCES, the
    /// scan finds nothing, and idle would otherwise wrongly engage.
    ///
    /// Contract: the gate fires exactly once (consumed on the first
    /// matching call), stays consumed on the second call, and is
    /// restored by `rearm()`.
    #[test]
    fn eacces_storm_logs_once_per_rearm_window() {
        let dir = tempfile::tempdir().expect("tempdir");
        // One unreadable pid dir, nothing else.
        let unreadable_fd_dir = dir.path().join("1234").join("fd");
        fs::create_dir_all(&unreadable_fd_dir).unwrap();
        let _restore = ScopeGuard::new({
            let p = unreadable_fd_dir.clone();
            move || {
                let _ = fs::set_permissions(&p, fs::Permissions::from_mode(0o700));
            }
        });
        fs::set_permissions(&unreadable_fd_dir, fs::Permissions::from_mode(0o000)).unwrap();

        let mut log = LogOnce::armed();
        assert!(log.eacces_storm.is_armed());

        // First call: finds nothing, sees one EACCES, consumes the gate.
        let status = count_external_consumers(dir.path(), OsStr::new("video10"), 9999, &mut log);
        assert_eq!(status, ConsumerStatus::Absent);
        assert!(
            !log.eacces_storm.is_armed(),
            "EACCES storm must consume the gate on first occurrence"
        );

        // Second call: still finds nothing, still sees EACCES, but
        // gate stays consumed (no re-log).
        let status = count_external_consumers(dir.path(), OsStr::new("video10"), 9999, &mut log);
        assert_eq!(status, ConsumerStatus::Absent);
        assert!(
            !log.eacces_storm.is_armed(),
            "consumed gate must stay consumed across calls"
        );

        // rearm restores it; next call fires it again.
        log.rearm();
        assert!(log.eacces_storm.is_armed());
        let _ = count_external_consumers(dir.path(), OsStr::new("video10"), 9999, &mut log);
        assert!(
            !log.eacces_storm.is_armed(),
            "re-armed gate is consumable again"
        );
    }

    /// Pragmatic survival test for the `run_detector` rearm boundary:
    /// drive the worker at the most aggressive cadence
    /// (`restat_interval_polls = 1`, so EVERY poll rearms) against a
    /// non-existent proc_root that keeps producing the `scan_failed`
    /// diagnostic, and verify the thread doesn't deadlock, panic, or
    /// burn CPU into oblivion.
    ///
    /// Note: this test validates that the rearm code path is
    /// exercised at the per-poll boundary AT MINIMUM. A finer
    /// assertion (e.g. counting rearm invocations) would require
    /// wiring a test-only Arc<AtomicU32> into `LogOnce`, which is
    /// disproportionate plumbing for one assertion.
    #[test]
    fn run_detector_rearms_log_flag_at_restat_boundary() {
        let running = Arc::new(AtomicBool::new(true));
        let detector = ConsumerDetector::spawn_with_proc_root(
            PathBuf::from("/dev/video10"),
            PathBuf::from("/no/such/proc/path"),
            9999,
            Duration::from_millis(20),
            Arc::clone(&running),
            1, // rearm on EVERY poll
            Arc::new(AtomicU64::new(0)),
        );
        let status = detector.status_handle();

        // Worker keeps producing Unknown because the proc_root is
        // unreadable; we just need the thread to survive the
        // aggressive cadence without crashing.
        assert!(
            wait_for_status(&status, ConsumerStatus::Unknown, Duration::from_millis(300)),
            "missing proc root must surface as Unknown"
        );

        // Let the loop iterate several times so the rearm path runs
        // many times in a row. If rearm panicked or deadlocked, the
        // shutdown below would hang.
        thread::sleep(Duration::from_millis(200));
        assert_eq!(
            ConsumerStatus::from_u8(status.load(Ordering::Acquire)),
            ConsumerStatus::Unknown,
            "worker must keep publishing Unknown across rearm cycles"
        );

        running.store(false, Ordering::Release);
        drop(detector);
    }

    /// Exercise the inotify event-driven path against a real file:
    /// open the watched node from the test thread; the detector's
    /// inotify watch fires `IN_OPEN`, the loop walks the fake `/proc`,
    /// finds the synthesised external entry, and publishes `Present`.
    ///
    /// Uses `NamedTempFile` as the device path because (a) it exists
    /// on disk so `inotify::watches().add` succeeds; (b) the test
    /// thread controls open/close cadence; (c) the basename is
    /// unique per run, avoiding cross-test interference.
    ///
    /// The fake `/proc/<pid>/fd/<fd>` symlink targets the same
    /// `NamedTempFile::path()`, so `count_external_consumers`
    /// compares matching basenames and the synthesised "consumer"
    /// counts.
    #[test]
    fn inotify_path_fires_on_external_open() {
        let proc_dir = tempfile::tempdir().expect("proc tempdir");
        let proc_root = proc_dir.path().to_path_buf();
        // Inert non-numeric entry so the dir scan starts non-empty.
        fs::write(proc_root.join("self"), "").unwrap();

        // The device path is a real file we control. NamedTempFile
        // creates it on disk; inotify::watches().add accepts any
        // regular file as a watch target.
        let device = tempfile::NamedTempFile::new().expect("device tempfile");
        let device_path = device.path().to_path_buf();

        // Pre-arm the synthesised consumer so the post-event walk
        // finds it. The detector won't walk before the first open
        // event arrives (baseline is the only walk before that, and
        // baseline = no /proc/1234 entry yet).
        make_proc_entry(&proc_root, 1234, 3, &device_path);

        let running = Arc::new(AtomicBool::new(true));
        let detector = ConsumerDetector::spawn_with_proc_root(
            device_path.clone(),
            proc_root.clone(),
            9999,
            Duration::from_millis(20),
            Arc::clone(&running),
            RESTAT_INTERVAL_POLLS,
            Arc::new(AtomicU64::new(0)),
        );
        let status = detector.status_handle();
        // Per-detector activation counter — no global, no cross-test race.
        let activations = detector.inotify_activations_handle();

        // Baseline walk runs immediately after `add_watch` succeeds.
        // Since the /proc entry already points at our device_path,
        // baseline must surface as Present.
        assert!(
            wait_for_status(&status, ConsumerStatus::Present, Duration::from_millis(500)),
            "baseline walk must surface synthesised consumer as Present"
        );
        assert!(
            activations.load(Ordering::Acquire) > 0,
            "inotify path must have activated (init + add_watch succeeded)"
        );

        // Detach the synthesised consumer — but the detector is
        // event-driven now, so it only re-walks on the NEXT open
        // event. Touch the file to trigger one.
        fs::remove_dir_all(proc_root.join("1234")).unwrap();
        {
            // Open + immediate close → fires IN_OPEN + IN_CLOSE_*.
            let _f = std::fs::File::open(&device_path).expect("open device");
        }
        assert!(
            wait_for_status(&status, ConsumerStatus::Absent, Duration::from_millis(500)),
            "event-driven walk must observe the consumer disappearing"
        );

        // Re-attach and re-trigger.
        make_proc_entry(&proc_root, 5678, 3, &device_path);
        {
            let _f = std::fs::File::open(&device_path).expect("open device again");
        }
        assert!(
            wait_for_status(&status, ConsumerStatus::Present, Duration::from_millis(500)),
            "event-driven walk must observe the consumer re-attaching"
        );

        running.store(false, Ordering::Release);
        drop(detector);
    }

    /// `EventMask::IGNORED` is the kernel's signal that the watched
    /// file went away (deleted, fs unmounted). The detector surfaces
    /// it as an `io::Error` from `run_detector_inotify`, and the
    /// dispatcher falls back to polling. This test verifies the
    /// fallback by deleting the watched file mid-run and confirming
    /// the detector keeps producing a status (polling fallback
    /// against the deleted-then-recreated /proc/<pid>/fd entry).
    ///
    /// Realistic operational scenario: `modprobe -r v4l2loopback`
    /// while fluxframe is running.
    #[test]
    fn inotify_watch_removed_falls_back_to_polling() {
        let proc_dir = tempfile::tempdir().expect("proc tempdir");
        let proc_root = proc_dir.path().to_path_buf();
        fs::write(proc_root.join("self"), "").unwrap();

        let device = tempfile::NamedTempFile::new().expect("device tempfile");
        let device_path = device.path().to_path_buf();

        make_proc_entry(&proc_root, 1234, 3, &device_path);

        let running = Arc::new(AtomicBool::new(true));
        let detector = ConsumerDetector::spawn_with_proc_root(
            device_path.clone(),
            proc_root.clone(),
            9999,
            Duration::from_millis(20),
            Arc::clone(&running),
            RESTAT_INTERVAL_POLLS,
            Arc::new(AtomicU64::new(0)),
        );
        let status = detector.status_handle();

        // Baseline must reach Present via inotify path.
        assert!(
            wait_for_status(&status, ConsumerStatus::Present, Duration::from_millis(500)),
            "baseline walk must surface synthesised consumer as Present"
        );

        // Delete the device file. The kernel fires IN_IGNORED on
        // the watch; run_detector_inotify returns Err; dispatcher
        // falls back to polling. After fallback, status should
        // still track /proc mutations.
        drop(device); // tempfile is unlink-on-drop
        // Give the inotify path a window to observe IN_IGNORED and
        // the dispatcher to enter the polling fallback.
        thread::sleep(Duration::from_millis(100));

        // Detach the consumer in /proc. Polling should observe.
        fs::remove_dir_all(proc_root.join("1234")).unwrap();
        assert!(
            wait_for_status(&status, ConsumerStatus::Absent, Duration::from_millis(500)),
            "polling fallback must observe consumer detach after IN_IGNORED"
        );

        running.store(false, Ordering::Release);
        drop(detector);
    }

    /// Tiny RAII guard so the `permission_denied_pid_skipped_silently`
    /// test can restore the 0o000 dir's permissions before tempdir
    /// drops. Pulled in here rather than as a dep — `defer-rs` /
    /// `scopeguard` would be overkill for one call site.
    struct ScopeGuard<F: FnMut()> {
        f: Option<F>,
    }

    impl<F: FnMut()> ScopeGuard<F> {
        fn new(f: F) -> Self {
            Self { f: Some(f) }
        }
    }

    impl<F: FnMut()> Drop for ScopeGuard<F> {
        fn drop(&mut self) {
            if let Some(mut f) = self.f.take() {
                f();
            }
        }
    }
}
