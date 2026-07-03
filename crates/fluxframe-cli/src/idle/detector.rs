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
//! On every event the detector classifies the device via two signals,
//! folded together in [`ConsumerPresence`]:
//!
//! 1. A `/proc/[0-9]+/fd/` walk ([`count_external_consumers`], filtering
//!    out `my_pid`) yields a [`WalkOutcome`]: a readable external holder
//!    (`External`), a fully-readable "nobody" (`CleanAbsent`), a walk
//!    where ≥1 candidate `/proc/<pid>/fd` returned `EACCES` so the answer
//!    is suspect (`UncertainAbsent` — a sandboxed/root-owned consumer is
//!    indistinguishable from "nobody"), or an unreadable `/proc` root
//!    (`Unreadable` → `Unknown`, fail-open).
//! 2. A net **open balance**: `IN_OPEN` is `+1`, `IN_CLOSE_*` is `−1`,
//!    accumulated since the watch armed. fluxframe never re-opens the
//!    loopback after startup (its output write-fd is opened before the
//!    watch and held for the whole run), so a positive balance means an
//!    external consumer is attached — *even one the walk cannot see*.
//!
//! `ConsumerPresence::observe` combines them: a readable walk is
//! authoritative (`External → Present`, `CleanAbsent → Absent`), while an
//! `UncertainAbsent` walk defers to the balance — a held open we could
//! not attribute (EACCES) still wakes to `Present`, and returns to
//! `Absent` when it closes. This is what lets a sandboxed browser (whose
//! capture process is non-dumpable → `/proc/<pid>/fd` = EACCES) wake the
//! daemon. `ENOENT` from a process that exited mid-scan is skipped.
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
//! * **Unattributable consumers rely on the open balance.** An
//!   unprivileged fluxframe cannot read a root-owned or sandboxed
//!   consumer's `/proc/<pid>/fd`, so the walk classifies it as
//!   `UncertainAbsent`. The net open balance still catches a *held*
//!   open (inotify fires regardless of who opened), so such a consumer
//!   does wake the daemon; the residual blind spot is a consumer that
//!   was already holding the device *before* the watch armed (its
//!   opening `IN_OPEN` predates the balance) — it reads as `Absent`
//!   until it re-opens. Running fluxframe as the same user as the
//!   consumer removes the EACCES entirely (the walk then sees it).
//! * **`/proc` must be mounted.** Hard containerization that hides
//!   `/proc` collapses the walk to `Unreadable` → `Unknown` (fail-open).
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

use fluxframe_core::Counters;
use fluxframe_gst::output::OutputSink;
use tracing::{debug, info, warn};

use super::state::ConsumerStatus;

/// Maximum number of unreadable-pid identifiers sampled into
/// [`WalkOutcome::UncertainAbsent`] for the diagnostic line. The full
/// EACCES count is kept separately; only this bounded sample is carried
/// so a pathological host cannot build an unbounded `Vec`.
const EACCES_PID_SAMPLE: usize = 8;

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
    /// Process-wide counters. The detector bumps
    /// `unattributable_wakes_total` when it wakes on an open it could
    /// not attribute via `/proc`.
    counters: &'a Counters,
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
    presence: &'a mut ConsumerPresence,
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

/// Outcome of one `/proc` walk — pure data, no side effects. The caller
/// ([`walk_observe_publish`]) fires the one-shot diagnostics and folds
/// this into the presence latch. Keeping the walk free of `&mut LogOnce`
/// makes it trivially unit-testable and puts every log line at one
/// consistent layer.
#[derive(Debug, Clone, PartialEq, Eq)]
enum WalkOutcome {
    /// A readable external process holds the device fd; `pid` is the
    /// first such holder (the walk short-circuits on the first match).
    External { pid: u32 },
    /// No external holder, and every candidate `/proc/<pid>/fd` was
    /// readable — authoritatively nobody.
    CleanAbsent,
    /// No external holder found, but ≥1 candidate `/proc/<pid>/fd`
    /// returned `EACCES`, so the answer is suspect: a sandboxed or
    /// root-owned consumer looks identical to "nobody". `eacces_count`
    /// is the full count; `pids` a bounded sample for diagnostics.
    UncertainAbsent { eacces_count: u32, pids: Vec<u32> },
    /// `/proc` root itself was unreadable — detection disabled.
    Unreadable(io::ErrorKind),
}

/// Why the detector published a given [`ConsumerStatus`]. `Copy` so it
/// threads into the diagnostic line and the wake counter without
/// allocating on the common unchanged-walk path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PresenceReason {
    /// A readable external holder was found.
    External,
    /// A fully-readable `/proc` walk found nobody — authoritative absence.
    CleanAbsent,
    /// An unattributable (EACCES) walk with a zero open balance — assumed
    /// absent, but not authoritative. Distinguished from `CleanAbsent` so
    /// an operator reading the diagnostic can tell a proven absence from a
    /// fail-open guess.
    UncertainAbsent,
    /// An open we could not attribute (EACCES) with a positive open
    /// balance — the fail-open wake.
    UnattributableWake,
    /// `/proc` root unreadable — fail-open `Unknown`.
    FailOpen,
}

/// Consumer-presence latch. Pure state machine mirroring
/// [`super::state::IdleStateMachine`]'s style — the producer end of the
/// detector's status channel, kept separate from the worker-side
/// consumer FSM (they sit on opposite ends of the `AtomicU8`).
///
/// `open_balance` tracks live external opens of the loopback: inotify
/// `IN_OPEN` is `+1`, `IN_CLOSE_*` is `−1`, accumulated since the watch
/// armed. inotify counts open *file descriptions*, so `dup`/`fork` do
/// not inflate it and a crashed consumer still emits its close. Because
/// fluxframe never re-opens the loopback after startup (the output
/// write-fd is opened before the watch and held for the whole run), a
/// positive balance means an external consumer is attached — even one
/// whose `/proc/<pid>/fd` is unreadable (a sandboxed browser → EACCES).
/// The readable walk (`External`/`CleanAbsent`) re-anchors the balance
/// authoritatively, so drift is self-correcting.
#[derive(Default)]
struct ConsumerPresence {
    open_balance: i32,
}

impl ConsumerPresence {
    /// Fold one walk outcome plus the inotify open/close delta into the
    /// latch and return the published status and the reason for it.
    fn observe(
        &mut self,
        walk: &WalkOutcome,
        net_opens: i32,
        overflow: bool,
    ) -> (ConsumerStatus, PresenceReason) {
        // `overflow` means we lost events: reset the balance and let the
        // walk re-anchor. Otherwise fold the batch's net delta, clamped
        // at zero so an unmatched close (a pre-watch fd closing after
        // the watch armed) cannot drive the balance negative.
        if overflow {
            self.open_balance = 0;
        } else {
            self.open_balance = (self.open_balance + net_opens).max(0);
        }
        match walk {
            WalkOutcome::Unreadable(_) => (ConsumerStatus::Unknown, PresenceReason::FailOpen),
            WalkOutcome::External { .. } => {
                self.open_balance = self.open_balance.max(1);
                (ConsumerStatus::Present, PresenceReason::External)
            }
            WalkOutcome::CleanAbsent => {
                self.open_balance = 0;
                (ConsumerStatus::Absent, PresenceReason::CleanAbsent)
            }
            WalkOutcome::UncertainAbsent { .. } => {
                if self.open_balance > 0 {
                    (ConsumerStatus::Present, PresenceReason::UnattributableWake)
                } else {
                    (ConsumerStatus::Absent, PresenceReason::UncertainAbsent)
                }
            }
        }
    }

    /// Reset the balance to zero. Called when the inotify path falls
    /// back to polling (no more edge counting), so a stale positive
    /// balance cannot latch `Present` until the next authoritative walk.
    fn reset(&mut self) {
        self.open_balance = 0;
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
        counters: Arc<Counters>,
    ) -> Self {
        Self::spawn_with_proc_root(
            device_path,
            PathBuf::from("/proc"),
            my_pid,
            poll_interval,
            running,
            RESTAT_INTERVAL_POLLS,
            counters,
            #[cfg(test)]
            Arc::new(AtomicU64::new(0)),
        )
    }

    /// Test-only constructor that lets tests inject a tempdir as
    /// `/proc`, tune the re-arm cadence, and observe how often the
    /// inotify path activated. Production code uses [`Self::spawn`]
    /// which fixes the defaults.
    // `too_many_arguments` fires only in the test build (the extra
    // `inotify_activations` arg is `#[cfg(test)]`), so `allow` rather
    // than `expect` — the latter would be unfulfilled in non-test builds.
    #[allow(
        clippy::too_many_arguments,
        reason = "test-seam constructor mirroring the thread entry point's owned inputs; \
                  bundling them into a struct only adds noise for a test-only helper"
    )]
    pub(crate) fn spawn_with_proc_root(
        device_path: PathBuf,
        proc_root: PathBuf,
        my_pid: u32,
        poll_interval: Duration,
        running: Arc<AtomicBool>,
        restat_interval_polls: u32,
        counters: Arc<Counters>,
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
                    counters,
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
    counters: Arc<Counters>,
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
    let mut presence = ConsumerPresence::default();
    let ctx = DetectorContext {
        proc_root: &proc_root,
        my_pid,
        running: &running,
        status: &status,
        stopped: &stopped,
        restat_interval_polls,
        device_basename: &device_basename,
        counters: &counters,
        #[cfg(test)]
        inotify_activations: &inotify_activations,
    };
    let mut state = DetectorState {
        log_once: &mut log_once,
        last_status: &mut last_status,
        presence: &mut presence,
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
    /// Net open/close delta over the batch: `IN_OPEN` is `+1`,
    /// `IN_CLOSE_*` is `−1`. Feeds [`ConsumerPresence`]'s open balance.
    net_opens: i32,
    overflow: bool,
    watch_removed: bool,
}

impl EventSummary {
    /// Fold a single event's mask into the running summary.
    fn observe(&mut self, mask: inotify::EventMask) {
        if mask.contains(inotify::EventMask::Q_OVERFLOW) {
            self.overflow = true;
        }
        // OPEN and CLOSE arrive as distinct events, but fold each
        // independently so a hypothetical combined mask is still counted
        // correctly. `needs_rescan` stays the walk trigger; `net_opens`
        // is only the balance input.
        if mask.contains(inotify::EventMask::OPEN) {
            self.needs_rescan = true;
            self.net_opens += 1;
        }
        if mask.intersects(inotify::EventMask::CLOSE_WRITE | inotify::EventMask::CLOSE_NOWRITE) {
            self.needs_rescan = true;
            self.net_opens -= 1;
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
    // may be never. No inotify delta yet (`net_opens = 0`).
    walk_observe_publish(ctx, state, 0, false);

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
        walk_observe_publish(ctx, state, summary.net_opens, summary.overflow);
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
    // The polling path has no open/close edges to count, so drop any
    // balance the inotify path accumulated before falling back — a
    // stale positive balance must not latch `Present` here.
    state.presence.reset();

    while ctx.running.load(Ordering::Acquire) && !ctx.stopped.load(Ordering::Acquire) {
        // No inotify delta on the polling path (`net_opens = 0`); the
        // walk alone drives the status.
        walk_observe_publish(ctx, state, 0, false);

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

/// Walk `/proc`, fold the outcome into the presence latch, publish any
/// status change, and fire the one-shot diagnostics. This is the single
/// place the walk → observe → publish → log → metrics sequence lives;
/// the baseline walk, the inotify event path, and the polling fallback
/// all call it. `net_opens` is the inotify open/close delta for this
/// batch (`0` for the baseline and polling paths, which have no edges).
fn walk_observe_publish(
    ctx: &DetectorContext<'_>,
    state: &mut DetectorState<'_>,
    net_opens: i32,
    overflow: bool,
) {
    let walk = count_external_consumers(ctx.proc_root, ctx.device_basename, ctx.my_pid);
    let prev = *state.last_status;
    let (status, reason) = state.presence.observe(&walk, net_opens, overflow);

    // Diagnostics — both one-shot gates fire from this single site. The
    // walk is pure data; the presence-layer fields (`open_balance`,
    // `net_opens`, the decision) are only known here, so the enriched
    // EACCES line cannot live inside the walk.
    match &walk {
        WalkOutcome::Unreadable(kind) => {
            if state.log_once.scan_failed.fire() {
                warn!(
                    target: "fluxframe::idle",
                    proc_root = %ctx.proc_root.display(),
                    error = ?kind,
                    "/proc scan failed — consumer detection disabled"
                );
            }
        }
        WalkOutcome::UncertainAbsent { eacces_count, pids } => {
            if state.log_once.eacces_storm.fire() {
                // `comm` is world-readable even for a process whose
                // `fd/` dir gave EACCES (non-dumpable/sandboxed), so it
                // names the likely holder ("chrome"). Read only now —
                // the gate fires ≤ once per rearm window, off the hot
                // path.
                let comms = read_comms(ctx.proc_root, pids);
                info!(
                    target: "fluxframe::idle",
                    eacces_count = *eacces_count,
                    net_opens,
                    open_balance = state.presence.open_balance,
                    decision = ?status,
                    reason = ?reason,
                    comms = ?comms,
                    "scanned /proc; candidate fd dirs returned EACCES — a sandboxed/privileged consumer may be present"
                );
            }
        }
        WalkOutcome::External { .. } | WalkOutcome::CleanAbsent => {}
    }

    // An open we could not attribute but chose to treat as a consumer
    // (fail-open) that just flipped us to Present.
    if reason == PresenceReason::UnattributableWake && prev != Some(ConsumerStatus::Present) {
        ctx.counters.inc_unattributable_wakes();
    }

    publish_if_changed(status, ctx.status, state.last_status);
}

/// Best-effort read of `/proc/<pid>/comm` for a sample of pids, for the
/// EACCES-storm diagnostic. Failures (ENOENT for a process that exited)
/// are skipped. Called only when the gate fires.
fn read_comms(proc_root: &Path, pids: &[u32]) -> Vec<String> {
    pids.iter()
        .filter_map(|&pid| {
            std::fs::read_to_string(proc_root.join(pid.to_string()).join("comm"))
                .ok()
                .map(|s| s.trim_end().to_string())
        })
        .collect()
}

/// Walk `proc_root` (typically `/proc`) and classify whether any
/// external process holds an fd whose symlink target's basename equals
/// `device_basename`. Pure — returns a [`WalkOutcome`]; the caller fires
/// diagnostics. The walk short-circuits on the first match.
///
/// * Top-level `read_dir(proc_root)` failure → [`WalkOutcome::Unreadable`].
/// * A readable pid holding the fd → [`WalkOutcome::External`].
/// * No holder, all candidates readable → [`WalkOutcome::CleanAbsent`].
/// * No holder but ≥1 `/proc/<pid>/fd` returned `EACCES` →
///   [`WalkOutcome::UncertainAbsent`] (the multi-user / sandboxed blind
///   spot where an unprivileged fluxframe cannot see the consumer).
/// * Per-pid `ENOENT`/other I/O and `read_link` failures skip silently.
fn count_external_consumers(proc_root: &Path, device_basename: &OsStr, my_pid: u32) -> WalkOutcome {
    let entries = match std::fs::read_dir(proc_root) {
        Ok(it) => it,
        Err(e) => return WalkOutcome::Unreadable(e.kind()),
    };

    let mut eacces_count: u32 = 0;
    let mut eacces_pids: Vec<u32> = Vec::new();

    for entry in entries.flatten() {
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
        // the "candidate unreadable" blind spot. ENOENT (process exited
        // mid-scan) and any other I/O error skip silently — steady-state
        // expected on a multi-user system.
        let fd_entries = match std::fs::read_dir(&fd_dir) {
            Ok(it) => it,
            Err(e) if e.kind() == io::ErrorKind::PermissionDenied => {
                eacces_count = eacces_count.saturating_add(1);
                if eacces_pids.len() < EACCES_PID_SAMPLE {
                    eacces_pids.push(pid);
                }
                continue;
            }
            Err(_) => continue,
        };

        for fd in fd_entries.flatten() {
            let Ok(link) = std::fs::read_link(fd.path()) else {
                continue;
            };
            if link.file_name() == Some(device_basename) {
                // One matching fd settles it — no need to scan further.
                return WalkOutcome::External { pid };
            }
        }
    }

    if eacces_count > 0 {
        WalkOutcome::UncertainAbsent {
            eacces_count,
            pids: eacces_pids,
        }
    } else {
        WalkOutcome::CleanAbsent
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
    fn one_external_pid_with_matching_fd_is_external() {
        let dir = tempfile::tempdir().expect("tempdir");
        make_proc_entry(dir.path(), 1234, 3, Path::new("/dev/video10"));

        let walk = count_external_consumers(dir.path(), OsStr::new("video10"), 9999);
        assert_eq!(walk, WalkOutcome::External { pid: 1234 });
    }

    #[test]
    fn multiple_fds_same_pid_is_external() {
        // Several fds from one pid all pointing at /dev/video10 still
        // classify as a single External holder (the walk short-circuits
        // on the first match).
        let dir = tempfile::tempdir().expect("tempdir");
        make_proc_entry(dir.path(), 1234, 3, Path::new("/dev/video10"));
        make_proc_entry(dir.path(), 1234, 4, Path::new("/dev/video10"));
        make_proc_entry(dir.path(), 1234, 5, Path::new("/dev/video10"));

        let walk = count_external_consumers(dir.path(), OsStr::new("video10"), 9999);
        assert_eq!(walk, WalkOutcome::External { pid: 1234 });
    }

    #[test]
    fn self_pid_only_is_clean_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        // fluxframe's own fd should not count as a consumer.
        make_proc_entry(dir.path(), 9999, 3, Path::new("/dev/video10"));

        let walk = count_external_consumers(dir.path(), OsStr::new("video10"), 9999);
        assert_eq!(walk, WalkOutcome::CleanAbsent);
    }

    #[test]
    fn no_matching_fds_is_clean_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        // External pid but holding unrelated fds.
        make_proc_entry(dir.path(), 1234, 3, Path::new("/dev/null"));
        make_proc_entry(dir.path(), 1234, 4, Path::new("/dev/video11"));
        make_proc_entry(dir.path(), 5678, 3, Path::new("/dev/zero"));

        let walk = count_external_consumers(dir.path(), OsStr::new("video10"), 9999);
        assert_eq!(walk, WalkOutcome::CleanAbsent);
    }

    /// A pid with a chmod-0o000 fd dir is unreadable (EACCES) but a
    /// readable pid holding a matching fd still settles the walk as
    /// `External` — one found consumer trumps any EACCES noise.
    #[test]
    fn permission_denied_pid_skipped_when_consumer_found() {
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

        let walk = count_external_consumers(dir.path(), OsStr::new("video10"), 9999);
        assert_eq!(walk, WalkOutcome::External { pid: 5678 });
    }

    /// Every candidate `/proc/<pid>/fd` unreadable and no match →
    /// `UncertainAbsent` carrying the full EACCES count (the sampled
    /// pids are a bounded sample; assert only the count to stay robust
    /// against `read_dir` ordering).
    #[test]
    fn all_eacces_no_match_is_uncertain_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let unreadable_fd_dir = dir.path().join("1234").join("fd");
        fs::create_dir_all(&unreadable_fd_dir).unwrap();
        let _restore = ScopeGuard::new({
            let p = unreadable_fd_dir.clone();
            move || {
                let _ = fs::set_permissions(&p, fs::Permissions::from_mode(0o700));
            }
        });
        fs::set_permissions(&unreadable_fd_dir, fs::Permissions::from_mode(0o000)).unwrap();

        let walk = count_external_consumers(dir.path(), OsStr::new("video10"), 9999);
        assert!(
            matches!(
                walk,
                WalkOutcome::UncertainAbsent {
                    eacces_count: 1,
                    ..
                }
            ),
            "expected UncertainAbsent with one EACCES, got {walk:?}"
        );
    }

    #[test]
    fn unreadable_proc_root_is_unreadable() {
        let walk =
            count_external_consumers(Path::new("/no/such/proc/path"), OsStr::new("video10"), 9999);
        assert!(matches!(walk, WalkOutcome::Unreadable(_)), "got {walk:?}");
    }

    #[test]
    fn mixed_self_plus_external_is_external() {
        let dir = tempfile::tempdir().expect("tempdir");
        make_proc_entry(dir.path(), 9999, 3, Path::new("/dev/video10"));
        make_proc_entry(dir.path(), 1234, 3, Path::new("/dev/video10"));

        let walk = count_external_consumers(dir.path(), OsStr::new("video10"), 9999);
        assert_eq!(walk, WalkOutcome::External { pid: 1234 });
    }

    // --- ConsumerPresence::observe pure matrix -------------------------

    fn uncertain(n: u32) -> WalkOutcome {
        WalkOutcome::UncertainAbsent {
            eacces_count: n,
            pids: Vec::new(),
        }
    }

    #[test]
    fn observe_external_is_present_and_floors_balance() {
        let mut p = ConsumerPresence::default();
        // Even with a stale negative delta, External floors to >=1.
        let (s, r) = p.observe(&WalkOutcome::External { pid: 1 }, -5, false);
        assert_eq!((s, r), (ConsumerStatus::Present, PresenceReason::External));
        assert!(p.open_balance >= 1);
    }

    #[test]
    fn observe_clean_absent_zeroes_balance() {
        let mut p = ConsumerPresence { open_balance: 4 };
        let (s, r) = p.observe(&WalkOutcome::CleanAbsent, 0, false);
        assert_eq!(
            (s, r),
            (ConsumerStatus::Absent, PresenceReason::CleanAbsent)
        );
        assert_eq!(p.open_balance, 0);
    }

    #[test]
    fn observe_unreadable_is_failopen_and_keeps_counting() {
        let mut p = ConsumerPresence { open_balance: 1 };
        let (s, r) = p.observe(&WalkOutcome::Unreadable(io::ErrorKind::NotFound), 1, false);
        assert_eq!((s, r), (ConsumerStatus::Unknown, PresenceReason::FailOpen));
        // Balance still folds the delta (valid regardless of /proc).
        assert_eq!(p.open_balance, 2);
    }

    #[test]
    fn observe_uncertain_wakes_on_positive_balance() {
        let mut p = ConsumerPresence::default();
        // Open arrives (net +1) but the walk can't attribute it → wake.
        let (s, r) = p.observe(&uncertain(1), 1, false);
        assert_eq!(
            (s, r),
            (ConsumerStatus::Present, PresenceReason::UnattributableWake)
        );
    }

    #[test]
    fn observe_uncertain_probe_open_then_close_stays_absent() {
        let mut p = ConsumerPresence::default();
        // A brief probe: OPEN then CLOSE in one batch → net 0 → Absent.
        let (s, r) = p.observe(&uncertain(1), 0, false);
        assert_eq!(
            (s, r),
            (ConsumerStatus::Absent, PresenceReason::UncertainAbsent)
        );
    }

    #[test]
    fn observe_uncertain_aux_fd_close_keeps_present() {
        let mut p = ConsumerPresence { open_balance: 2 };
        // One of several held fds closes (net -1); balance 1 > 0 → still Present.
        let (s, r) = p.observe(&uncertain(1), -1, false);
        assert_eq!(
            (s, r),
            (ConsumerStatus::Present, PresenceReason::UnattributableWake)
        );
        assert_eq!(p.open_balance, 1);
    }

    #[test]
    fn observe_uncertain_steady_no_open_is_absent() {
        let mut p = ConsumerPresence::default();
        // Steady EACCES noise with no open events → stays Absent.
        let (s, r) = p.observe(&uncertain(3), 0, false);
        assert_eq!(
            (s, r),
            (ConsumerStatus::Absent, PresenceReason::UncertainAbsent)
        );
    }

    #[test]
    fn observe_overflow_resets_balance_before_deciding() {
        let mut p = ConsumerPresence { open_balance: 5 };
        // Overflow drops the balance to 0 before the walk decides; an
        // uncertain walk with the balance reset → Absent.
        let (s, _) = p.observe(&uncertain(1), 3, true);
        assert_eq!(s, ConsumerStatus::Absent);
        assert_eq!(p.open_balance, 0);
    }

    #[test]
    fn observe_balance_clamps_at_zero() {
        let mut p = ConsumerPresence::default();
        // Unmatched close (pre-watch fd) must not drive balance negative.
        let (_s, _r) = p.observe(&uncertain(1), -3, false);
        assert_eq!(p.open_balance, 0);
    }

    // --- read_comms + gate wiring --------------------------------------

    #[test]
    fn read_comms_reads_present_and_skips_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pid_dir = dir.path().join("1234");
        fs::create_dir_all(&pid_dir).unwrap();
        fs::write(pid_dir.join("comm"), "chrome\n").unwrap();
        // 1234 has a comm (trailing newline trimmed); 5678 has none → skipped.
        assert_eq!(
            read_comms(dir.path(), &[1234, 5678]),
            vec!["chrome".to_string()]
        );
    }

    /// Run one `walk_observe_publish` against `proc_root` and report which
    /// [`LogOnce`] gates remain armed afterwards — the seam that maps a
    /// [`WalkOutcome`] to its diagnostic gate now that the walk itself is
    /// gate-free.
    fn gates_after_one_walk(proc_root: &Path) -> (bool, bool) {
        let running = AtomicBool::new(true);
        let status = AtomicU8::new(ConsumerStatus::Unknown.as_u8());
        let stopped = AtomicBool::new(false);
        let counters = Counters::new();
        let device_basename = OsString::from("video10");
        let activations = AtomicU64::new(0);
        let ctx = DetectorContext {
            proc_root,
            my_pid: 9999,
            running: &running,
            status: &status,
            stopped: &stopped,
            restat_interval_polls: RESTAT_INTERVAL_POLLS,
            device_basename: &device_basename,
            counters: &counters,
            inotify_activations: &activations,
        };
        let mut log_once = LogOnce::armed();
        let mut last_status = None;
        let mut presence = ConsumerPresence::default();
        let mut state = DetectorState {
            log_once: &mut log_once,
            last_status: &mut last_status,
            presence: &mut presence,
        };
        walk_observe_publish(&ctx, &mut state, 0, false);
        (
            log_once.scan_failed.is_armed(),
            log_once.eacces_storm.is_armed(),
        )
    }

    #[test]
    fn walk_observe_publish_fires_scan_failed_on_unreadable() {
        // Unreadable /proc root → Unreadable → scan_failed consumed,
        // eacces_storm untouched.
        let (scan_armed, eacces_armed) = gates_after_one_walk(Path::new("/no/such/proc/path"));
        assert!(!scan_armed, "Unreadable walk must consume scan_failed");
        assert!(eacces_armed, "Unreadable walk must not touch eacces_storm");
    }

    #[test]
    fn walk_observe_publish_fires_eacces_storm_on_uncertain() {
        let dir = tempfile::tempdir().expect("tempdir");
        let unreadable_fd_dir = dir.path().join("1234").join("fd");
        fs::create_dir_all(&unreadable_fd_dir).unwrap();
        let _restore = ScopeGuard::new({
            let p = unreadable_fd_dir.clone();
            move || {
                let _ = fs::set_permissions(&p, fs::Permissions::from_mode(0o700));
            }
        });
        fs::set_permissions(&unreadable_fd_dir, fs::Permissions::from_mode(0o000)).unwrap();
        // All-EACCES → UncertainAbsent → eacces_storm consumed, scan_failed untouched.
        let (scan_armed, eacces_armed) = gates_after_one_walk(dir.path());
        assert!(
            scan_armed,
            "UncertainAbsent walk must not touch scan_failed"
        );
        assert!(
            !eacces_armed,
            "UncertainAbsent walk must consume eacces_storm"
        );
    }

    #[test]
    fn walk_observe_publish_fires_no_gate_on_clean_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        make_proc_entry(dir.path(), 1234, 3, Path::new("/dev/null"));
        // Fully-readable, no match → CleanAbsent → neither gate fires.
        let (scan_armed, eacces_armed) = gates_after_one_walk(dir.path());
        assert!(
            scan_armed && eacces_armed,
            "CleanAbsent must not fire any gate"
        );
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
            Arc::new(Counters::new()),
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
            Arc::new(Counters::new()),
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

    /// One-shot gate mechanics used by both diagnostic gates: `fire()`
    /// returns `true` exactly once per `armed()`/`rearm()` cycle, then
    /// `false`, and `rearm()` restores it. The diagnostics that consume
    /// these gates now live in [`walk_observe_publish`] rather than the
    /// walk; the gate contract itself is what these tests pin.
    #[test]
    fn log_once_gates_fire_once_and_rearm_in_lockstep() {
        let mut log = LogOnce::armed();
        assert!(log.scan_failed.is_armed());
        assert!(log.eacces_storm.is_armed());

        assert!(log.scan_failed.fire(), "first fire returns true");
        assert!(!log.scan_failed.is_armed(), "gate consumed after fire");
        assert!(!log.scan_failed.fire(), "second fire returns false");

        // eacces_storm is independent until rearm.
        assert!(log.eacces_storm.is_armed());
        assert!(log.eacces_storm.fire());
        assert!(!log.eacces_storm.is_armed());

        // rearm restores every gate in lockstep.
        log.rearm();
        assert!(log.scan_failed.is_armed(), "rearm restores scan_failed");
        assert!(log.eacces_storm.is_armed(), "rearm restores eacces_storm");
        assert!(log.scan_failed.fire(), "re-armed gate is consumable again");
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
            Arc::new(Counters::new()),
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
            Arc::new(Counters::new()),
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

    /// The core net-count wake: a consumer whose `/proc/<pid>/fd` is
    /// unreadable (EACCES — sandboxed browser) is invisible to the walk
    /// (`UncertainAbsent`), but a *held* `IN_OPEN` drives the open
    /// balance positive and must wake the detector to `Present`;
    /// closing it must return to `Absent`. Reproduces the incident's
    /// root cause without a real camera or Chrome by chmod-0'ing the
    /// fake `/proc/<pid>/fd` dir and holding the watched file open
    /// across an inotify drain.
    #[test]
    fn unattributable_held_open_wakes_via_net_count() {
        let proc_dir = tempfile::tempdir().expect("proc tempdir");
        let proc_root = proc_dir.path().to_path_buf();
        // One unreadable pid dir → every walk returns UncertainAbsent.
        let unreadable_fd_dir = proc_root.join("1234").join("fd");
        fs::create_dir_all(&unreadable_fd_dir).unwrap();
        let _restore = ScopeGuard::new({
            let p = unreadable_fd_dir.clone();
            move || {
                let _ = fs::set_permissions(&p, fs::Permissions::from_mode(0o700));
            }
        });
        fs::set_permissions(&unreadable_fd_dir, fs::Permissions::from_mode(0o000)).unwrap();

        let device = tempfile::NamedTempFile::new().expect("device tempfile");
        let device_path = device.path().to_path_buf();

        let running = Arc::new(AtomicBool::new(true));
        let detector = ConsumerDetector::spawn_with_proc_root(
            device_path.clone(),
            proc_root.clone(),
            9999,
            Duration::from_millis(20),
            Arc::clone(&running),
            RESTAT_INTERVAL_POLLS,
            Arc::new(Counters::new()),
            Arc::new(AtomicU64::new(0)),
        );
        let status = detector.status_handle();

        // Baseline: UncertainAbsent with a zero open balance → Absent.
        assert!(
            wait_for_status(&status, ConsumerStatus::Absent, Duration::from_millis(500)),
            "uncertain-absent with zero balance must read Absent"
        );

        // Hold the device open across a drain: IN_OPEN (+1), no matching
        // close yet → balance 1 → Present despite the EACCES walk.
        let held = std::fs::File::open(&device_path).expect("open device");
        assert!(
            wait_for_status(&status, ConsumerStatus::Present, Duration::from_millis(500)),
            "held open on an unattributable consumer must wake to Present"
        );

        // Close → IN_CLOSE_NOWRITE (−1) → balance 0 → Absent.
        drop(held);
        assert!(
            wait_for_status(&status, ConsumerStatus::Absent, Duration::from_millis(500)),
            "closing the held fd must return to Absent"
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
            Arc::new(Counters::new()),
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

    /// Tiny RAII guard so the EACCES tests can restore a 0o000 dir's
    /// permissions before the tempdir drops (some platforms refuse to
    /// remove a 0o000 directory). Pulled in here rather than as a dep —
    /// `defer-rs` / `scopeguard` would be overkill for one call site.
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
