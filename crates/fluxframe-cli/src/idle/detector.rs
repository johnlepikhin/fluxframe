// Stage 15 Step 4 wires the detector into the supervisor; until then
// every item here is dead code from the binary target's perspective.
// The `cfg_attr(not(test), expect(...))` form keeps tests warning-free
// while the binary build's expectation fires the moment Step 4 starts
// consuming the module.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Stage 15 Step 4 wires the detector into the supervisor"
    )
)]

//! Sysfs-based v4l2loopback consumer presence detector.
//!
//! Linux-only — v4l2loopback exposes the sysfs `state` attribute only
//! on Linux. On non-Linux targets this module is not compiled, and the
//! supervisor (Step 4) gates its wiring through the same `cfg`.
//!
//! v4l2loopback exposes a `state` attribute under
//! `/sys/class/video4linux/<videoN>/state` that reads either
//! `"capture"` (a reader has called `STREAMON`) or `"output"` (the
//! producer is the only fd opened against the device). The detector
//! polls this file at [`IdleConfig::poll_interval_ms`] and publishes
//! the observed value into a shared `Arc<AtomicU8>` that the
//! supervisor's idle state machine consults on every tick.
//!
//! ## Failure modes
//!
//! * **File missing (`ENOENT`)** — older v4l2loopback or a
//!   `fakesink`/`testsrc` sink configuration. Logged once at `info`
//!   level (operator-actionable but not an error), status pinned to
//!   [`ConsumerStatus::Unknown`]. The state machine treats
//!   `Unknown` as `Present` (fail-open), so idle never fires.
//! * **Permission denied (`EACCES`)** — sysfs file readable by root
//!   only on some setups. Logged once at `warn`, same fallback.
//! * **Other I/O errors / unexpected payload** — logged once at
//!   `warn` with the raw error, same fallback.
//!
//! Each error category logs only on first occurrence (one-shot
//! `LogOnce` flags) to keep steady-state logs free of noise; a
//! re-stat every [`RESTAT_INTERVAL_POLLS`] iterations clears the
//! flags so a `modprobe -r v4l2loopback && modprobe v4l2loopback`
//! cycle (which re-creates `videoN` with a fresh inode) re-arms the
//! detector.
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
//! `poll_interval` is — see [`Self::SHUTDOWN_POLL_GRANULARITY`].
//!
//! [`IdleConfig::poll_interval_ms`]: fluxframe_core::IdleConfig::poll_interval_ms

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use fluxframe_gst::output::OutputSink;
use tracing::{debug, info, warn};

use super::state::ConsumerStatus;

/// Number of poll iterations between sysfs re-stats. With the default
/// 250 ms poll cadence this is one re-stat every 15 s — fast enough
/// to pick up a re-modprobed `videoN`, slow enough not to flood the
/// kernel with redundant `read(2)`s.
const RESTAT_INTERVAL_POLLS: u32 = 60;

/// Sysfs root for v4l2loopback `state` attribute. The class path is
/// canonical and stable across udev rule changes — the underlying
/// `devices/virtual/video4linux/...` location is an implementation
/// detail of the kernel layout.
const SYSFS_CLASS_ROOT: &str = "/sys/class/video4linux";

/// Granularity of the worker's between-poll sleep. The loop wakes at
/// least this often to re-check `running` / `stopped`, capping
/// shutdown latency at this value regardless of `poll_interval`.
const SHUTDOWN_POLL_GRANULARITY: Duration = Duration::from_millis(50);

/// Resolve `/sys/class/video4linux/<basename(device)>/state` for a
/// V4L2 output sink. Returns `None` for non-V4L2 sinks
/// (`Fake` / `Auto` / `Pipewire`) — the supervisor uses the
/// `Option` to gate detector spawn.
#[must_use]
pub(crate) fn sysfs_state_path(sink: &OutputSink) -> Option<PathBuf> {
    let OutputSink::V4l2Loopback { device } = sink else {
        return None;
    };
    let basename = device.file_name()?;
    Some(PathBuf::from(SYSFS_CLASS_ROOT).join(basename).join("state"))
}

/// One-shot diagnostic gates for each `read_state` failure category.
/// Each field starts `true` (armed); the first occurrence of the
/// matching error flips it to `false` so the diagnostic logs only
/// once per re-stat window.
///
/// Clippy's `struct_excessive_bools` lint flags 4-bool structs as a
/// candidate for a state machine or enum, but here each field
/// represents an independent, orthogonal one-shot gate — there is no
/// single "state" to encode as variants. A flag set would also work
/// but adds bit-twiddling without a readability gain.
#[allow(
    clippy::struct_excessive_bools,
    reason = "four orthogonal one-shot diagnostic gates, not a state"
)]
#[derive(Default)]
struct LogOnce {
    notfound: bool,
    permission: bool,
    io_other: bool,
    unexpected: bool,
}

impl LogOnce {
    /// All categories armed — log on next occurrence of each.
    fn armed() -> Self {
        Self {
            notfound: true,
            permission: true,
            io_other: true,
            unexpected: true,
        }
    }

    /// Re-arm every category (called on re-stat).
    fn rearm(&mut self) {
        *self = Self::armed();
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
}

impl ConsumerDetector {
    /// Spawn the detector thread polling `sysfs_path` every
    /// `poll_interval`. The thread shuts down when either `running`
    /// flips to `false` or the returned handle is dropped.
    ///
    /// The `Unknown` initial status is reset to the first observed
    /// value on the first poll — the supervisor's state machine
    /// treats `Unknown` as `Present`, so the first tick is a no-op
    /// (Active stays Active) regardless of what the file reads.
    pub(crate) fn spawn(
        sysfs_path: PathBuf,
        poll_interval: Duration,
        running: Arc<AtomicBool>,
    ) -> Self {
        Self::spawn_with_restat(sysfs_path, poll_interval, running, RESTAT_INTERVAL_POLLS)
    }

    /// Test-only constructor that lets tests tune the re-stat cadence.
    /// Production code uses [`Self::spawn`] which fixes this at
    /// [`RESTAT_INTERVAL_POLLS`].
    pub(crate) fn spawn_with_restat(
        sysfs_path: PathBuf,
        poll_interval: Duration,
        running: Arc<AtomicBool>,
        restat_interval_polls: u32,
    ) -> Self {
        let status = Arc::new(AtomicU8::new(ConsumerStatus::Unknown.as_u8()));
        let stopped = Arc::new(AtomicBool::new(false));
        let status_for_thread = Arc::clone(&status);
        let stopped_for_thread = Arc::clone(&stopped);
        let handle = thread::Builder::new()
            .name("fluxframe-detector".into())
            .spawn(move || {
                run_detector(
                    sysfs_path,
                    poll_interval,
                    running,
                    status_for_thread,
                    stopped_for_thread,
                    restat_interval_polls,
                );
            })
            .expect("spawn detector thread");
        Self {
            handle: Some(handle),
            status,
            stopped,
        }
    }

    /// Cheap clone of the shared status atomic. The worker reads this
    /// every loop iteration; only the detector writes to it. Ordering
    /// is `Acquire` on the worker side, `Release` on the detector
    /// side — see the comment in [`run_detector`] for the rationale.
    pub(crate) fn status_handle(&self) -> Arc<AtomicU8> {
        Arc::clone(&self.status)
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

#[allow(
    clippy::needless_pass_by_value,
    reason = "thread entry-point: values are moved into the spawned thread on construction; \
              taking them by reference would require lifetime gymnastics across the thread \
              boundary"
)]
fn run_detector(
    sysfs_path: PathBuf,
    poll_interval: Duration,
    running: Arc<AtomicBool>,
    status: Arc<AtomicU8>,
    stopped: Arc<AtomicBool>,
    restat_interval_polls: u32,
) {
    info!(
        target: "fluxframe::idle",
        path = %sysfs_path.display(),
        poll_ms = poll_interval.as_millis() as u64,
        "consumer detector started"
    );

    let mut log_once = LogOnce::armed();
    let mut last_status: Option<ConsumerStatus> = None;
    let mut polls_since_restat: u32 = 0;

    while running.load(Ordering::Acquire) && !stopped.load(Ordering::Acquire) {
        let new_status = read_state(&sysfs_path, &mut log_once);

        if Some(new_status) != last_status {
            // `Release` here pairs with the worker's `Acquire` load
            // in `IdleStateMachine::tick` — guarantees the timer
            // restart on the worker side is ordered after the
            // observed transition.
            status.store(new_status.as_u8(), Ordering::Release);
            if matches!(new_status, ConsumerStatus::Present | ConsumerStatus::Absent) {
                // Promoted to `info` (rare event, no spam risk) so
                // the operator can see attach/detach transitions
                // with `RUST_LOG=info` without enabling debug-level
                // globally. Unknown ↔ {Present,Absent} transitions
                // stay below the log threshold by virtue of the
                // matches! gate above.
                info!(
                    target: "fluxframe::idle",
                    from = ?last_status,
                    to = ?new_status,
                    "consumer status changed"
                );
            } else {
                // Keep an explicit debug! for Unknown so operators
                // running at debug-level still see the transition.
                debug!(
                    target: "fluxframe::idle",
                    from = ?last_status,
                    to = ?new_status,
                    "consumer status changed"
                );
            }
            last_status = Some(new_status);
        }

        polls_since_restat = polls_since_restat.wrapping_add(1);
        if polls_since_restat >= restat_interval_polls {
            // Clear the one-shot log flags so a re-modprobed
            // v4l2loopback (fresh inode for `videoN`) re-arms the
            // diagnostic surface. Status is re-read next iteration.
            log_once.rearm();
            polls_since_restat = 0;
        }

        // Bounded shutdown latency: chunk the sleep so we re-check
        // `running` / `stopped` at least every
        // `SHUTDOWN_POLL_GRANULARITY`. Without this cap, a multi-
        // second `poll_interval` would delay Ctrl-C / Drop for the
        // full interval.
        let mut remaining = poll_interval;
        while remaining > Duration::ZERO
            && running.load(Ordering::Acquire)
            && !stopped.load(Ordering::Acquire)
        {
            let chunk = remaining.min(SHUTDOWN_POLL_GRANULARITY);
            thread::sleep(chunk);
            remaining = remaining.saturating_sub(chunk);
        }
    }

    info!(target: "fluxframe::idle", "consumer detector exiting");
}

fn read_state(sysfs_path: &Path, log_once: &mut LogOnce) -> ConsumerStatus {
    match std::fs::read_to_string(sysfs_path) {
        Ok(s) => match s.trim() {
            "capture" => ConsumerStatus::Present,
            "output" => ConsumerStatus::Absent,
            other => {
                if log_once.unexpected {
                    warn!(
                        target: "fluxframe::idle",
                        path = %sysfs_path.display(),
                        value = other,
                        "sysfs state has unexpected value; treating as Unknown"
                    );
                    log_once.unexpected = false;
                }
                ConsumerStatus::Unknown
            }
        },
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            if log_once.notfound {
                info!(
                    target: "fluxframe::idle",
                    path = %sysfs_path.display(),
                    "sysfs state file missing — idle detection disabled (older v4l2loopback?)"
                );
                log_once.notfound = false;
            }
            ConsumerStatus::Unknown
        }
        Err(e) if e.kind() == io::ErrorKind::PermissionDenied => {
            if log_once.permission {
                warn!(
                    target: "fluxframe::idle",
                    path = %sysfs_path.display(),
                    "sysfs state file unreadable (permission denied) — idle detection disabled"
                );
                log_once.permission = false;
            }
            ConsumerStatus::Unknown
        }
        Err(e) => {
            if log_once.io_other {
                warn!(
                    target: "fluxframe::idle",
                    path = %sysfs_path.display(),
                    error = %e,
                    "sysfs state read failed — idle detection disabled"
                );
                log_once.io_other = false;
            }
            ConsumerStatus::Unknown
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
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

    #[test]
    fn sysfs_state_path_for_v4l2_loopback() {
        let sink = OutputSink::V4l2Loopback {
            device: PathBuf::from("/dev/video10"),
        };
        let path = sysfs_state_path(&sink).expect("v4l2 sink yields a sysfs path");
        assert_eq!(path, PathBuf::from("/sys/class/video4linux/video10/state"));
    }

    #[test]
    fn sysfs_state_path_none_for_non_v4l2_sinks() {
        assert!(sysfs_state_path(&OutputSink::Fake).is_none());
        assert!(sysfs_state_path(&OutputSink::Auto).is_none());
        assert!(
            sysfs_state_path(&OutputSink::Pipewire {
                node_name: Some("fluxframe".into())
            })
            .is_none()
        );
    }

    /// Walk the full transition cycle expected at runtime:
    /// `output` → `capture` → `output` → file removed (Unknown) →
    /// file restored. Each transition must surface on the shared
    /// atomic within the poll window.
    #[test]
    fn detector_cycles_through_states_via_fake_sysfs_file() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("state");
        fs::write(&path, "output\n").expect("seed file");

        let running = Arc::new(AtomicBool::new(true));
        let detector = ConsumerDetector::spawn(
            path.clone(),
            Duration::from_millis(20),
            Arc::clone(&running),
        );
        let status = detector.status_handle();

        assert!(
            wait_for_status(&status, ConsumerStatus::Absent, Duration::from_millis(500)),
            "initial \"output\" must surface as Absent"
        );

        fs::write(&path, "capture\n").expect("flip to capture");
        assert!(
            wait_for_status(&status, ConsumerStatus::Present, Duration::from_millis(500)),
            "\"capture\" must surface as Present"
        );

        fs::write(&path, "output\n").expect("flip back to output");
        assert!(
            wait_for_status(&status, ConsumerStatus::Absent, Duration::from_millis(500)),
            "back-to-\"output\" must surface as Absent"
        );

        fs::remove_file(&path).expect("simulate modprobe -r");
        assert!(
            wait_for_status(&status, ConsumerStatus::Unknown, Duration::from_millis(500)),
            "missing file must surface as Unknown"
        );

        fs::write(&path, "capture\n").expect("simulate modprobe +");
        assert!(
            wait_for_status(&status, ConsumerStatus::Present, Duration::from_millis(500)),
            "restored \"capture\" must surface as Present"
        );

        running.store(false, Ordering::Release);
        drop(detector);
    }

    #[test]
    fn missing_file_publishes_unknown_without_panic() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("does_not_exist");

        let running = Arc::new(AtomicBool::new(true));
        let detector = ConsumerDetector::spawn(
            path.clone(),
            Duration::from_millis(20),
            Arc::clone(&running),
        );
        let status = detector.status_handle();

        assert!(
            wait_for_status(&status, ConsumerStatus::Unknown, Duration::from_millis(300)),
            "ENOENT must surface as Unknown"
        );

        running.store(false, Ordering::Release);
        drop(detector);
    }

    #[test]
    fn unexpected_payload_publishes_unknown() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let path = dir.path().join("state");
        fs::write(&path, "weird-value\n").expect("seed file");

        let running = Arc::new(AtomicBool::new(true));
        let detector = ConsumerDetector::spawn(
            path.clone(),
            Duration::from_millis(20),
            Arc::clone(&running),
        );
        let status = detector.status_handle();

        assert!(
            wait_for_status(&status, ConsumerStatus::Unknown, Duration::from_millis(300)),
            "unrecognised sysfs payload must surface as Unknown"
        );

        running.store(false, Ordering::Release);
        drop(detector);
    }

    /// `read_state` is the pure inner function — unit-test it
    /// directly without spinning up a thread to keep the contract
    /// table-driven.
    #[test]
    fn read_state_classifies_each_payload() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state");

        fs::write(&path, "capture\n").unwrap();
        assert_eq!(
            read_state(&path, &mut LogOnce::armed()),
            ConsumerStatus::Present
        );

        fs::write(&path, "output").unwrap(); // no trailing newline either
        assert_eq!(
            read_state(&path, &mut LogOnce::armed()),
            ConsumerStatus::Absent
        );

        fs::write(&path, "unknown\n").unwrap();
        assert_eq!(
            read_state(&path, &mut LogOnce::armed()),
            ConsumerStatus::Unknown
        );

        fs::remove_file(&path).unwrap();
        assert_eq!(
            read_state(&path, &mut LogOnce::armed()),
            ConsumerStatus::Unknown
        );
    }

    /// Verifies the re-arm semantics that `run_detector` relies on
    /// every `restat_interval_polls` iterations. Synchronous unit
    /// test — no thread, no `tracing-subscriber` capture needed.
    /// The visible contract is: after a one-shot flag fires (going
    /// from `true` to `false`), `rearm()` flips it back to `true`,
    /// and a subsequent matching error will fire it again.
    #[test]
    fn log_once_rearm_re_arms_all_categories() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("missing");

        let mut log_once = LogOnce::armed();
        assert!(log_once.notfound);
        assert!(log_once.permission);
        assert!(log_once.io_other);
        assert!(log_once.unexpected);

        // First missing-file read flips `notfound` to false.
        assert_eq!(read_state(&path, &mut log_once), ConsumerStatus::Unknown);
        assert!(!log_once.notfound, "first ENOENT consumed the gate");

        // A second read does NOT re-flip — gate is consumed.
        assert_eq!(read_state(&path, &mut log_once), ConsumerStatus::Unknown);
        assert!(!log_once.notfound, "gate stays consumed across reads");

        // `rearm` resets every category, including notfound.
        log_once.rearm();
        assert!(log_once.notfound, "rearm restores notfound");
        assert!(log_once.permission, "rearm restores permission");
        assert!(log_once.io_other, "rearm restores io_other");
        assert!(log_once.unexpected, "rearm restores unexpected");

        // Now a third missing-file read consumes it again — i.e. the
        // detector would emit the diagnostic a second time after the
        // re-stat boundary, which is exactly Fix #8's contract.
        assert_eq!(read_state(&path, &mut log_once), ConsumerStatus::Unknown);
        assert!(!log_once.notfound, "re-armed gate is consumable again");
    }

    /// Shutdown latency is bounded by `SHUTDOWN_POLL_GRANULARITY`
    /// (~50 ms), NOT by `poll_interval`. The 500 ms tolerance leaves
    /// CI plenty of slack to avoid flakiness.
    #[test]
    fn shutdown_completes_quickly_even_with_long_poll_interval() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("state");
        fs::write(&path, "output\n").unwrap();

        let running = Arc::new(AtomicBool::new(true));
        let detector = ConsumerDetector::spawn(
            path,
            Duration::from_secs(2), // long poll
            Arc::clone(&running),
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
}
