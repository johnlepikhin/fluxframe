//! Per-role CPU accounting for the metrics reporter.
//!
//! Stage timings are wall-clock: they cannot tell whether the 4–5 ms
//! the worker spends inside `infer()` is a sleep on the NPU or a
//! spin, and they say nothing about the GStreamer conversion threads
//! on either side of the effect chain.  This module samples the CPU
//! time of every thread in the process from `/proc/self/task/*/stat`
//! once per reporter tick and folds the deltas into
//! [`CpuShares`] by thread role.
//!
//! Roles are recognised by thread name (`comm`, 15 bytes):
//!
//! | role     | threads                                                      |
//! |----------|--------------------------------------------------------------|
//! | worker   | the main thread — it runs the frame loop                     |
//! | rayon    | `rayon-*` (named in `cap_rayon_pool`)                        |
//! | input    | GStreamer capture streaming threads (`input_*`)              |
//! | output   | GStreamer output threads (`output_*`) + `fluxframe-out-writer` |
//! | other    | everything else                                              |
//!
//! Linux only by construction (`/proc`); on any other host
//! [`ThreadCpuSampler::sample`] returns `None` and the reporter simply
//! omits the line.  Reading `/proc` is plain `std::fs`, which keeps the
//! workspace's `forbid(unsafe_code)` intact — `getrusage(RUSAGE_THREAD)`
//! would need an FFI call.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use fluxframe_core::metrics::CpuShares;

/// Clock ticks per second for the `utime` / `stime` fields of
/// `/proc/<pid>/task/<tid>/stat`.  The kernel reports them in
/// `USER_HZ`, which is 100 on every Linux architecture regardless of
/// the scheduler's `CONFIG_HZ`; `sysconf(_SC_CLK_TCK)` would return the
/// same constant but needs libc.
const USER_HZ: f64 = 100.0;

/// Thread role a `/proc` `comm` maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Worker,
    Rayon,
    Input,
    Output,
    Other,
}

/// One thread's CPU reading: its role and cumulative ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ThreadTicks {
    role: Role,
    ticks: u64,
}

/// Samples per-thread CPU time and turns consecutive samples into
/// per-window shares.
#[derive(Debug)]
pub(crate) struct ThreadCpuSampler {
    prev: HashMap<u64, ThreadTicks>,
    prev_at: Instant,
    pid: u64,
}

impl ThreadCpuSampler {
    /// Take the initial baseline so the first tick already reports a
    /// proper delta.  Returns `None` when `/proc` is not readable.
    pub(crate) fn new() -> Option<Self> {
        let pid = std::process::id().into();
        let prev = read_thread_ticks(pid)?;
        Some(Self {
            prev,
            prev_at: Instant::now(),
            pid,
        })
    }

    /// CPU shares since the previous call (or since construction).
    /// `None` when `/proc` could not be read this time or the window
    /// is empty; the baseline is refreshed either way.
    pub(crate) fn sample(&mut self) -> Option<CpuShares> {
        let now = Instant::now();
        let window = now.duration_since(self.prev_at);
        let cur = read_thread_ticks(self.pid);
        self.prev_at = now;
        let cur = cur?;
        let shares = shares_between(&self.prev, &cur, window);
        self.prev = cur;
        shares
    }
}

/// Fold two samples into per-role shares over `window`.  A thread
/// present only in `cur` started inside the window, so its whole
/// count is attributed; a thread present only in `prev` exited, and
/// its final ticks are unknown — they are dropped (a bounded
/// undercount for short-lived threads).
fn shares_between(
    prev: &HashMap<u64, ThreadTicks>,
    cur: &HashMap<u64, ThreadTicks>,
    window: Duration,
) -> Option<CpuShares> {
    let secs = window.as_secs_f64();
    if secs <= 0.0 {
        return None;
    }
    let mut shares = CpuShares::default();
    for (tid, now) in cur {
        let before = prev.get(tid).map_or(0, |p| p.ticks);
        let pct = now.ticks.saturating_sub(before) as f64 / USER_HZ / secs * 100.0;
        shares.process += pct;
        let slot = match now.role {
            Role::Worker => &mut shares.worker,
            Role::Rayon => &mut shares.rayon,
            Role::Input => &mut shares.input,
            Role::Output => &mut shares.output,
            Role::Other => &mut shares.other,
        };
        *slot += pct;
    }
    Some(shares)
}

/// Read every thread's role and cumulative ticks.  `None` when the
/// task directory cannot be listed at all; individual threads that
/// vanish mid-read are skipped.
fn read_thread_ticks(pid: u64) -> Option<HashMap<u64, ThreadTicks>> {
    let entries = std::fs::read_dir("/proc/self/task").ok()?;
    let mut out = HashMap::new();
    for entry in entries.flatten() {
        let Some(tid) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<u64>().ok())
        else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        if let Some((comm, ticks)) = parse_stat(&stat) {
            let role = if tid == pid {
                Role::Worker
            } else {
                classify(comm)
            };
            out.insert(tid, ThreadTicks { role, ticks });
        }
    }
    Some(out)
}

/// Extract `comm` and `utime + stime` from one `/proc/.../stat` line.
///
/// The line is `pid (comm) state ppid …`; `comm` may itself contain
/// spaces or parentheses, so the split is on the *last* `)`.  After
/// it, `utime` and `stime` are the 12th and 13th whitespace-separated
/// fields (overall fields 14 and 15).
fn parse_stat(stat: &str) -> Option<(&str, u64)> {
    let open = stat.find('(')?;
    let close = stat.rfind(')')?;
    let comm = stat.get(open + 1..close)?;
    let mut rest = stat.get(close + 1..)?.split_ascii_whitespace();
    let utime: u64 = rest.nth(11)?.parse().ok()?;
    let stime: u64 = rest.next()?.parse().ok()?;
    Some((comm, utime + stime))
}

/// Map a thread name to its role (main thread is handled by tid).
fn classify(comm: &str) -> Role {
    if comm.starts_with("rayon-") {
        Role::Rayon
    } else if comm.starts_with("input") {
        Role::Input
    } else if comm.starts_with("output") || comm.starts_with("fluxframe-out") {
        Role::Output
    } else {
        Role::Other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stat_reads_comm_and_cpu_ticks() {
        // Real shape: fields 14/15 are utime/stime = 120 + 30.
        let line = "4204 (.fluxframe-real) S 7170 4204 4204 0 -1 4194560 12 0 0 0 120 30 0 0 20 0 34 0 1 2 3 4";
        assert_eq!(parse_stat(line), Some((".fluxframe-real", 150)));
    }

    #[test]
    fn parse_stat_tolerates_parentheses_in_comm() {
        let line = "1 (a (b) c) R 0 0 0 0 0 0 0 0 0 0 5 7 0 0 0 0 0 0 0 0 0 0";
        assert_eq!(parse_stat(line), Some(("a (b) c", 12)));
    }

    #[test]
    fn parse_stat_rejects_short_lines() {
        assert_eq!(parse_stat("1 (x) R 0"), None);
        assert_eq!(parse_stat("garbage"), None);
    }

    #[test]
    fn classify_by_thread_name() {
        assert_eq!(classify("rayon-3"), Role::Rayon);
        assert_eq!(classify("input_queue:src"), Role::Input);
        assert_eq!(classify("output_queue:sr"), Role::Output);
        assert_eq!(classify("fluxframe-out-w"), Role::Output);
        assert_eq!(classify("fluxframe-metri"), Role::Other);
        assert_eq!(classify(".fluxframe-real"), Role::Other);
    }

    #[test]
    fn shares_are_percent_of_one_core_over_the_window() {
        let t = |role, ticks| ThreadTicks { role, ticks };
        let prev = HashMap::from([(1, t(Role::Worker, 100)), (2, t(Role::Rayon, 50))]);
        // Worker +50 ticks, rayon +25, plus a rayon thread born in the
        // window with 25 ticks, over 5 s: 10% + 5% + 5%.
        let cur = HashMap::from([
            (1, t(Role::Worker, 150)),
            (2, t(Role::Rayon, 75)),
            (3, t(Role::Rayon, 25)),
        ]);
        let s = shares_between(&prev, &cur, Duration::from_secs(5)).expect("window > 0");
        assert!((s.worker - 10.0).abs() < 1e-9);
        assert!((s.rayon - 10.0).abs() < 1e-9);
        assert!((s.process - 20.0).abs() < 1e-9);
        assert!(s.input.abs() < 1e-9);
    }

    #[test]
    fn zero_window_yields_none() {
        assert!(shares_between(&HashMap::new(), &HashMap::new(), Duration::ZERO).is_none());
    }

    #[test]
    fn live_sampler_reads_this_process() {
        // Sanity check against the real /proc: this test binary has at
        // least its own main thread, which is classified as the worker.
        let Some(mut sampler) = ThreadCpuSampler::new() else {
            eprintln!("no /proc — skipping");
            return;
        };
        std::thread::sleep(Duration::from_millis(20));
        let shares = sampler.sample().expect("window elapsed");
        assert!(shares.process >= 0.0);
        assert!(shares.worker >= 0.0);
    }
}
