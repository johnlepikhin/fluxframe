//! Idle-mode state machine.
//!
//! Pure function over (current state, observed consumer status,
//! `now: Instant`, [`IdleConfig`]). No I/O, no global clock — the
//! worker injects the timestamp it observed. This makes every
//! transition reproducible in unit tests with synthetic timestamps.
//!
//! ## State diagram
//!
//! ```text
//!                 consumer Present ("capture")
//!                ┌──────────────────────────────────────┐
//!                │                                      │
//!                ▼                                      │
//!        ┌───────────────┐  Absent 5 s        ┌────────┴────────┐
//!        │    Active     │ ───────────────►   │      Idle       │
//!        │ (full chain)  │                    │ (input Null,    │
//!        │ engine_ready  │ ◄──────────────    │  placeholder,   │
//!        └───────┬───────┘   Present          │  engine warm)   │
//!                │ Absent                     └────────┬────────┘
//!                │ (5s cooldown timer)                 │ Absent 30 s
//!                ▼                                     ▼
//!        ┌───────────────┐                    ┌─────────────────┐
//!        │   Cooldown    │                    │    DeepIdle     │
//!        │ (still full,  │                    │ (ONNX dropped,  │
//!        │  flips soon)  │                    │  placeholder)   │
//!        └───────────────┘                    └─────────────────┘
//! ```
//!
//! Cooldown is observably identical to Active from the outside — it
//! just delays the flip to Idle so a consumer reopen within 5 s never
//! sees input torn down. A `Present` observation at any depth cancels
//! the cooldown timer and triggers `ResumeActive`.

use std::time::{Duration, Instant};

use fluxframe_core::IdleConfig;

/// Observed consumer status from the sysfs detector. The numeric
/// values are stable because Step 3 stores them in an `AtomicU8`.
///
/// ## Invariants
///
/// - `#[repr(u8)]` guarantees ABI stability of the discriminants across
///   rustc versions, so the `AtomicU8` payload survives recompiles.
/// - [`Self::from_u8`] is total: every `u8` maps to a defined variant
///   (unknown discriminants collapse to [`ConsumerStatus::Unknown`]).
///   This is the fail-open contract — a corrupted or future-tagged
///   write never panics the worker.
/// - The Step 3 detector → worker channel is single-producer /
///   single-consumer and carries advisory data only, so `Relaxed`
///   ordering would suffice in principle. The detector publishes
///   with `Release` and the worker loads with `Acquire` as a
///   defensive default: cost on x86_64 is identical to `Relaxed`,
///   and the stronger ordering simplifies correctness proofs if
///   future state is ever piggybacked on the status atomic.
/// - [`IdleStateMachine`] is owned exclusively by the worker thread.
///   The atomic only crosses the detector → worker boundary; the
///   state machine itself never sees concurrent access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum ConsumerStatus {
    /// At least one reader is attached and streaming (sysfs reads
    /// `"capture"`).
    Present = 1,
    /// No reader streaming (sysfs reads `"output"`).
    Absent = 0,
    /// Sysfs file unreadable (missing, permission denied, or other I/O
    /// error). Treated as `Present` by the state machine — fail-open so
    /// idle never fires on a misconfigured detector.
    Unknown = 2,
}

impl ConsumerStatus {
    /// Decode an `AtomicU8` value. Unknown numeric values fall back to
    /// [`ConsumerStatus::Unknown`].
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Stage 15 Step 4 wires the supervisor")
    )]
    pub(crate) fn from_u8(raw: u8) -> Self {
        match raw {
            0 => ConsumerStatus::Absent,
            1 => ConsumerStatus::Present,
            _ => ConsumerStatus::Unknown,
        }
    }

    /// Encode for the `AtomicU8`.
    pub(crate) fn as_u8(self) -> u8 {
        self as u8
    }

    /// Should the state machine treat this status as "consumer is
    /// there"? `Present` and `Unknown` are both fail-open; only
    /// `Absent` advances the cooldown.
    fn is_present_or_unknown(self) -> bool {
        !matches!(self, ConsumerStatus::Absent)
    }
}

/// Discrete lifecycle state. The state machine internally tracks the
/// instant of the last transition to compute cooldown elapsed time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IdleState {
    /// Full processing pipeline running.
    Active,
    /// Consumer just disconnected; the supervisor is waiting out
    /// `teardown_secs` before flipping to Idle. Externally identical
    /// to Active.
    Cooldown,
    /// Input torn down, placeholder published. ONNX session still
    /// resident.
    Idle,
    /// Input torn down, ONNX session dropped, placeholder published.
    DeepIdle,
}

/// One-shot side effect fired at a state transition. The worker
/// dispatches input pipeline state changes, reload-thread spawns,
/// and the placeholder-flush on `ResumeActive` based on this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IdleEdge {
    /// No transition happened on this tick.
    None,
    /// Active/Cooldown → Idle. Tear down input, flush stale frames.
    EnterIdle,
    /// Idle → DeepIdle. Drop the ONNX session.
    EnterDeepIdle,
    /// Any depth → Active. Spawn the reload thread; the worker keeps
    /// publishing placeholder until `engine_ready` flips true.
    ResumeActive,
}

/// Steady-state behaviour for this iteration. The worker uses this
/// to choose between `process_one_frame` (Active) and
/// `push_placeholder` (Placeholder).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IdleLevel {
    /// Pull a frame from the slot and run the effect chain.
    Active,
    /// Push the cached placeholder.
    Placeholder,
}

/// Tick output. Edge fires once per transition; level describes the
/// steady-state for the next loop iteration; `next_tick_in` is the
/// upper bound the worker should park before re-evaluating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IdleTick {
    pub edge: IdleEdge,
    pub level: IdleLevel,
    pub next_tick_in: Duration,
}

/// Normalised, structurally exhaustive view of [`ConsumerStatus`]
/// used inside [`IdleStateMachine::tick`]. Unknown collapses to
/// `Present` at the call site (fail-open), so the match in `tick`
/// covers all `(IdleState, Effective)` pairs without a guard.
///
/// Private to the module — never appears in the public surface.
enum Effective {
    Present,
    Absent,
}

/// The state machine. Field visibility is `pub(crate)` only on the
/// public surface needed by the worker; the timer is internal.
///
/// Owned exclusively by the worker thread — never shared. The detector
/// publishes status through an `AtomicU8`; the worker decodes it once
/// per tick and feeds it here.
#[derive(Debug)]
pub(crate) struct IdleStateMachine {
    state: IdleState,
    /// Instant of the last state transition. Used to compute the
    /// cooldown elapsed time. Initialised to the construction instant
    /// so the first tick has a sensible baseline.
    last_change: Instant,
}

impl IdleStateMachine {
    /// New state machine starting in [`IdleState::Active`].
    ///
    /// `now` is injected so unit tests can fix the initial baseline.
    /// The production caller passes `Instant::now()` once at startup.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Stage 15 Step 4 wires the supervisor")
    )]
    pub(crate) fn new(now: Instant) -> Self {
        Self {
            state: IdleState::Active,
            last_change: now,
        }
    }

    /// Current state — exposed for telemetry / log lines only. The
    /// worker should never branch on this directly; use the
    /// `IdleTick::level` returned by [`Self::tick`].
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Stage 15 Step 4 wires the supervisor")
    )]
    pub(crate) fn state(&self) -> IdleState {
        self.state
    }

    /// Advance the state machine.
    ///
    /// Pure function of (current state, observed status, elapsed
    /// since last transition, config). Returns the side-effect edge,
    /// the steady-state level for the next loop iteration, and a hint
    /// for how long the worker should park before re-evaluating.
    ///
    /// Cooldown/deep-idle thresholds come from `cfg` so a runtime
    /// `reload` of the config picks them up on the next tick without
    /// reconstructing the machine.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Stage 15 Step 4 wires the supervisor")
    )]
    pub(crate) fn tick(
        &mut self,
        status: ConsumerStatus,
        now: Instant,
        cfg: &IdleConfig,
    ) -> IdleTick {
        let elapsed = now.saturating_duration_since(self.last_change);
        let teardown = Duration::from_secs(u64::from(cfg.teardown_secs));
        let deep = Duration::from_secs(u64::from(cfg.deep_idle_secs));

        // Normalise the tri-state observation into a structurally
        // exhaustive two-state input. Unknown collapses to Present —
        // fail-open so a misconfigured detector never advances the
        // cooldown. The match below operates on
        // `(IdleState, Effective)` and covers all eight pairs
        // explicitly; no `unreachable!()` placeholder is needed.
        let effective = if status.is_present_or_unknown() {
            Effective::Present
        } else {
            Effective::Absent
        };

        let (new_state, edge) = match (self.state, effective) {
            // Active
            (IdleState::Active, Effective::Present) => (IdleState::Active, IdleEdge::None),
            (IdleState::Active, Effective::Absent) => {
                // Enter Cooldown but do not tear anything down yet —
                // the timer ticks while the worker keeps pulling
                // real frames. The transition resets `last_change`.
                (IdleState::Cooldown, IdleEdge::None)
            }

            // Cooldown
            (IdleState::Cooldown, Effective::Present) => {
                // Reader came back during the grace window — cancel
                // the cooldown and snap back to Active. No teardown
                // happened, so no resume side-effect is needed.
                (IdleState::Active, IdleEdge::None)
            }
            (IdleState::Cooldown, Effective::Absent) => {
                if elapsed >= teardown {
                    (IdleState::Idle, IdleEdge::EnterIdle)
                } else {
                    (IdleState::Cooldown, IdleEdge::None)
                }
            }

            // Idle / DeepIdle — resume path is identical from both
            // depths; the reload thread spawned by `ResumeActive` is
            // a no-op when the engine is already warm (Idle), and a
            // real rebuild when it was dropped (DeepIdle).
            (IdleState::Idle | IdleState::DeepIdle, Effective::Present) => {
                (IdleState::Active, IdleEdge::ResumeActive)
            }

            // Idle Absent — flip to DeepIdle once the deep timer expires.
            (IdleState::Idle, Effective::Absent) => {
                if elapsed >= deep {
                    (IdleState::DeepIdle, IdleEdge::EnterDeepIdle)
                } else {
                    (IdleState::Idle, IdleEdge::None)
                }
            }

            // DeepIdle Absent — terminal until a consumer reattaches.
            (IdleState::DeepIdle, Effective::Absent) => (IdleState::DeepIdle, IdleEdge::None),
        };

        if new_state != self.state {
            self.last_change = now;
            self.state = new_state;
        }

        let level = match new_state {
            IdleState::Active | IdleState::Cooldown => IdleLevel::Active,
            IdleState::Idle | IdleState::DeepIdle => IdleLevel::Placeholder,
        };

        // Next-tick hint. In steady-state Active/Cooldown the worker
        // re-evaluates every frame, so any non-zero value works; we
        // pick the poll interval so the worker does not over-park
        // waiting for status changes. In Idle/DeepIdle the worker
        // sleeps `1 / fps` between placeholder pushes.
        let next_tick_in = match new_state {
            IdleState::Active | IdleState::Cooldown => {
                Duration::from_millis(u64::from(cfg.poll_interval_ms))
            }
            IdleState::Idle | IdleState::DeepIdle => {
                // fps is validated > 0 by FluxConfig::validate, but
                // saturate defensively in case the runtime updates the
                // field through a Stage 13 control command before
                // validation lands.
                let fps = cfg.fps.max(1);
                Duration::from_millis(1000 / u64::from(fps))
            }
        };

        IdleTick {
            edge,
            level,
            next_tick_in,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxframe_core::IdleConfig;

    /// Build a config with explicit thresholds. The validated TOML
    /// defaults are: teardown = 5 s, deep = 30 s; tests use shorter
    /// values to keep the timestamp arithmetic readable.
    fn cfg(teardown_secs: u32, deep_idle_secs: u32) -> IdleConfig {
        IdleConfig {
            enabled: true,
            teardown_secs,
            deep_idle_secs,
            ..IdleConfig::default()
        }
    }

    fn t0() -> Instant {
        Instant::now()
    }

    // ============================================================
    // Active row (4 cases — Present, Absent, Unknown × before/after
    // are collapsed since Active has no internal timer that matters).
    // ============================================================

    #[test]
    fn active_present_stays_active() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        let tick = sm.tick(ConsumerStatus::Present, now, &cfg(5, 30));
        assert_eq!(sm.state(), IdleState::Active);
        assert_eq!(tick.edge, IdleEdge::None);
        assert_eq!(tick.level, IdleLevel::Active);
    }

    #[test]
    fn active_unknown_stays_active() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        let tick = sm.tick(ConsumerStatus::Unknown, now, &cfg(5, 30));
        assert_eq!(sm.state(), IdleState::Active);
        assert_eq!(tick.edge, IdleEdge::None);
        assert_eq!(tick.level, IdleLevel::Active);
    }

    #[test]
    fn active_absent_enters_cooldown_with_no_edge() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        let tick = sm.tick(ConsumerStatus::Absent, now, &cfg(5, 30));
        assert_eq!(sm.state(), IdleState::Cooldown);
        assert_eq!(
            tick.edge,
            IdleEdge::None,
            "cooldown entry has no side effect — input still running"
        );
        assert_eq!(tick.level, IdleLevel::Active);
    }

    // ============================================================
    // Cooldown row (6 cases — Present/Absent/Unknown × before timer
    // expires and after, except Present/Unknown which short-circuit).
    // ============================================================

    #[test]
    fn cooldown_present_cancels_to_active_no_edge() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        sm.tick(ConsumerStatus::Absent, now, &cfg(5, 30)); // → Cooldown
        let tick = sm.tick(
            ConsumerStatus::Present,
            now + Duration::from_secs(2),
            &cfg(5, 30),
        );
        assert_eq!(sm.state(), IdleState::Active);
        assert_eq!(
            tick.edge,
            IdleEdge::None,
            "no teardown happened in Cooldown, so resume has no side effect"
        );
        assert_eq!(tick.level, IdleLevel::Active);
    }

    #[test]
    fn cooldown_unknown_cancels_to_active_no_edge() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        sm.tick(ConsumerStatus::Absent, now, &cfg(5, 30));
        let tick = sm.tick(
            ConsumerStatus::Unknown,
            now + Duration::from_secs(2),
            &cfg(5, 30),
        );
        assert_eq!(sm.state(), IdleState::Active);
        assert_eq!(tick.edge, IdleEdge::None);
    }

    #[test]
    fn cooldown_absent_before_timer_stays_cooldown() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        sm.tick(ConsumerStatus::Absent, now, &cfg(5, 30));
        let tick = sm.tick(
            ConsumerStatus::Absent,
            now + Duration::from_secs(2),
            &cfg(5, 30),
        );
        assert_eq!(sm.state(), IdleState::Cooldown);
        assert_eq!(tick.edge, IdleEdge::None);
        assert_eq!(tick.level, IdleLevel::Active);
    }

    #[test]
    fn cooldown_absent_at_timer_flips_to_idle_with_edge() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        sm.tick(ConsumerStatus::Absent, now, &cfg(5, 30));
        let tick = sm.tick(
            ConsumerStatus::Absent,
            now + Duration::from_secs(5),
            &cfg(5, 30),
        );
        assert_eq!(sm.state(), IdleState::Idle);
        assert_eq!(
            tick.edge,
            IdleEdge::EnterIdle,
            "cooldown expiry must fire EnterIdle so worker tears down input"
        );
        assert_eq!(tick.level, IdleLevel::Placeholder);
    }

    #[test]
    fn cooldown_absent_past_timer_flips_to_idle() {
        // Worker may miss exact tick at the boundary; advance well
        // past the threshold.
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        sm.tick(ConsumerStatus::Absent, now, &cfg(5, 30));
        let tick = sm.tick(
            ConsumerStatus::Absent,
            now + Duration::from_secs(10),
            &cfg(5, 30),
        );
        assert_eq!(sm.state(), IdleState::Idle);
        assert_eq!(tick.edge, IdleEdge::EnterIdle);
    }

    // ============================================================
    // Idle row (6 cases — Present/Absent/Unknown × before/after the
    // deep-idle threshold).
    // ============================================================

    #[test]
    fn idle_present_resumes_active_with_edge() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        // Force into Idle.
        sm.tick(ConsumerStatus::Absent, now, &cfg(1, 10));
        sm.tick(
            ConsumerStatus::Absent,
            now + Duration::from_secs(1),
            &cfg(1, 10),
        );
        assert_eq!(sm.state(), IdleState::Idle);

        let tick = sm.tick(
            ConsumerStatus::Present,
            now + Duration::from_secs(2),
            &cfg(1, 10),
        );
        assert_eq!(sm.state(), IdleState::Active);
        assert_eq!(
            tick.edge,
            IdleEdge::ResumeActive,
            "Idle → Active must fire ResumeActive so worker spawns reload thread"
        );
        assert_eq!(tick.level, IdleLevel::Active);
    }

    #[test]
    fn idle_unknown_resumes_active() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        sm.tick(ConsumerStatus::Absent, now, &cfg(1, 10));
        sm.tick(
            ConsumerStatus::Absent,
            now + Duration::from_secs(1),
            &cfg(1, 10),
        );
        let tick = sm.tick(
            ConsumerStatus::Unknown,
            now + Duration::from_secs(2),
            &cfg(1, 10),
        );
        assert_eq!(sm.state(), IdleState::Active);
        assert_eq!(tick.edge, IdleEdge::ResumeActive);
    }

    #[test]
    fn idle_absent_before_deep_timer_stays_idle() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        sm.tick(ConsumerStatus::Absent, now, &cfg(1, 30));
        sm.tick(
            ConsumerStatus::Absent,
            now + Duration::from_secs(1),
            &cfg(1, 30),
        );
        let tick = sm.tick(
            ConsumerStatus::Absent,
            now + Duration::from_secs(15),
            &cfg(1, 30),
        );
        assert_eq!(sm.state(), IdleState::Idle);
        assert_eq!(tick.edge, IdleEdge::None);
        assert_eq!(tick.level, IdleLevel::Placeholder);
    }

    #[test]
    fn idle_absent_at_deep_timer_flips_to_deep_idle() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        sm.tick(ConsumerStatus::Absent, now, &cfg(1, 30));
        let after_idle = now + Duration::from_secs(1);
        sm.tick(ConsumerStatus::Absent, after_idle, &cfg(1, 30));
        assert_eq!(sm.state(), IdleState::Idle);

        // Deep timer is measured from Idle entry (last_change), so
        // we need 30 s past `after_idle` to reach DeepIdle.
        let tick = sm.tick(
            ConsumerStatus::Absent,
            after_idle + Duration::from_secs(30),
            &cfg(1, 30),
        );
        assert_eq!(sm.state(), IdleState::DeepIdle);
        assert_eq!(tick.edge, IdleEdge::EnterDeepIdle);
        assert_eq!(tick.level, IdleLevel::Placeholder);
    }

    // ============================================================
    // DeepIdle row (3 cases — Present/Absent/Unknown all decisive).
    // ============================================================

    #[test]
    fn deep_idle_present_resumes_active_with_edge() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        sm.tick(ConsumerStatus::Absent, now, &cfg(1, 5));
        let t1 = now + Duration::from_secs(1);
        sm.tick(ConsumerStatus::Absent, t1, &cfg(1, 5));
        let t2 = t1 + Duration::from_secs(5);
        sm.tick(ConsumerStatus::Absent, t2, &cfg(1, 5));
        assert_eq!(sm.state(), IdleState::DeepIdle);

        let tick = sm.tick(
            ConsumerStatus::Present,
            t2 + Duration::from_secs(1),
            &cfg(1, 5),
        );
        assert_eq!(sm.state(), IdleState::Active);
        assert_eq!(
            tick.edge,
            IdleEdge::ResumeActive,
            "DeepIdle → Active must fire ResumeActive (reload thread will rebuild ONNX)"
        );
    }

    #[test]
    fn deep_idle_unknown_resumes_active() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        sm.tick(ConsumerStatus::Absent, now, &cfg(1, 5));
        let t1 = now + Duration::from_secs(1);
        sm.tick(ConsumerStatus::Absent, t1, &cfg(1, 5));
        let t2 = t1 + Duration::from_secs(5);
        sm.tick(ConsumerStatus::Absent, t2, &cfg(1, 5));

        let tick = sm.tick(
            ConsumerStatus::Unknown,
            t2 + Duration::from_secs(1),
            &cfg(1, 5),
        );
        assert_eq!(sm.state(), IdleState::Active);
        assert_eq!(tick.edge, IdleEdge::ResumeActive);
    }

    #[test]
    fn deep_idle_absent_stays_deep_idle() {
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        sm.tick(ConsumerStatus::Absent, now, &cfg(1, 5));
        let t1 = now + Duration::from_secs(1);
        sm.tick(ConsumerStatus::Absent, t1, &cfg(1, 5));
        let t2 = t1 + Duration::from_secs(5);
        sm.tick(ConsumerStatus::Absent, t2, &cfg(1, 5));
        assert_eq!(sm.state(), IdleState::DeepIdle);

        let tick = sm.tick(
            ConsumerStatus::Absent,
            t2 + Duration::from_secs(100),
            &cfg(1, 5),
        );
        assert_eq!(sm.state(), IdleState::DeepIdle);
        assert_eq!(tick.edge, IdleEdge::None);
    }

    // ============================================================
    // Cross-cutting properties.
    // ============================================================

    #[test]
    fn full_lifecycle_walk() {
        // Active → Cooldown → Idle → DeepIdle → Active in one walk.
        let cfg = cfg(1, 5);
        let now = t0();
        let mut sm = IdleStateMachine::new(now);

        // Active → Cooldown (Absent).
        let t = now;
        sm.tick(ConsumerStatus::Absent, t, &cfg);
        assert_eq!(sm.state(), IdleState::Cooldown);

        // Cooldown → Idle (timer fires).
        let t = t + Duration::from_secs(1);
        let tick = sm.tick(ConsumerStatus::Absent, t, &cfg);
        assert_eq!(tick.edge, IdleEdge::EnterIdle);
        assert_eq!(sm.state(), IdleState::Idle);

        // Idle → DeepIdle (timer fires).
        let t = t + Duration::from_secs(5);
        let tick = sm.tick(ConsumerStatus::Absent, t, &cfg);
        assert_eq!(tick.edge, IdleEdge::EnterDeepIdle);
        assert_eq!(sm.state(), IdleState::DeepIdle);

        // DeepIdle → Active (Present).
        let t = t + Duration::from_secs(1);
        let tick = sm.tick(ConsumerStatus::Present, t, &cfg);
        assert_eq!(tick.edge, IdleEdge::ResumeActive);
        assert_eq!(sm.state(), IdleState::Active);
    }

    #[test]
    fn consumer_status_round_trips_through_u8() {
        for s in [
            ConsumerStatus::Present,
            ConsumerStatus::Absent,
            ConsumerStatus::Unknown,
        ] {
            assert_eq!(ConsumerStatus::from_u8(s.as_u8()), s);
        }
        // Out-of-range numeric value falls back to Unknown rather than
        // panicking — the detector might one day write a status the
        // worker does not yet know about; fail-open is safer than
        // crashing.
        assert_eq!(ConsumerStatus::from_u8(99), ConsumerStatus::Unknown);
    }

    #[test]
    fn level_tracks_state() {
        let cfg = cfg(1, 5);
        let now = t0();
        let mut sm = IdleStateMachine::new(now);

        // Active level
        let tick = sm.tick(ConsumerStatus::Present, now, &cfg);
        assert_eq!(tick.level, IdleLevel::Active);

        // Cooldown level (still Active)
        let tick = sm.tick(ConsumerStatus::Absent, now, &cfg);
        assert_eq!(tick.level, IdleLevel::Active);

        // Idle level
        let tick = sm.tick(ConsumerStatus::Absent, now + Duration::from_secs(1), &cfg);
        assert_eq!(tick.level, IdleLevel::Placeholder);
    }

    #[test]
    fn next_tick_in_uses_poll_interval_when_active() {
        let cfg = IdleConfig {
            enabled: true,
            poll_interval_ms: 250,
            ..IdleConfig::default()
        };
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        let tick = sm.tick(ConsumerStatus::Present, now, &cfg);
        assert_eq!(tick.next_tick_in, Duration::from_millis(250));
    }

    #[test]
    fn next_tick_in_uses_fps_period_when_idle() {
        let cfg = IdleConfig {
            enabled: true,
            teardown_secs: 1,
            deep_idle_secs: 10,
            fps: 5,
            ..IdleConfig::default()
        };
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        sm.tick(ConsumerStatus::Absent, now, &cfg);
        let tick = sm.tick(ConsumerStatus::Absent, now + Duration::from_secs(1), &cfg);
        assert_eq!(sm.state(), IdleState::Idle);
        assert_eq!(
            tick.next_tick_in,
            Duration::from_millis(200),
            "5 fps → 200 ms per tick"
        );
    }

    #[test]
    fn resume_active_fires_once_on_double_present() {
        // Once we land back in Active, a second consecutive Present
        // observation must not re-fire ResumeActive — there is no
        // transition, so the worker must not spawn another reload.
        let cfg = cfg(1, 5);
        let now = t0();
        let mut sm = IdleStateMachine::new(now);

        // Force into Idle.
        sm.tick(ConsumerStatus::Absent, now, &cfg);
        let after_idle = now + Duration::from_secs(1);
        sm.tick(ConsumerStatus::Absent, after_idle, &cfg);
        assert_eq!(sm.state(), IdleState::Idle);

        // First Present: edge fires.
        let t1 = after_idle + Duration::from_secs(1);
        let first = sm.tick(ConsumerStatus::Present, t1, &cfg);
        assert_eq!(sm.state(), IdleState::Active);
        assert_eq!(first.edge, IdleEdge::ResumeActive);

        // Second Present: state already Active, no edge.
        let t2 = t1 + Duration::from_millis(100);
        let second = sm.tick(ConsumerStatus::Present, t2, &cfg);
        assert_eq!(sm.state(), IdleState::Active);
        assert_eq!(
            second.edge,
            IdleEdge::None,
            "ResumeActive must not re-fire while we remain in Active"
        );
    }

    #[test]
    fn cooldown_unknown_after_timer_cancels_to_active() {
        // Unknown is fail-open ⇒ structurally equivalent to Present.
        // In Cooldown, even past the teardown threshold, an Unknown
        // observation must snap us back to Active with no edge —
        // mirroring the Present-after-timer path exactly. This pins
        // down the fail-open invariant against future regressions.
        let cfg = cfg(1, 30);
        let now = t0();
        let mut sm = IdleStateMachine::new(now);

        // Active → Cooldown.
        sm.tick(ConsumerStatus::Absent, now, &cfg);
        assert_eq!(sm.state(), IdleState::Cooldown);

        // Advance past teardown but observe Unknown rather than
        // Present or Absent.
        let tick = sm.tick(ConsumerStatus::Unknown, now + Duration::from_secs(5), &cfg);
        assert_eq!(
            sm.state(),
            IdleState::Active,
            "Unknown must be treated as Present and cancel the cooldown"
        );
        assert_eq!(
            tick.edge,
            IdleEdge::None,
            "no teardown happened in Cooldown, so no resume side effect"
        );
        assert_eq!(tick.level, IdleLevel::Active);
    }

    #[test]
    fn flapping_reader_does_not_destabilise() {
        // Absent → Present → Absent → Present rapid burst stays Active.
        let cfg = cfg(5, 30);
        let now = t0();
        let mut sm = IdleStateMachine::new(now);
        for (i, status) in [
            ConsumerStatus::Absent,
            ConsumerStatus::Present,
            ConsumerStatus::Absent,
            ConsumerStatus::Present,
            ConsumerStatus::Absent,
        ]
        .iter()
        .enumerate()
        {
            let t = now + Duration::from_millis(i as u64 * 100);
            sm.tick(*status, t, &cfg);
        }
        // Even though we ended on Absent, no tick exceeded teardown,
        // so we cannot have left Cooldown.
        assert!(
            matches!(sm.state(), IdleState::Active | IdleState::Cooldown),
            "flapping under teardown threshold must keep us out of Idle, got {:?}",
            sm.state()
        );
    }
}
