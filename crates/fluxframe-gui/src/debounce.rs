//! Per-parameter debounce on the UI thread.
//!
//! Slider drags fire `value-changed` faster than the daemon's
//! 16-deep bounded channel can absorb. We coalesce updates per
//! parameter path via `glib::timeout_add_local_once`: the latest
//! pending value wins, and previous timeouts are cancelled when a
//! new value arrives within the window.
//!
//! Lives on the UI thread (single-threaded `HashMap<String,
//! SourceId>`); the debounce logic is invoked from gtk signal
//! handlers and the relm4 update loop, both of which run there.
//!
//! ## Thread affinity
//!
//! [`Debouncer`] holds an `Rc<RefCell<...>>` and is therefore
//! `!Send`. It must live entirely on the gtk main thread; the type
//! system enforces this — any attempt to capture a `Debouncer` in
//! a `Send` closure (e.g. moving one into an IPC worker payload)
//! will fail to compile. The compile error can be cryptic, so
//! consumers should treat this as a designed invariant rather than
//! a coincidence.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Duration;

use gtk::glib;

/// Reusable per-path debounce store.
///
/// Stores at most one scheduled timeout per parameter path. A new
/// `schedule` call for the same path cancels the previous timeout
/// and replaces it; the registered closure fires after the configured
/// `Duration` of inactivity.
#[derive(Default, Clone)]
pub(crate) struct Debouncer {
    pending: Rc<RefCell<HashMap<String, glib::SourceId>>>,
}

impl Debouncer {
    /// Construct an empty debouncer.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Schedule `f` to run after `delay`, cancelling any previously-
    /// scheduled callback for the same `path`. `f` runs on the
    /// gtk main loop.
    ///
    /// # Panics
    ///
    /// In glib 0.21 (the version pulled by gtk4 0.10) `SourceId::remove`
    /// panics if the source has already been removed by some other code
    /// path. This is impossible under the current invariant — the only
    /// producer of entries is `schedule`, and the timeout callback drops
    /// its entry from `pending` before invoking `f()`. Future
    /// refactorings must preserve that invariant.
    pub(crate) fn schedule(&self, path: String, delay: Duration, f: impl FnOnce() + 'static) {
        // Cancel any pending timeout for this path.
        // Bind first so the `RefCell` borrow is released before the
        // source (and its captured closure) is dropped.
        let prev = self.pending.borrow_mut().remove(&path);
        if let Some(prev) = prev {
            prev.remove();
        }
        // Schedule new callback. On fire, drop the entry from the
        // map so a subsequent schedule for the same path is not a
        // no-op.
        let pending = Rc::clone(&self.pending);
        let path_for_remove = path.clone();
        let id = glib::timeout_add_local_once(delay, move || {
            pending.borrow_mut().remove(&path_for_remove);
            f();
        });
        self.pending.borrow_mut().insert(path, id);
    }

    /// Send `f` immediately, bypassing the debounce queue. Used for
    /// `CommitStrategy::Instant` and `CommitStrategy::OnCommit`
    /// parameters.
    ///
    /// # Panics
    ///
    /// Same invariant as [`Self::schedule`].
    pub(crate) fn flush(&self, path: &str, f: impl FnOnce()) {
        let prev = self.pending.borrow_mut().remove(path);
        if let Some(prev) = prev {
            prev.remove();
        }
        f();
    }

    /// Cancel every pending callback without running it.
    ///
    /// Call before the context the callbacks were scheduled for goes
    /// away (preset switch, revert, chain edit, chain page rebuild) so
    /// a stale value is never sent into the new context.
    ///
    /// # Panics
    ///
    /// Same invariant as [`Self::schedule`]: every stored `SourceId` is
    /// still attached, because a fired callback removes its own entry
    /// before running.
    pub(crate) fn cancel_all(&self) {
        // Collect first so the `RefCell` borrow is released before any
        // source is removed.
        let drained: Vec<glib::SourceId> = self
            .pending
            .borrow_mut()
            .drain()
            .map(|(_, id)| id)
            .collect();
        for id in drained {
            id.remove();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::time::Instant;

    use super::*;

    /// Schedule one counting callback per path on `debouncer`.
    fn schedule_counters(debouncer: &Debouncer, fired: &Rc<Cell<u32>>) {
        for path in ["bg.blur.radius", "fg.sharpen.amount"] {
            let fired = Rc::clone(fired);
            debouncer.schedule(path.to_string(), Duration::from_millis(10), move || {
                fired.set(fired.get() + 1);
            });
        }
    }

    /// Iterate the default main context well past the debounce delay.
    fn run_default_context_past_delay() {
        let ctx = glib::MainContext::default();
        let deadline = Instant::now() + Duration::from_millis(60);
        while Instant::now() < deadline {
            ctx.iteration(false);
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// Regression: pending debounced sends must not survive
    /// `cancel_all` (e.g. a slider drag followed by a preset switch).
    ///
    /// `timeout_add_local_once` always attaches to the global default
    /// context, so both phases live in one test to keep parallel test
    /// threads from contending for it. The first phase is the control:
    /// it proves the loop really runs the callbacks.
    #[test]
    fn cancel_all_drops_every_pending_callback() {
        let fired = Rc::new(Cell::new(0_u32));

        let debouncer = Debouncer::new();
        schedule_counters(&debouncer, &fired);
        run_default_context_past_delay();
        assert_eq!(fired.get(), 2, "control: callbacks fire without cancel_all");
        assert!(debouncer.pending.borrow().is_empty());

        fired.set(0);
        schedule_counters(&debouncer, &fired);
        debouncer.cancel_all();
        assert!(debouncer.pending.borrow().is_empty());
        run_default_context_past_delay();
        assert_eq!(fired.get(), 0, "cancel_all must drop pending callbacks");
    }
}
