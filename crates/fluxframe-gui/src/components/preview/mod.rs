//! Embedded live preview of `/dev/video10`.
//!
//! The widget is a bare `gtk::Picture` that gets letterboxed inside
//! whatever parent slot owns it (today: the start child of a
//! `gtk::Paned` belonging to `AppModel`). Behind it we run a minimal
//! GStreamer pipeline (see [`pipeline`]) that pulls RGBA samples
//! through `appsink`, parks them in a single-slot mailbox, and a
//! main-loop timer drains the mailbox at ~25 Hz, wrapping the bytes in
//! a `gtk::gdk::MemoryTexture` and setting it as the `Picture`'s
//! paintable. Sizing reference:
//! <https://discourse.gnome.org/t/gtkpicture-fixed-aspect-ratio-and-size/17127>
//!
//! Choice of polling-mailbox over per-sample dispatch: `gtk::gdk::Texture`
//! and `gtk::Picture` are GObjects that are not `Send`, so a
//! per-sample `MainContext::invoke` would need a `Send` closure
//! holding raw bytes anyway. A shared `Mutex<Option<PreviewSample>>`
//! keeps the cross-thread surface to one type and matches the
//! latest-wins semantics of the upstream `LatestFrameSlot` in
//! `fluxframe-gst`.
//!
//! Lifecycle: the preview is *active* (pipeline driven to `Playing`,
//! polling + retry timers and bus watch installed) only while `root`
//! is mapped and its window is not suspended (minimised / hidden).
//! Otherwise the timers are removed, the pipeline goes to `Null` and
//! the placeholder is shown — so the preview does not hold
//! `/dev/video10` open as a reader and the daemon can drop to idle.
//!
//! Failure modes (no v4l2loopback module, daemon down, device missing)
//! are handled by showing an `adw::StatusPage` placeholder and
//! retrying `set_state(Null → Playing)` every 5 s. All state
//! transitions while the GUI runs happen on a background worker (we
//! never block the GTK main thread on one); only `Drop` stops the
//! pipeline synchronously. A bus `Error` / `Eos` raises a
//! `needs_restart` flag, because after a streaming error the pipeline
//! usually still reports `Playing` and the state check alone would
//! never restart it. We never panic; preview is a best-effort
//! secondary feature. If GStreamer failed to initialise, the preview
//! is an inert placeholder that makes no GStreamer calls at all.
//!
//! glib has two in-tree versions in this binary: gstreamer brings
//! `glib` 0.20 and gtk4 brings `glib` 0.21. Callbacks must use the
//! version matching their parent crate — bus watches use
//! `gstreamer::glib`, GTK timers and signals use `gtk::glib`.
//!
//! Hardcoded to `/dev/video10` for now (see `DEFAULT_PREVIEW_DEVICE`
//! in app.rs).

mod pipeline;

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::rc::{Rc, Weak};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use gstreamer::prelude::{ElementExt, ElementExtManual};
use gtk::glib;
use gtk::prelude::*;
use tracing::{debug, info, warn};

use pipeline::{Mailbox, RestartFlags};

/// Visible-child names of the root `gtk::Stack`.
const PLACEHOLDER_CHILD: &str = "placeholder";
const PICTURE_CHILD: &str = "picture";

/// Polling cadence for the mailbox → texture pump on the GTK main
/// thread. 40 ms targets ~25 Hz so we never poll faster than samples
/// arrive (no shared constant with `fluxframe-gst` yet; revisit if it
/// exposes one).
const POLL_INTERVAL: Duration = Duration::from_millis(40);

/// Retry interval when the pipeline fails to start (no device, no
/// v4l2loopback module loaded, daemon never started).
const RETRY_INTERVAL: Duration = Duration::from_secs(5);

/// Preview widget + its owned GStreamer pipeline.
///
/// `root` is a `gtk::Stack` switching between an `adw::StatusPage`
/// placeholder (no signal) and a `gtk::Picture` letterboxed inside
/// whatever slot its parent gives it (typically a `gtk::Paned` start
/// child).
///
/// `Drop` disconnects the lifecycle signals, removes the GTK timer
/// sources and bus watch, then tears the pipeline down to `Null`
/// synchronously — order matters so the retry timer cannot re-arm
/// `Playing` on a closing widget.
pub struct Preview {
    /// Container root — embed into a parent that controls the
    /// allocation (e.g. `gtk::Paned`). The inner Picture is
    /// aspect-preserving via `content_fit = Contain` +
    /// `can_shrink = true`.
    pub root: gtk::Stack,
    /// `None` for an inert preview (GStreamer not initialised or
    /// elements missing): no pipeline, no timers, no signal handlers.
    live: Option<Rc<Live>>,
    /// `map` / `unmap` handlers on `root`; empty when inert.
    root_handlers: Vec<glib::SignalHandlerId>,
}

/// State of a non-inert preview. Signal closures hold it only weakly,
/// so the sole strong reference is [`Preview::live`].
struct Live {
    stack: gtk::Stack,
    picture: gtk::Picture,
    pipeline: gstreamer::Pipeline,
    flags: Arc<RestartFlags>,
    mailbox: Mailbox,
    /// Sources installed while active; `None` while inactive.
    running: RefCell<Option<Running>>,
    /// Window whose `suspended` property we follow, with the handler
    /// id so a re-map (or reparent) does not subscribe twice.
    window: RefCell<Option<(glib::WeakRef<gtk::Window>, glib::SignalHandlerId)>>,
}

/// Per-activation resources, removed on deactivation and in `Drop`.
struct Running {
    polling_source: glib::SourceId,
    retry_source: glib::SourceId,
    bus_watch: Option<gstreamer::bus::BusWatchGuard>,
}

impl Running {
    fn remove(self) {
        self.polling_source.remove();
        self.retry_source.remove();
        drop(self.bus_watch);
    }
}

impl Drop for Preview {
    fn drop(&mut self) {
        let Some(live) = self.live.take() else {
            return;
        };
        for id in self.root_handlers.drain(..) {
            self.root.disconnect(id);
        }
        if let Some((window, id)) = live.window.borrow_mut().take() {
            if let Some(window) = window.upgrade() {
                window.disconnect(id);
            }
        }
        // Order: kill the timers and bus watch FIRST so the retry
        // timer cannot fire after we ask the pipeline to go Null.
        if let Some(running) = live.running.borrow_mut().take() {
            running.remove();
        }

        // Tell an in-flight restart worker not to re-arm Playing; it
        // rolls back to Null if it already did.
        live.flags.active.store(false, Ordering::SeqCst);
        live.flags.shutting_down.store(true, Ordering::SeqCst);
        if let Err(e) = live.pipeline.set_state(gstreamer::State::Null) {
            warn!(error = %e, "preview pipeline failed to stop cleanly");
        }
    }
}

/// Build the preview pane. The returned [`Preview`] owns its pipeline
/// and must be kept alive for as long as the widget should display;
/// dropping it tears the pipeline down.
///
/// `gst_ready` reports whether `gstreamer::init()` succeeded; when it
/// is `false` the preview is an inert placeholder and no GStreamer API
/// is called. The pipeline itself is only started once `root` is
/// mapped.
#[must_use]
pub fn build(device_path: &Path, gst_ready: bool) -> Preview {
    // The Picture does NOT constrain its own size — the parent
    // (`gtk::Paned`) owns the allocation. `content_fit = Contain`
    // letterboxes the texture inside whatever the Paned hands us;
    // `can_shrink = true` allows shrinking below the natural texture
    // size; `vexpand + hexpand = true` lets it claim the whole slot.
    let picture = gtk::Picture::builder()
        .content_fit(gtk::ContentFit::Contain)
        .can_shrink(true)
        .vexpand(true)
        .hexpand(true)
        .build();
    picture.set_alternative_text(Some("Live output preview"));
    let placeholder = adw::StatusPage::builder()
        .icon_name("camera-disabled-symbolic")
        .title("No Preview Signal")
        .description(format!(
            "Waiting for frames from {}.",
            device_path.display()
        ))
        .vexpand(true)
        .hexpand(true)
        .build();
    let stack = gtk::Stack::builder().vexpand(true).hexpand(true).build();
    stack.add_named(&placeholder, Some(PLACEHOLDER_CHILD));
    stack.add_named(&picture, Some(PICTURE_CHILD));
    stack.set_visible_child_name(PLACEHOLDER_CHILD);

    let inert = |description: &str| {
        // Inert preview: placeholder only, no timers, no retry storm.
        placeholder.set_description(Some(description));
        Preview {
            root: stack.clone(),
            live: None,
            root_handlers: Vec::new(),
        }
    };
    if !gst_ready {
        return inert("Preview is unavailable: GStreamer failed to initialise.");
    }

    let mailbox: Mailbox = Mailbox::default();
    let Some(pipeline) = pipeline::build_pipeline(device_path, Arc::clone(&mailbox)) else {
        return inert("Preview is unavailable: required GStreamer elements are missing.");
    };
    info!(device = %device_path.display(), "preview pipeline built");

    let live = Rc::new(Live {
        stack: stack.clone(),
        picture,
        pipeline,
        flags: Arc::new(RestartFlags::default()),
        mailbox,
        running: RefCell::new(None),
        window: RefCell::new(None),
    });

    let weak = Rc::downgrade(&live);
    let map_handler = stack.connect_map(move |_| {
        if let Some(live) = weak.upgrade() {
            Live::follow_window(&live);
            let suspended = live.window_suspended();
            live.set_active(should_be_active(true, suspended));
        }
    });
    let weak = Rc::downgrade(&live);
    let unmap_handler = stack.connect_unmap(move |_| {
        if let Some(live) = weak.upgrade() {
            live.set_active(should_be_active(false, live.window_suspended()));
        }
    });

    Preview {
        root: stack,
        live: Some(live),
        root_handlers: vec![map_handler, unmap_handler],
    }
}

/// Pure activity decision: stream only while visible on screen.
fn should_be_active(mapped: bool, window_suspended: bool) -> bool {
    mapped && !window_suspended
}

impl Live {
    /// Subscribe to `suspended` on the toplevel window `stack` is
    /// currently in, replacing a subscription on a previous window.
    /// The closure holds `Live` weakly to avoid an `Rc` cycle through
    /// the window's signal table.
    fn follow_window(this: &Rc<Self>) {
        let Some(window) = this
            .stack
            .root()
            .and_then(|root| root.downcast::<gtk::Window>().ok())
        else {
            return;
        };
        let mut slot = this.window.borrow_mut();
        if let Some((current, _)) = slot.as_ref() {
            if current.upgrade().as_ref() == Some(&window) {
                return;
            }
        }
        if let Some((old, id)) = slot.take() {
            if let Some(old) = old.upgrade() {
                old.disconnect(id);
            }
        }
        let weak: Weak<Self> = Rc::downgrade(this);
        let id = window.connect_suspended_notify(move |window| {
            if let Some(live) = weak.upgrade() {
                live.set_active(should_be_active(
                    live.stack.is_mapped(),
                    window.is_suspended(),
                ));
            }
        });
        *slot = Some((window.downgrade(), id));
    }

    fn window_suspended(&self) -> bool {
        self.window
            .borrow()
            .as_ref()
            .and_then(|(window, _)| window.upgrade())
            .is_some_and(|window| window.is_suspended())
    }

    /// Transition between active and inactive; idempotent.
    fn set_active(&self, active: bool) {
        let mut running = self.running.borrow_mut();
        match (active, running.is_some()) {
            (true, false) => {
                debug!("preview activated");
                self.flags.needs_restart.store(false, Ordering::SeqCst);
                self.flags.active.store(true, Ordering::SeqCst);
                // If a worker is still in flight it picks up the new
                // desired state itself; see `spawn_restart_worker`.
                pipeline::spawn_restart_worker(&self.pipeline, &self.flags);
                *running = Some(Running {
                    polling_source: install_polling_timer(
                        &self.stack,
                        &self.picture,
                        Arc::clone(&self.mailbox),
                        Arc::clone(&self.flags),
                    ),
                    retry_source: install_retry_timer(&self.pipeline, Arc::clone(&self.flags)),
                    bus_watch: pipeline::install_bus_watch(&self.pipeline, Arc::clone(&self.flags)),
                });
            }
            (false, true) => {
                debug!("preview deactivated");
                if let Some(r) = running.take() {
                    r.remove();
                }
                self.flags.active.store(false, Ordering::SeqCst);
                pipeline::spawn_restart_worker(&self.pipeline, &self.flags);
                // Drop the stale frame so re-activation does not flash
                // it before the first fresh sample.
                if let Ok(mut slot) = self.mailbox.lock() {
                    slot.take();
                }
                self.stack.set_visible_child_name(PLACEHOLDER_CHILD);
            }
            _ => {}
        }
    }
}

/// Pump the mailbox → texture transfer on the GTK main thread and
/// switch the stack between placeholder and picture: a fresh sample
/// shows the picture, a pending `needs_restart` (bus Error/EOS) shows
/// the placeholder. The returned `SourceId` is owned by [`Running`].
fn install_polling_timer(
    stack: &gtk::Stack,
    picture: &gtk::Picture,
    mailbox: Mailbox,
    flags: Arc<RestartFlags>,
) -> glib::SourceId {
    let stack = stack.clone();
    let picture = picture.clone();
    glib::source::timeout_add_local(POLL_INTERVAL, move || {
        if flags.needs_restart.load(Ordering::SeqCst) {
            // Drop any stale frame queued before the error so it does
            // not flip the picture back in.
            if let Ok(mut slot) = mailbox.lock() {
                slot.take();
            }
            if stack.visible_child_name().as_deref() != Some(PLACEHOLDER_CHILD) {
                stack.set_visible_child_name(PLACEHOLDER_CHILD);
            }
            return glib::ControlFlow::Continue;
        }
        let sample = mailbox.lock().ok().and_then(|mut slot| slot.take());
        if let Some(sample) = sample {
            let bytes = glib::Bytes::from_owned(sample.bytes);
            let texture = gtk::gdk::MemoryTexture::new(
                sample.width,
                sample.height,
                gtk::gdk::MemoryFormat::R8g8b8a8,
                &bytes,
                sample.stride,
            );
            picture.set_paintable(Some(&texture));
            if stack.visible_child_name().as_deref() != Some(PICTURE_CHILD) {
                stack.set_visible_child_name(PICTURE_CHILD);
            }
        }
        glib::ControlFlow::Continue
    })
}

/// Periodically nudge the pipeline back into `Playing` so transient
/// failures (daemon launched after the GUI, v4l2loopback reloaded
/// mid-run, streaming error / EOS reported on the bus) recover without
/// user action.
///
/// The GTK-side timer only inspects state and *decides* whether to
/// spawn a worker; the actual transition happens in
/// [`pipeline::spawn_restart_worker`]. The returned `SourceId` is
/// owned by [`Running`].
fn install_retry_timer(pipeline: &gstreamer::Pipeline, flags: Arc<RestartFlags>) -> glib::SourceId {
    let pipeline = pipeline.clone();
    // `attempts == 0` means "last tick was healthy or we never failed
    // yet"; on transition to broken we log warn. Subsequent broken
    // ticks log only on power-of-two attempts (1, 2, 4, 8, …) so a
    // long outage doesn't spam logs. Reset to 0 on healthy.
    let attempts: Rc<Cell<u32>> = Rc::new(Cell::new(0));
    glib::source::timeout_add_local(RETRY_INTERVAL, move || {
        let current = pipeline.current_state();
        let pending = pipeline.pending_state();
        let needs_restart = flags.needs_restart.load(Ordering::SeqCst);
        let in_flight = flags.restart_in_flight.load(Ordering::SeqCst);

        if !pipeline::should_restart(current, pending, needs_restart, in_flight) {
            if current == gstreamer::State::Playing && !in_flight {
                let prev_attempts = attempts.replace(0);
                if prev_attempts > 0 {
                    info!(state = ?current, "preview pipeline recovered to Playing");
                }
            } else {
                // Mid-transition or a worker is still running; don't
                // pile on a second set_state.
                debug!(
                    current = ?current,
                    pending = ?pending,
                    in_flight,
                    "preview pipeline restart pending — skip retry"
                );
            }
            return glib::ControlFlow::Continue;
        }

        let n = attempts.get();
        let should_log = n == 0 || (n > 0 && n.is_power_of_two());
        if should_log {
            warn!(
                state = ?current,
                needs_restart,
                attempts = n,
                "preview pipeline not streaming — restarting on worker"
            );
        } else {
            debug!(state = ?current, attempts = n, "preview pipeline retry tick");
        }
        attempts.set(n.saturating_add(1));

        // Clear only once a worker actually took the job, so a racing
        // in-flight worker does not swallow the restart request.
        if pipeline::spawn_restart_worker(&pipeline, &flags) {
            flags.needs_restart.store(false, Ordering::SeqCst);
        }

        glib::ControlFlow::Continue
    })
}

#[cfg(test)]
mod tests {
    use super::should_be_active;

    #[test]
    fn active_only_when_mapped_and_not_suspended() {
        assert!(should_be_active(true, false));
        assert!(!should_be_active(true, true));
        assert!(!should_be_active(false, false));
        assert!(!should_be_active(false, true));
    }
}
