//! Embedded live preview of `/dev/video10`.
//!
//! The widget is a bare `gtk::Picture` that gets letterboxed inside
//! whatever parent slot owns it (today: the start child of a
//! `gtk::Paned` belonging to `AppModel`). Behind it we run a minimal
//! GStreamer pipeline that pulls RGBA samples through `appsink`, parks
//! them in a single-slot mailbox, and a `glib` main-loop timer drains
//! the mailbox at ~25 Hz, wrapping the bytes in a
//! `gtk::gdk::MemoryTexture` and setting it as the `Picture`'s
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
//! Failure modes (no v4l2loopback module, daemon down, device missing)
//! are handled by surfacing an empty Picture and retrying
//! `set_state(Playing)` every 5 s on a background worker (we never
//! block the GTK main thread on a state transition). We never panic;
//! preview is a best-effort secondary feature.
//!
//! glib has two in-tree versions in this binary: gstreamer brings
//! `glib` 0.20 and gtk4 brings `glib` 0.21. Callbacks must use the
//! version matching their parent crate — bus watches use bare `glib`,
//! GTK timers use `gtkglib` (the alias declared below).
//!
//! Hardcoded to `/dev/video10` for now. Reading the daemon's actual
//! `output.device` is tracked by the same TODO already noted at
//! `app.rs::open_preview` and will be wired in a follow-up once the
//! `GetConfig` IPC round-trip is exposed to the AppModel.

use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use gstreamer::prelude::*;
use gstreamer_app::AppSink;
use tracing::{debug, info, trace, warn};

// glib-0.20 (vendored by gstreamer) is reached via the bare `glib`
// path inside the bus-watch closure.
// glib-0.21 (vendored by gtk4) is needed for GTK main-loop callbacks
// and `gdk::MemoryTexture`'s `Bytes` argument — alias it so type
// mismatches surface as clear `gtkglib::*` errors instead of E0308 on
// look-alike `Bytes`.
use gtk::glib as gtkglib;

/// Negotiated preview frame, ready to be wrapped in a
/// `gtk::gdk::MemoryTexture` on the main thread. Named `PreviewSample`
/// (not `Frame`) to avoid shadowing `gtk::Frame` and
/// `gstreamer_video::VideoFrame`.
struct PreviewSample {
    bytes: Vec<u8>,
    width: i32,
    height: i32,
    stride: usize,
}

/// Preview widget + its owned GStreamer pipeline.
///
/// `root` is a bare `gtk::Picture` letterboxed inside whatever slot
/// its parent gives it (typically a `gtk::Paned` start child).
///
/// `Drop` removes the GTK timer sources and bus watch, then tears the
/// pipeline down to `Null` — order matters so the retry timer cannot
/// re-arm `Playing` on a closing widget.
pub struct Preview {
    /// Container root — embed into a parent that controls the
    /// allocation (e.g. `gtk::Paned`). Aspect-preserving via
    /// `content_fit = Contain` + `can_shrink = true`.
    pub root: gtk::Picture,
    pipeline: gstreamer::Pipeline,
    polling_source: Option<gtkglib::SourceId>,
    retry_source: Option<gtkglib::SourceId>,
    bus_watch: Option<gstreamer::bus::BusWatchGuard>,
}

impl Drop for Preview {
    fn drop(&mut self) {
        // Order: kill the timers and bus watch FIRST so the retry
        // timer cannot fire after we ask the pipeline to go Null.
        if let Some(id) = self.polling_source.take() {
            id.remove();
        }
        if let Some(id) = self.retry_source.take() {
            id.remove();
        }
        drop(self.bus_watch.take());

        if let Err(e) = self.pipeline.set_state(gstreamer::State::Null) {
            warn!(error = %e, "preview pipeline failed to stop cleanly");
        }
    }
}

/// Output dimensions baked into the pipeline's `capsfilter`. Larger
/// than the previous 320×180 because the `gtk::Paned` slot is now
/// resizable and frequently larger than the old fixed strip; 640×360
/// stays sub-millisecond for CPU `videoscale` and looks crisp on
/// HiDPI without buffering megabytes.
const PREVIEW_WIDTH: i32 = 640;
const PREVIEW_HEIGHT: i32 = 360;

/// Polling cadence for the mailbox → texture pump on the GTK main
/// thread. 40 ms targets ~25 Hz so we never poll faster than samples
/// arrive (no shared constant with `fluxframe-gst` yet; revisit if it
/// exposes one).
const POLL_INTERVAL: Duration = Duration::from_millis(40);

/// Retry interval when the pipeline fails to start (no device, no
/// v4l2loopback module loaded, daemon never started).
const RETRY_INTERVAL: Duration = Duration::from_secs(5);

/// Build the preview pane. The returned [`Preview`] owns its pipeline
/// and must be kept alive for as long as the widget should display;
/// dropping it tears the pipeline down.
///
/// # Panics
///
/// `gstreamer::init()` must already have been called (e.g. from
/// `main.rs`) — this constructor does not initialise GStreamer itself
/// and will panic deep in element factory calls otherwise.
#[must_use]
pub fn build(device_path: &Path) -> Preview {
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

    let mailbox: Arc<Mutex<Option<PreviewSample>>> = Arc::new(Mutex::new(None));
    let Some(pipeline) = build_pipeline(device_path, Arc::clone(&mailbox)) else {
        // Inert preview: empty Picture, no timers, no retry storm.
        return Preview {
            root: picture,
            pipeline: gstreamer::Pipeline::new(),
            polling_source: None,
            retry_source: None,
            bus_watch: None,
        };
    };

    // Best-effort start. On failure we fall through to the retry
    // timer below; the placeholder Picture is empty.
    match pipeline.set_state(gstreamer::State::Playing) {
        Ok(_) => info!(device = %device_path.display(), "preview pipeline started"),
        Err(e) => {
            warn!(
                device = %device_path.display(),
                error = %e,
                "preview pipeline failed to enter Playing"
            );
        }
    }

    let bus_watch = install_bus_watch(&pipeline);
    let polling_source = install_polling_timer(&picture, Arc::clone(&mailbox));
    let retry_source = install_retry_timer(&pipeline);

    Preview {
        root: picture,
        pipeline,
        polling_source: Some(polling_source),
        retry_source: Some(retry_source),
        bus_watch,
    }
}

/// Construct the consumer pipeline:
///
/// ```text
/// v4l2src ! videoconvert ! videoscale
///         ! video/x-raw,format=RGBA,width=640,height=360
///         ! appsink emit-signals=false sync=false max-buffers=1 drop=true
/// ```
///
/// Returns `None` if any factory or wiring step fails — the caller
/// must treat preview as inert and skip installing timers, otherwise
/// the retry loop will spam logs against an unwired pipeline.
fn build_pipeline(
    device_path: &Path,
    mailbox: Arc<Mutex<Option<PreviewSample>>>,
) -> Option<gstreamer::Pipeline> {
    let pipeline = gstreamer::Pipeline::new();

    let src = match gstreamer::ElementFactory::make("v4l2src")
        .property("device", device_path.display().to_string())
        .build()
    {
        Ok(e) => e,
        Err(err) => {
            warn!(error = %err, "preview: v4l2src factory missing; preview will be inert");
            return None;
        }
    };

    let convert = match gstreamer::ElementFactory::make("videoconvert").build() {
        Ok(e) => e,
        Err(err) => {
            warn!(error = %err, "preview: videoconvert factory missing; preview will be inert");
            return None;
        }
    };
    let scale = match gstreamer::ElementFactory::make("videoscale").build() {
        Ok(e) => e,
        Err(err) => {
            warn!(error = %err, "preview: videoscale factory missing; preview will be inert");
            return None;
        }
    };

    let caps = gstreamer::Caps::builder("video/x-raw")
        .field("format", "RGBA")
        .field("width", PREVIEW_WIDTH)
        .field("height", PREVIEW_HEIGHT)
        .build();
    let filter = match gstreamer::ElementFactory::make("capsfilter")
        .property("caps", &caps)
        .build()
    {
        Ok(e) => e,
        Err(err) => {
            warn!(error = %err, "preview: capsfilter factory missing; preview will be inert");
            return None;
        }
    };

    // factory-then-downcast keeps `add_many` happy with `&Element`.
    let appsink_elem = match gstreamer::ElementFactory::make("appsink")
        .name("preview-sink")
        .property("sync", false)
        .property("max-buffers", 1u32)
        .property("drop", true)
        .property("emit-signals", false)
        .build()
    {
        Ok(e) => e,
        Err(err) => {
            warn!(error = %err, "preview: appsink factory missing; preview will be inert");
            return None;
        }
    };

    let elements = [&src, &convert, &scale, &filter, &appsink_elem];
    if let Err(e) = pipeline.add_many(elements) {
        warn!(error = %e, "preview pipeline add_many failed");
        return None;
    }
    if let Err(e) = gstreamer::Element::link_many(elements) {
        warn!(error = %e, "preview pipeline link_many failed");
        return None;
    }

    let Ok(appsink): Result<AppSink, _> = appsink_elem.downcast() else {
        warn!("preview: appsink element failed to downcast to AppSink; preview will be inert");
        return None;
    };
    attach_appsink_callback(&appsink, mailbox);
    Some(pipeline)
}

/// Pull buffer/caps/info/map out of a `Sample` and copy the bytes into
/// an owned [`PreviewSample`]. Returns `None` on any negotiation
/// failure or numeric overflow — caller logs at trace level via the
/// closure that wraps this.
///
/// Cost note: `map.as_slice().to_vec()` copies ~922 KB per sample at
/// 640×360 RGBA. At 25 fps that's ~23 MB/s of allocator traffic.
/// Long-term we'd switch to `gtk4paintablesink` (gst-plugins-rs,
/// currently not packaged in Guix) and bypass the copy entirely.
fn extract_sample(sample: &gstreamer::Sample) -> Option<PreviewSample> {
    let Some(buffer) = sample.buffer() else {
        trace!(reason = "no buffer", "preview: sample dropped");
        return None;
    };
    let Some(caps) = sample.caps() else {
        trace!(reason = "no caps", "preview: sample dropped");
        return None;
    };
    let Ok(info) = gstreamer_video::VideoInfo::from_caps(caps) else {
        trace!(
            reason = "VideoInfo::from_caps failed",
            "preview: sample dropped"
        );
        return None;
    };
    let Ok(map) = buffer.map_readable() else {
        trace!(reason = "map_readable failed", "preview: sample dropped");
        return None;
    };

    let Ok(width) = i32::try_from(info.width()) else {
        trace!("preview: width overflows i32; dropping sample");
        return None;
    };
    let Ok(height) = i32::try_from(info.height()) else {
        trace!("preview: height overflows i32; dropping sample");
        return None;
    };
    let raw_stride = info.stride()[0];
    let Ok(stride) = usize::try_from(raw_stride) else {
        trace!(
            stride = raw_stride,
            "preview: negative stride; dropping sample"
        );
        return None;
    };

    Some(PreviewSample {
        bytes: map.as_slice().to_vec(),
        width,
        height,
        stride,
    })
}

fn attach_appsink_callback(appsink: &AppSink, mailbox: Arc<Mutex<Option<PreviewSample>>>) {
    appsink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                let sample = match sink.pull_sample() {
                    Ok(s) => s,
                    Err(e) => {
                        trace!(error = %e, "preview appsink pull_sample failed");
                        return Err(gstreamer::FlowError::Error);
                    }
                };
                let Some(sample) = extract_sample(&sample) else {
                    return Ok(gstreamer::FlowSuccess::Ok);
                };
                // Latest-wins: replace whatever was there. We never
                // queue, so an idle main thread won't accumulate a
                // backlog of allocations. Poisoning would only happen
                // if a future caller below adds a panic; revisit this
                // branch then.
                if let Ok(mut slot) = mailbox.lock() {
                    *slot = Some(sample);
                }
                Ok(gstreamer::FlowSuccess::Ok)
            })
            .build(),
    );
}

/// Pump the mailbox → texture transfer on the GTK main thread. The
/// returned `SourceId` is owned by [`Preview`] and removed in `Drop`.
fn install_polling_timer(
    picture: &gtk::Picture,
    mailbox: Arc<Mutex<Option<PreviewSample>>>,
) -> gtkglib::SourceId {
    let picture = picture.clone();
    gtkglib::source::timeout_add_local(POLL_INTERVAL, move || {
        let sample = mailbox.lock().ok().and_then(|mut slot| slot.take());
        if let Some(sample) = sample {
            let bytes = gtkglib::Bytes::from_owned(sample.bytes);
            let texture = gtk::gdk::MemoryTexture::new(
                sample.width,
                sample.height,
                gtk::gdk::MemoryFormat::R8g8b8a8,
                &bytes,
                sample.stride,
            );
            picture.set_paintable(Some(&texture));
        }
        gtkglib::ControlFlow::Continue
    })
}

/// Bus watcher logging error/EOS. The retry timer below independently
/// nudges the pipeline back to `Playing`, so we don't have to
/// coordinate state changes from the bus callback. The returned
/// [`BusWatchGuard`] keeps the watch alive; dropping it removes it.
fn install_bus_watch(pipeline: &gstreamer::Pipeline) -> Option<gstreamer::bus::BusWatchGuard> {
    let bus = pipeline.bus()?;
    match bus.add_watch_local(|_, msg| {
        match msg.view() {
            gstreamer::MessageView::Error(err) => {
                warn!(
                    error = %err.error(),
                    debug = ?err.debug(),
                    "preview pipeline bus error (retry timer will recover)"
                );
            }
            gstreamer::MessageView::Eos(_) => {
                info!("preview pipeline EOS (retry timer will recover)");
            }
            _ => {}
        }
        glib::ControlFlow::Continue
    }) {
        Ok(guard) => Some(guard),
        Err(e) => {
            warn!(error = %e, "preview: bus watch install failed");
            None
        }
    }
}

/// Periodically nudge the pipeline back into `Playing` so transient
/// startup failures (daemon launched after the GUI, v4l2loopback
/// reloaded mid-run) recover without user action.
///
/// The GTK-side timer only inspects state and *decides* whether to
/// spawn a worker; the actual `set_state(Null) → Playing` happens on a
/// detached `std::thread` so the GTK main thread is never blocked on
/// a state transition. The returned `SourceId` is owned by [`Preview`]
/// and removed in `Drop`.
fn install_retry_timer(pipeline: &gstreamer::Pipeline) -> gtkglib::SourceId {
    let pipeline = pipeline.clone();
    // `attempts == 0` means "last tick was healthy or we never failed
    // yet"; on transition to broken we log warn. Subsequent broken
    // ticks log only on power-of-two attempts (1, 2, 4, 8, …) so a
    // long outage doesn't spam logs. Reset to 0 on healthy.
    let attempts: Rc<Cell<u32>> = Rc::new(Cell::new(0));
    gtkglib::source::timeout_add_local(RETRY_INTERVAL, move || {
        let current = pipeline.current_state();
        let pending = pipeline.pending_state();

        if current == gstreamer::State::Playing {
            let prev_attempts = attempts.replace(0);
            if prev_attempts > 0 {
                info!(state = ?current, "preview pipeline recovered to Playing");
            }
            return gtkglib::ControlFlow::Continue;
        }
        if pending == gstreamer::State::Playing {
            // Already mid-transition; don't pile on a second
            // set_state from the GTK thread.
            debug!(
                current = ?current,
                pending = ?pending,
                "preview pipeline transitioning toward Playing — skip retry"
            );
            return gtkglib::ControlFlow::Continue;
        }

        let n = attempts.get();
        let should_log = n == 0 || (n > 0 && n.is_power_of_two());
        if should_log {
            warn!(
                state = ?current,
                attempts = n,
                "preview pipeline below Playing — retrying on worker"
            );
        } else {
            debug!(state = ?current, attempts = n, "preview pipeline retry tick");
        }
        attempts.set(n.saturating_add(1));

        // Run the (potentially blocking) state transitions off the GTK
        // main loop. Capture an owned clone so the closure stays
        // `'static`.
        let pipeline_for_worker = pipeline.clone();
        std::thread::spawn(move || {
            if let Err(e) = pipeline_for_worker.set_state(gstreamer::State::Null) {
                warn!(error = %e, "preview pipeline reset to Null failed");
            }
            if let Err(e) = pipeline_for_worker.set_state(gstreamer::State::Playing) {
                warn!(error = %e, "preview pipeline retry to Playing failed");
            }
        });

        gtkglib::ControlFlow::Continue
    })
}
