//! GStreamer side of the embedded preview: pipeline construction,
//! appsink → mailbox transfer, bus watch and the state-transition
//! worker. Nothing in here touches GTK; the widget half lives in
//! [`super`].
//!
//! Every `set_state` call made while the GUI runs goes through
//! [`spawn_restart_worker`], so the GTK main thread never blocks on a
//! state transition. The only synchronous transition is the final
//! `Null` in `Preview::drop`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use gstreamer::prelude::*;
use gstreamer_app::AppSink;
use tracing::{debug, info, trace, warn};

/// Negotiated preview frame, ready to be wrapped in a
/// `gtk::gdk::MemoryTexture` on the main thread. Named `PreviewSample`
/// (not `Frame`) to avoid shadowing `gtk::Frame` and
/// `gstreamer_video::VideoFrame`.
pub(super) struct PreviewSample {
    pub(super) bytes: Vec<u8>,
    pub(super) width: i32,
    pub(super) height: i32,
    pub(super) stride: usize,
}

/// Single-slot, latest-wins frame mailbox shared between the appsink
/// streaming thread and the GTK polling timer.
pub(super) type Mailbox = Arc<Mutex<Option<PreviewSample>>>;

/// Flags shared between the bus watch, the GTK timers, the restart
/// worker thread and `Drop`.
#[derive(Default)]
pub(super) struct RestartFlags {
    /// Set by the bus watch on `Error` / `Eos`; cleared by the retry
    /// timer when it spawns a restart worker.
    pub(super) needs_restart: AtomicBool,
    /// A restart worker thread is running; no second one is spawned.
    pub(super) restart_in_flight: AtomicBool,
    /// Set by `Drop` before the pipeline goes to `Null`, so a worker
    /// already running does not re-arm `Playing` afterwards.
    pub(super) shutting_down: AtomicBool,
    /// Desired state: `true` while the widget is mapped and its window
    /// is not suspended. The worker re-reads it after each transition
    /// and rolls back if it changed meanwhile.
    pub(super) active: AtomicBool,
}

/// Output dimensions baked into the pipeline's `capsfilter`. Larger
/// than the previous 320×180 because the `gtk::Paned` slot is now
/// resizable and frequently larger than the old fixed strip; 640×360
/// stays sub-millisecond for CPU `videoscale` and looks crisp on
/// HiDPI without buffering megabytes.
const PREVIEW_WIDTH: i32 = 640;
const PREVIEW_HEIGHT: i32 = 360;

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
pub(super) fn build_pipeline(device_path: &Path, mailbox: Mailbox) -> Option<gstreamer::Pipeline> {
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

fn attach_appsink_callback(appsink: &AppSink, mailbox: Mailbox) {
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

/// Bus watcher for error/EOS. It only raises `needs_restart`: after a
/// streaming error or EOS the pipeline typically still reports
/// `Playing`, so the retry timer needs this flag to know a restart is
/// due. State changes themselves stay on the restart worker. The
/// returned [`gstreamer::bus::BusWatchGuard`] keeps the watch alive;
/// dropping it removes it.
pub(super) fn install_bus_watch(
    pipeline: &gstreamer::Pipeline,
    flags: Arc<RestartFlags>,
) -> Option<gstreamer::bus::BusWatchGuard> {
    let bus = pipeline.bus()?;
    match bus.add_watch_local(move |_, msg| {
        match msg.view() {
            gstreamer::MessageView::Error(err) => {
                warn!(
                    error = %err.error(),
                    debug = ?err.debug(),
                    "preview pipeline bus error (restart scheduled)"
                );
                flags.needs_restart.store(true, Ordering::SeqCst);
            }
            gstreamer::MessageView::Eos(_) => {
                info!("preview pipeline EOS (restart scheduled)");
                flags.needs_restart.store(true, Ordering::SeqCst);
            }
            _ => {}
        }
        gstreamer::glib::ControlFlow::Continue
    }) {
        Ok(guard) => Some(guard),
        Err(e) => {
            warn!(error = %e, "preview: bus watch install failed");
            None
        }
    }
}

/// Pure retry decision for the retry timer.
///
/// - a worker already in flight → never spawn a second one;
/// - `needs_restart` (bus Error/EOS) → restart even if the pipeline
///   still reports `Playing`;
/// - otherwise restart only when the pipeline is below `Playing` and
///   not already transitioning toward it.
pub(super) fn should_restart(
    current: gstreamer::State,
    pending: gstreamer::State,
    needs_restart: bool,
    in_flight: bool,
) -> bool {
    if in_flight {
        return false;
    }
    if needs_restart {
        return true;
    }
    current != gstreamer::State::Playing && pending != gstreamer::State::Playing
}

/// Pure target-state decision for the restart worker: `Playing` only
/// while the widget wants to be active and `Drop` has not started.
fn desired_state(active: bool, shutting_down: bool) -> gstreamer::State {
    if active && !shutting_down {
        gstreamer::State::Playing
    } else {
        gstreamer::State::Null
    }
}

fn current_desired_state(flags: &RestartFlags) -> gstreamer::State {
    desired_state(
        flags.active.load(Ordering::SeqCst),
        flags.shutting_down.load(Ordering::SeqCst),
    )
}

/// Drive the pipeline to the desired state (`Null`, or a fresh
/// `Null → Playing`) on a `std::thread`, so the GTK main thread is
/// never blocked on a state transition.
///
/// At most one worker runs at a time (`restart_in_flight`); returns
/// `false` without spawning when one is already running. That is safe
/// for callers that just changed `active`: the running worker re-reads
/// the desired state after clearing `restart_in_flight` and loops
/// again if it no longer matches what it applied, so the last writer
/// of `active` always wins.
pub(super) fn spawn_restart_worker(
    pipeline: &gstreamer::Pipeline,
    flags: &Arc<RestartFlags>,
) -> bool {
    if flags
        .restart_in_flight
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return false;
    }
    // Capture owned clones so the closure stays `'static`.
    let pipeline = pipeline.clone();
    let flags = Arc::clone(flags);
    std::thread::spawn(move || {
        loop {
            let target = current_desired_state(&flags);
            // Always reset to Null first: a restart after bus
            // Error/EOS must leave the stale `Playing` state.
            if let Err(e) = pipeline.set_state(gstreamer::State::Null) {
                warn!(error = %e, "preview pipeline reset to Null failed");
            }
            if target == gstreamer::State::Playing {
                if let Err(e) = pipeline.set_state(gstreamer::State::Playing) {
                    warn!(error = %e, "preview pipeline retry to Playing failed");
                }
            }
            debug!(target = ?target, "preview pipeline state applied");
            flags.restart_in_flight.store(false, Ordering::SeqCst);

            // Deactivation / `Drop` may have changed the desired state
            // while we were transitioning; roll forward to it unless
            // another worker already took over.
            if current_desired_state(&flags) == target
                || flags
                    .restart_in_flight
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                    .is_err()
            {
                break;
            }
        }
    });
    true
}

#[cfg(test)]
mod tests {
    use super::{desired_state, should_restart};
    use gstreamer::State::{Null, Paused, Playing, VoidPending};

    #[test]
    fn playing_without_flag_does_not_restart() {
        assert!(!should_restart(Playing, VoidPending, false, false));
    }

    /// Regression: after a bus Error/EOS the pipeline still reports
    /// `Playing`; the old state-only check never restarted it.
    #[test]
    fn playing_with_needs_restart_restarts() {
        assert!(should_restart(Playing, VoidPending, true, false));
    }

    #[test]
    fn pending_playing_does_not_restart() {
        assert!(!should_restart(Paused, Playing, false, false));
    }

    #[test]
    fn in_flight_worker_blocks_restart() {
        assert!(!should_restart(Null, VoidPending, false, true));
        assert!(!should_restart(Playing, VoidPending, true, true));
    }

    #[test]
    fn null_restarts() {
        assert!(should_restart(Null, VoidPending, false, false));
    }

    #[test]
    fn desired_state_is_playing_only_when_active_and_not_shutting_down() {
        assert_eq!(desired_state(true, false), Playing);
        assert_eq!(desired_state(false, false), Null);
        assert_eq!(desired_state(true, true), Null);
        assert_eq!(desired_state(false, true), Null);
    }
}
