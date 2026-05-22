//! Typed GStreamer bus event façade.
//!
//! `fluxframe-cli` does not know about `gstreamer::*` — it consumes
//! [`BusEvent`]s through this module.  [`BusListener::spawn`] drains the
//! bus(es) on a background thread and forwards events via a callback.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use gstreamer::prelude::*;
use tracing::{trace, warn};

/// Polling interval for `bus.timed_pop_filtered`.
///
/// Short enough that shutdown propagation is responsive (~50ms), long
/// enough that idle pipelines don't spin.
const BUS_POLL_TIMEOUT: Duration = Duration::from_millis(50);

/// Typed event extracted from a GStreamer bus.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BusEvent {
    /// A fatal element error.  The pipeline should be torn down.
    FatalError {
        /// Pipeline element name that originated the error.
        element: String,
        /// Human-readable error message.
        message: String,
        /// Optional debug payload.
        debug: Option<String>,
        /// Label identifying which pipeline produced the event ("input"/"output"/etc).
        source_label: &'static str,
    },
    /// A non-fatal element warning.
    Warning {
        /// Pipeline element name that originated the warning.
        element: String,
        /// Human-readable warning message.
        message: String,
        /// Optional debug payload.
        debug: Option<String>,
        /// Label identifying which pipeline produced the event.
        source_label: &'static str,
    },
    /// Pipeline reached end-of-stream.
    Eos {
        /// Label identifying which pipeline produced the event.
        source_label: &'static str,
    },
}

/// A registered pipeline whose bus the listener should drain.
pub struct WatchedPipeline {
    /// Stable label used in [`BusEvent`] `source_label` fields.
    pub label: &'static str,
    /// Bus to drain.
    pub bus: gstreamer::Bus,
}

/// Background bus listener that calls `on_event` for each surfaced event.
pub struct BusListener {
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl BusListener {
    /// Spawn a listener draining `pipelines`.  `on_event` is invoked
    /// from the listener thread; it must be `Send + Sync` and short
    /// (no blocking work).
    #[must_use]
    pub fn spawn<F>(pipelines: Vec<WatchedPipeline>, on_event: F) -> Self
    where
        F: Fn(BusEvent) + Send + Sync + 'static,
    {
        let running = Arc::new(AtomicBool::new(true));
        let running_thread = Arc::clone(&running);
        let handle = std::thread::Builder::new()
            .name("fluxframe-bus".into())
            .spawn(move || run_loop(&pipelines, &running_thread, &on_event))
            .expect("spawn bus listener thread");
        Self {
            running,
            handle: Some(handle),
        }
    }

    /// Signal the listener to stop and block until it exits.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            if let Err(payload) = handle.join() {
                warn!(?payload, "bus listener thread panicked during join");
            }
        }
    }
}

impl Drop for BusListener {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run_loop<F>(pipelines: &[WatchedPipeline], running: &AtomicBool, on_event: &F)
where
    F: Fn(BusEvent),
{
    let poll = match u64::try_from(BUS_POLL_TIMEOUT.as_millis()) {
        Ok(ms) => gstreamer::ClockTime::from_mseconds(ms),
        Err(_) => gstreamer::ClockTime::from_mseconds(50),
    };
    while running.load(Ordering::Acquire) {
        for p in pipelines {
            let Some(msg) = p.bus.timed_pop_filtered(
                Some(poll),
                &[
                    gstreamer::MessageType::Error,
                    gstreamer::MessageType::Warning,
                    gstreamer::MessageType::Eos,
                ],
            ) else {
                continue;
            };
            on_event(translate(&msg, p.label));
        }
    }
    trace!("bus listener loop exited");
}

fn translate(msg: &gstreamer::Message, source_label: &'static str) -> BusEvent {
    use gstreamer::MessageView;
    match msg.view() {
        MessageView::Error(e) => BusEvent::FatalError {
            element: e
                .src()
                .map_or_else(|| "?".into(), |s| s.name().to_string()),
            message: e.error().to_string(),
            debug: e.debug().map(|d| d.to_string()),
            source_label,
        },
        MessageView::Warning(w) => BusEvent::Warning {
            element: w
                .src()
                .map_or_else(|| "?".into(), |s| s.name().to_string()),
            message: w.error().to_string(),
            debug: w.debug().map(|d| d.to_string()),
            source_label,
        },
        MessageView::Eos(_) => BusEvent::Eos { source_label },
        _ => unreachable!("filter excludes other message types"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bus_event_debug_compiles() {
        // Sanity check that the enum derives `Debug` and `Clone`.
        let evt = BusEvent::Eos {
            source_label: "test",
        };
        let _cloned = evt.clone();
        let s = format!("{evt:?}");
        assert!(s.contains("Eos"));

        let fatal = BusEvent::FatalError {
            element: "decoder".into(),
            message: "boom".into(),
            debug: Some("trace".into()),
            source_label: "input",
        };
        let s = format!("{fatal:?}");
        assert!(s.contains("FatalError"));
        assert!(s.contains("decoder"));

        let warn_evt = BusEvent::Warning {
            element: "encoder".into(),
            message: "minor".into(),
            debug: None,
            source_label: "output",
        };
        let s = format!("{warn_evt:?}");
        assert!(s.contains("Warning"));
    }

    #[test]
    fn bus_listener_stops_on_drop() {
        // No pipelines — the run loop just polls `running` and exits when
        // we drop the listener (Drop calls stop, which joins the thread).
        {
            let _listener = BusListener::spawn(Vec::new(), |_| {});
            // Hand the listener a brief moment to enter the loop, then drop.
            std::thread::sleep(Duration::from_millis(10));
        }
        // Reaching here means `Drop::drop` joined the thread without panicking.
    }

    #[test]
    fn bus_listener_explicit_stop_is_idempotent() {
        let mut listener = BusListener::spawn(Vec::new(), |_| {});
        listener.stop();
        // Second stop after the handle has been taken should still be safe.
        listener.stop();
    }
}
