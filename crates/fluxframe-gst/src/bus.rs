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

/// Which pipeline the event originated from.  Typed source label so
/// downstream EBUSY translation can dispatch into the right
/// [`fluxframe_core::error::PipelineError`] variant without parsing
/// magic strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BusSource {
    /// Input/capture pipeline.
    Input,
    /// Output/sink pipeline.
    Output,
}

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
        /// Typed source identifying which pipeline produced the event.
        source: BusSource,
    },
    /// A non-fatal element warning.
    Warning {
        /// Pipeline element name that originated the warning.
        element: String,
        /// Human-readable warning message.
        message: String,
        /// Optional debug payload.
        debug: Option<String>,
        /// Typed source identifying which pipeline produced the event.
        source: BusSource,
    },
    /// Pipeline reached end-of-stream.
    Eos {
        /// Typed source identifying which pipeline produced the event.
        source: BusSource,
    },
}

/// A registered pipeline whose bus the listener should drain.
pub struct WatchedPipeline {
    /// Typed source identifying which pipeline emitted the event.
    pub source: BusSource,
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
            on_event(translate(&msg, p.source));
        }
    }
    trace!("bus listener loop exited");
}

fn translate(msg: &gstreamer::Message, source: BusSource) -> BusEvent {
    use gstreamer::MessageView;
    match msg.view() {
        MessageView::Error(e) => BusEvent::FatalError {
            element: e.src().map_or_else(|| "?".into(), |s| s.name().to_string()),
            message: e.error().to_string(),
            debug: e.debug().map(|d| d.to_string()),
            source,
        },
        MessageView::Warning(w) => BusEvent::Warning {
            element: w.src().map_or_else(|| "?".into(), |s| s.name().to_string()),
            message: w.error().to_string(),
            debug: w.debug().map(|d| d.to_string()),
            source,
        },
        MessageView::Eos(_) => BusEvent::Eos { source },
        _ => unreachable!("filter excludes other message types"),
    }
}

/// Promote a [`BusEvent::FatalError`] to a structured
/// [`fluxframe_core::error::PipelineError`].
///
/// Recognises `EBUSY`-style messages and routes them to
/// [`fluxframe_core::error::PipelineError::InputDeviceUnavailable`] /
/// [`fluxframe_core::error::PipelineError::OutputDeviceUnavailable`]
/// per `event.source`.  Everything else becomes
/// [`fluxframe_core::error::PipelineError::BusError`] carrying the
/// original message.
///
/// **Returns `None`** if the event is not a `FatalError`.
#[must_use]
pub fn translate_fatal(event: &BusEvent) -> Option<fluxframe_core::error::PipelineError> {
    use fluxframe_core::error::PipelineError;
    let BusEvent::FatalError {
        element,
        message,
        debug,
        source,
    } = event
    else {
        return None;
    };
    if looks_busy(message) {
        // `BusSource` is `#[non_exhaustive]`, but within this crate the
        // compiler still sees every variant.  We list them explicitly so
        // adding a new variant is a compile error here — forcing a
        // conscious decision about EBUSY mapping for the new source.
        return Some(match source {
            BusSource::Input => PipelineError::InputDeviceUnavailable {
                device: element.clone(),
                reason: "device is busy".into(),
                hint: "another application is holding the device; close it (e.g. browser tab, OBS)"
                    .into(),
            },
            BusSource::Output => PipelineError::OutputDeviceUnavailable {
                device: element.clone(),
                reason: "device is busy".into(),
                hint: "another application is using the loopback device".into(),
            },
        });
    }
    Some(PipelineError::BusError {
        element: element.clone(),
        message: message.clone(),
        debug: debug.clone(),
    })
}

/// Narrow `EBUSY`-style substring detector.
///
/// Only matches the kernel/GStreamer phrasings that genuinely mean
/// "device or resource busy".  A bare `"busy"` substring would catch
/// false positives like `"keep busy retrying..."`.
fn looks_busy(message: &str) -> bool {
    let lowered = message.to_lowercase();
    lowered.contains("device or resource busy")
        || lowered.contains("resource busy")
        || lowered.contains("device busy")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bus_event_debug_compiles() {
        // Sanity check that the enum derives `Debug` and `Clone`.
        let evt = BusEvent::Eos {
            source: BusSource::Input,
        };
        let _cloned = evt.clone();
        let s = format!("{evt:?}");
        assert!(s.contains("Eos"));

        let fatal = BusEvent::FatalError {
            element: "decoder".into(),
            message: "boom".into(),
            debug: Some("trace".into()),
            source: BusSource::Input,
        };
        let s = format!("{fatal:?}");
        assert!(s.contains("FatalError"));
        assert!(s.contains("decoder"));

        let warn_evt = BusEvent::Warning {
            element: "encoder".into(),
            message: "minor".into(),
            debug: None,
            source: BusSource::Output,
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

    #[test]
    fn translate_fatal_promotes_input_busy() {
        let event = BusEvent::FatalError {
            element: "v4l2src0".into(),
            message: "Device or resource busy".into(),
            debug: None,
            source: BusSource::Input,
        };
        let err = translate_fatal(&event).expect("fatal");
        assert!(matches!(
            err,
            fluxframe_core::error::PipelineError::InputDeviceUnavailable { .. }
        ));
    }

    #[test]
    fn translate_fatal_promotes_output_busy() {
        let event = BusEvent::FatalError {
            element: "v4l2sink0".into(),
            message: "Resource busy: cannot open".into(),
            debug: None,
            source: BusSource::Output,
        };
        let err = translate_fatal(&event).expect("fatal");
        assert!(matches!(
            err,
            fluxframe_core::error::PipelineError::OutputDeviceUnavailable { .. }
        ));
    }

    #[test]
    fn translate_fatal_falls_back_to_bus_error() {
        let event = BusEvent::FatalError {
            element: "videoconvert0".into(),
            message: "Internal data stream error".into(),
            debug: None,
            source: BusSource::Input,
        };
        let err = translate_fatal(&event).expect("fatal");
        assert!(matches!(
            err,
            fluxframe_core::error::PipelineError::BusError { .. }
        ));
    }

    #[test]
    fn translate_fatal_ignores_unrelated_busy_substring() {
        let event = BusEvent::FatalError {
            element: "filter".into(),
            // generic "busy" — must NOT trigger EBUSY promote
            message: "Keep busy retrying...".into(),
            debug: None,
            source: BusSource::Input,
        };
        let err = translate_fatal(&event).expect("fatal");
        assert!(matches!(
            err,
            fluxframe_core::error::PipelineError::BusError { .. }
        ));
    }

    #[test]
    fn translate_fatal_returns_none_for_warning() {
        let event = BusEvent::Warning {
            element: "x".into(),
            message: "y".into(),
            debug: None,
            source: BusSource::Input,
        };
        assert!(translate_fatal(&event).is_none());
    }
}
