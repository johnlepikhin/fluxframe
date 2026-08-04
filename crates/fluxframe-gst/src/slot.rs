//! Latest-frame slot — bounded single-element buffer with drop-old semantics.
//!
//! Implements §22 of the spec ("better lose a frame than accumulate latency"):
//! the producer (an `appsink` callback) writes the freshest captured frame
//! into the slot, overwriting any frame the consumer has not yet picked up.
//! The processing worker pulls the most recent frame via
//! [`LatestFrameSlot::recv_timeout`].
//!
//! Stage 1 minimal implementation; full bounded-queue + drop-policy
//! instrumentation lands in Stage 5.

use std::sync::Arc;
use std::time::Duration;

use fluxframe_core::frame::VideoFrame;
use parking_lot::{Condvar, Mutex};

/// Single-frame slot shared between the capture thread and the processing
/// worker.
///
/// Cloning the slot is cheap — it shares the same backing storage via `Arc`.
#[derive(Clone)]
pub struct LatestFrameSlot {
    inner: Arc<Inner>,
}

struct Inner {
    state: Mutex<State>,
    cond: Condvar,
}

#[derive(Default)]
struct State {
    /// Most recent frame produced by the capture stage, if any.
    frame: Option<VideoFrame>,
    /// Monotonically increasing tally of frames dropped because the
    /// previous frame had not been consumed yet.  Surfaces in metrics
    /// from Stage 5.
    dropped: u64,
    /// `true` once [`LatestFrameSlot::close`] has been called.
    closed: bool,
}

impl Default for LatestFrameSlot {
    fn default() -> Self {
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State::default()),
                cond: Condvar::new(),
            }),
        }
    }
}

impl LatestFrameSlot {
    /// Construct an empty slot.
    ///
    /// In production a slot is only ever instantiated by
    /// [`crate::input::InputPipeline`], which then hands a clone to the
    /// consumer via `InputPipeline::slot()`. It is public because the
    /// slot is a self-contained data structure with no GStreamer
    /// dependency, which lets the supervisor's tests exercise
    /// slot-lifecycle behaviour without a live camera.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Has the slot been closed?
    ///
    /// Closing is one-way — there is no reopen — so this is effectively
    /// "is this slot permanently dead". Callers that recover a pipeline
    /// in place need to be sure they never closed it, because every
    /// subsequent [`LatestFrameSlot::push`] would be a silent no-op.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.inner.state.lock().closed
    }

    /// Publish `frame`, replacing any previous frame that has not been
    /// consumed.  Increments the drop counter on overwrite.  No-op if the
    /// slot has been closed.
    ///
    /// Crate-private: only the capture `appsink` callback pushes; external
    /// code consumes via [`Self::recv_timeout`].
    pub(crate) fn push(&self, frame: VideoFrame) {
        let mut state = self.inner.state.lock();
        if state.closed {
            return;
        }
        if state.frame.is_some() {
            state.dropped = state.dropped.saturating_add(1);
        }
        state.frame = Some(frame);
        self.inner.cond.notify_one();
    }

    /// Receive the latest frame, blocking up to `timeout` if the slot is
    /// empty.  Returns `None` if the slot was closed or the timeout
    /// elapsed.
    ///
    /// Named to match the semantics of
    /// `crossbeam_channel::Receiver::recv_timeout` — a blocking timed
    /// receive, *not* the std-library `Option::take` non-blocking take
    /// the previous name suggested.
    #[must_use]
    pub fn recv_timeout(&self, timeout: Duration) -> Option<VideoFrame> {
        let mut state = self.inner.state.lock();
        loop {
            if let Some(frame) = state.frame.take() {
                return Some(frame);
            }
            if state.closed {
                return None;
            }
            if self.inner.cond.wait_for(&mut state, timeout).timed_out() {
                return None;
            }
        }
    }

    /// Snapshot of frames dropped due to overwrite since slot creation.
    #[must_use]
    pub fn dropped_count(&self) -> u64 {
        self.inner.state.lock().dropped
    }

    /// Discard any pending frame without affecting the drop counter or
    /// the closed flag. Used by Stage 15 idle entry to drop the last
    /// frame the capture thread published before the input pipeline
    /// transitioned to `Null` — without this, resuming the worker
    /// would process one stale frame from before the idle window.
    pub fn clear(&self) {
        let mut state = self.inner.state.lock();
        state.frame = None;
    }

    /// Mark the slot closed and wake any waiting consumer.  Subsequent
    /// pushes are no-ops; subsequent receives return `None` once the
    /// buffered frame (if any) has been drained.
    ///
    /// Public because the CLI Ctrl-C handler needs to wake a worker
    /// blocked in [`Self::recv_timeout`] without having access to the
    /// owning [`crate::input::InputPipeline`].
    pub fn close(&self) {
        let mut state = self.inner.state.lock();
        state.closed = true;
        // Drop the last buffered frame so it doesn't sit in the slot
        // holding onto its pixel buffer until the slot itself is dropped.
        state.frame = None;
        self.inner.cond.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use fluxframe_core::frame::{FrameBuffer, FrameMeta, PixelFormat};

    use super::*;

    fn dummy_frame(seq: u64) -> VideoFrame {
        VideoFrame::new_packed(
            FrameBuffer::Owned(vec![0u8; 12]),
            2,
            2,
            PixelFormat::Rgb,
            FrameMeta {
                sequence: seq,
                ..Default::default()
            },
        )
        .expect("dummy frame builds")
    }

    #[test]
    fn push_overwrite_increments_drop_counter() {
        let slot = LatestFrameSlot::new();
        slot.push(dummy_frame(1));
        slot.push(dummy_frame(2));
        slot.push(dummy_frame(3));
        assert_eq!(slot.dropped_count(), 2);
        let f = slot
            .recv_timeout(Duration::from_millis(10))
            .expect("frame ready");
        assert_eq!(f.meta.sequence, 3, "must return the freshest frame");
    }

    #[test]
    fn recv_timeout_blocks_then_wakes_on_push() {
        let slot = LatestFrameSlot::new();
        let producer = {
            let slot = slot.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(20));
                slot.push(dummy_frame(42));
            })
        };
        let frame = slot
            .recv_timeout(Duration::from_millis(500))
            .expect("producer publishes within window");
        assert_eq!(frame.meta.sequence, 42);
        producer.join().unwrap();
    }

    #[test]
    fn recv_timeout_returns_none_on_timeout() {
        let slot = LatestFrameSlot::new();
        let r = slot.recv_timeout(Duration::from_millis(10));
        assert!(r.is_none());
    }

    #[test]
    fn close_unblocks_consumer() {
        let slot = LatestFrameSlot::new();
        let closer = {
            let slot = slot.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(20));
                slot.close();
            })
        };
        let r = slot.recv_timeout(Duration::from_secs(5));
        assert!(r.is_none(), "closed slot returns None");
        closer.join().unwrap();
    }

    #[test]
    fn push_after_close_is_noop() {
        let slot = LatestFrameSlot::new();
        slot.close();
        slot.push(dummy_frame(1));
        assert!(slot.recv_timeout(Duration::from_millis(10)).is_none());
    }
}
