//! IPC layer: connects the GUI to the daemon's control socket.
//!
//! The worker runs on a background relm4 thread; the AppModel
//! communicates with it via `WorkerInput`/`WorkerOutput` messages.
//! Synchronous `UnixStream` I/O is wrapped in a small thread rather
//! than driven through `gio::SocketClient` — keeps the dependency
//! surface minimal and avoids glib executor pitfalls. Latency is not
//! a concern here (commands are infrequent and per-request).

mod wire;
pub mod worker;

pub use worker::{IpcWorker, WorkerInput, WorkerOutput};
