//! Live-reconfiguration via UNIX socket (Stage 13).
//!
//! The control surface is opt-in (`[control].enabled = true` in TOML).
//! When enabled, the daemon spawns a listener thread that accepts
//! line-delimited JSON commands on a `0600`-mode UNIX socket and
//! routes them to the worker thread through a bounded channel.
//!
//! See `socket.rs` for the listener loop, `commands.rs` for the wire
//! format. Worker-side dispatch lives in `runtime.rs`.

pub mod commands;
pub mod socket;

pub use commands::{Command, Response};
pub use socket::{CommandHandler, ListenerHandle, default_socket_path, spawn};
