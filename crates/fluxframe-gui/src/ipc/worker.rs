//! Background relm4 worker driving the UNIX-socket IPC.
//!
//! The worker holds an open [`UnixStream`] to the daemon. Each
//! [`WorkerInput::Send`] writes the command as a JSON line and reads
//! the response back; the result returns via [`WorkerOutput::Reply`].
//! The Stage-14 handshake (`list_effects` + `list_presets` +
//! `current_preset` + `get_config`) runs once on init and the result
//! is delivered as a single [`WorkerOutput::Connected`] message so
//! the AppModel does not have to weave four reply tags.
//!
//! Failures at any IPC step transition the worker into a "disconnected"
//! state and surface as [`WorkerOutput::Disconnected`]; the AppModel
//! is expected to recreate the worker on user-driven retry.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

use fluxframe_core::protocol::{Command, Response};
use relm4::Worker;
use relm4::prelude::ComponentSender;

use crate::ipc::wire::{parse_inventory, parse_presets};
use crate::state::EffectInventory;

/// Commands sent from the AppModel into the worker thread.
#[derive(Debug)]
pub enum WorkerInput {
    /// Send a typed [`Command`]; the worker tags the reply so the
    /// AppModel can match it against the originating request.
    Send {
        /// Caller-supplied correlation id.
        tag: u64,
        /// The command to write on the wire.
        command: Command,
    },
}

/// Messages the worker emits back to the AppModel.
#[derive(Debug)]
pub enum WorkerOutput {
    /// Connection established and the four startup queries completed.
    Connected(Box<InitialState>),
    /// Reply to a `WorkerInput::Send`. `tag` echoes the request.
    Reply {
        /// Caller-supplied correlation id.
        tag: u64,
        /// Daemon response, parsed.
        response: Response,
    },
    /// Connection failed (initial connect, handshake, or steady-state
    /// IO). `reason` is suitable for an `adw::StatusPage` body.
    Disconnected {
        /// Human-readable reason for the disconnect.
        reason: String,
    },
}

/// Snapshot of the daemon-side state captured by the handshake.
///
/// Boxed because the [`Connected`](WorkerOutput::Connected) variant
/// would otherwise dominate the enum's size — `Vec<EffectMetadata>`
/// inside `EffectInventory` makes the struct itself ~200 bytes; the
/// rest of `WorkerOutput` is small.
#[allow(
    dead_code,
    reason = "`active_config` consumed by Stage 14 Step 5 chain editor"
)]
#[derive(Debug)]
pub struct InitialState {
    /// Inventory + build features.
    pub inventory: EffectInventory,
    /// All preset names.
    pub presets: Vec<String>,
    /// Currently-active preset name.
    pub active_preset: String,
    /// Active preset's full configuration, as returned by
    /// `get_config { path: None }`. Reserved for the chain editor in
    /// Step 5.
    pub active_config: serde_json::Value,
}

/// Read timeout applied to the socket so a wedged daemon does not
/// hang the GUI indefinitely. 5 s is well above any realistic
/// `set_preset` cost (which includes ONNX session reload).
const READ_TIMEOUT: Duration = Duration::from_secs(5);

/// The relm4 worker handle.
pub struct IpcWorker {
    /// Daemon socket; `None` after a disconnect.
    stream: Option<BufferedStream>,
}

/// Bundled reader+writer halves of the connected socket.
struct BufferedStream {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl BufferedStream {
    fn connect(path: &std::path::Path) -> std::io::Result<Self> {
        let stream = UnixStream::connect(path)?;
        tracing::info!(path = %path.display(), "socket connected, starting handshake");
        stream.set_read_timeout(Some(READ_TIMEOUT))?;
        let writer = stream.try_clone()?;
        Ok(Self {
            reader: BufReader::new(stream),
            writer,
        })
    }

    /// Send one command line, read one response line, parse it.
    fn round_trip(&mut self, cmd: &Command) -> Result<Response, String> {
        tracing::debug!(?cmd, "ipc round trip start");
        let payload =
            serde_json::to_string(cmd).map_err(|e| format!("serialise command failed: {e}"))?;
        self.writer
            .write_all(payload.as_bytes())
            .and_then(|()| self.writer.write_all(b"\n"))
            .and_then(|()| self.writer.flush())
            .map_err(|e| format!("write failed: {e}"))?;
        let mut line = String::new();
        let n = self
            .reader
            .read_line(&mut line)
            .map_err(|e| format!("read failed: {e}"))?;
        if n == 0 {
            return Err("daemon closed the connection".into());
        }
        serde_json::from_str::<Response>(line.trim_end())
            .map_err(|e| format!("parse response failed: {e}"))
    }
}

impl Worker for IpcWorker {
    /// `init` receives the socket path; opening the connection is
    /// part of `init` so any failure surfaces immediately rather than
    /// hiding behind the first `Send`.
    type Init = PathBuf;
    type Input = WorkerInput;
    type Output = WorkerOutput;

    fn init(socket_path: Self::Init, sender: ComponentSender<Self>) -> Self {
        match BufferedStream::connect(&socket_path) {
            Ok(mut stream) => match handshake(&mut stream) {
                Ok(initial) => {
                    let _ = sender.output(WorkerOutput::Connected(Box::new(initial)));
                    Self {
                        stream: Some(stream),
                    }
                }
                Err(reason) => {
                    let _ = sender.output(WorkerOutput::Disconnected { reason });
                    Self { stream: None }
                }
            },
            Err(e) => {
                let _ = sender.output(WorkerOutput::Disconnected {
                    reason: format!("connect to {} failed: {e}", socket_path.display()),
                });
                Self { stream: None }
            }
        }
    }

    fn update(&mut self, msg: Self::Input, sender: ComponentSender<Self>) {
        let WorkerInput::Send { tag, command } = msg;
        let Some(stream) = self.stream.as_mut() else {
            let _ = sender.output(WorkerOutput::Reply {
                tag,
                response: Response::err("not connected", None),
            });
            return;
        };
        match stream.round_trip(&command) {
            Ok(response) => {
                let _ = sender.output(WorkerOutput::Reply { tag, response });
            }
            Err(reason) => {
                // Tear down the dead stream so subsequent Send calls
                // surface "not connected" rather than retrying on a
                // half-broken socket.
                self.stream = None;
                let _ = sender.output(WorkerOutput::Reply {
                    tag,
                    response: Response::err(reason.clone(), None),
                });
                let _ = sender.output(WorkerOutput::Disconnected { reason });
            }
        }
    }
}

/// Run the Stage-14 startup handshake: `list_effects` →
/// `list_presets` → `current_preset` → `get_config(None)`.
///
/// Pulled out as a free function so a future test can drive it
/// against a mock `BufferedStream` without spinning up the relm4
/// runtime.
#[tracing::instrument(skip(stream))]
fn handshake(stream: &mut BufferedStream) -> Result<InitialState, String> {
    let inventory = one(stream, &Command::ListEffects, "list_effects", |d| {
        parse_inventory(&d)
    })?;
    let presets = one(stream, &Command::ListPresets, "list_presets", |d| {
        parse_presets(&d)
    })?;
    let active_preset = one(stream, &Command::CurrentPreset, "current_preset", |d| {
        d.as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("current_preset payload is not a string: {d}"))
    })?;
    let active_config = one(stream, &Command::GetConfig { path: None }, "get_config", Ok)?;
    Ok(InitialState {
        inventory,
        presets,
        active_preset,
        active_config,
    })
}

/// Execute one handshake step: round-trip the command, then either
/// run `parse` on the `Ok` payload or convert the daemon's `Err`
/// response into a labelled error string.
#[tracing::instrument(skip(stream, parse))]
fn one<T>(
    stream: &mut BufferedStream,
    cmd: &Command,
    label: &str,
    parse: impl FnOnce(serde_json::Value) -> Result<T, String>,
) -> Result<T, String> {
    match stream.round_trip(cmd)? {
        Response::Ok { data } => parse(data),
        Response::Err { error, hint } => Err(format!("{label} failed: {error}{}", fmt_hint(hint))),
    }
}

fn fmt_hint(hint: Option<String>) -> String {
    hint.map_or_else(String::new, |h| format!(" (hint: {h})"))
}
