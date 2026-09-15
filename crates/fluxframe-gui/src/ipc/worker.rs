//! Background relm4 worker driving the UNIX-socket IPC.
//!
//! The worker holds an open [`UnixStream`] to the daemon. Each
//! [`WorkerInput::Send`] writes the command as a JSON line and reads
//! the response back; the result returns via [`WorkerOutput::Reply`].
//! The Stage-14 handshake (`list_effects` + `list_presets` +
//! `current_preset` + `get_config` + `config_path`) runs once on the
//! worker thread right after init (via a self-sent
//! [`WorkerInput::Connect`] — relm4 runs `Worker::init` on the
//! caller's, i.e. the GTK main, thread, so `init` itself must not
//! block) and the result is delivered as a single
//! [`WorkerOutput::Connected`] message so
//! the AppModel does not have to weave five reply tags.
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
    /// Internal: open the socket and run the handshake. Self-sent by
    /// `init` so the blocking IO happens on the worker thread instead
    /// of the GTK main thread. A repeated `Connect` while already
    /// connected is a no-op.
    Connect,
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
    /// Connection established and the five startup queries completed.
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
/// would otherwise dominate the enum's size — the `EffectSchema`
/// vectors inside `EffectInventory` makes the struct itself ~200 bytes; the
/// rest of `WorkerOutput` is small.
#[derive(Debug)]
pub struct InitialState {
    /// Inventory + build features.
    pub inventory: EffectInventory,
    /// All preset names.
    pub presets: Vec<String>,
    /// Currently-active preset name.
    pub active_preset: String,
    /// Active preset's full configuration, as returned by
    /// `get_config { path: None }`. Consumed by the chain editor.
    pub active_config: serde_json::Value,
    /// Writable TOML path the daemon will Save into, or `None` when
    /// the daemon was started with no resolvable config. Surfaced by
    /// the GUI's Save button tooltip and used to grey the button out
    /// when `None`.
    pub config_path: Option<PathBuf>,
}

/// Read timeout applied to the socket so a wedged daemon does not
/// hang the GUI indefinitely. 5 s is well above any realistic
/// `set_preset` cost (which includes ONNX session reload).
const READ_TIMEOUT: Duration = Duration::from_secs(5);

/// The relm4 worker handle.
pub struct IpcWorker {
    /// Daemon socket path, consumed by [`WorkerInput::Connect`].
    socket_path: PathBuf,
    /// Daemon socket; `None` before the connect and after a disconnect.
    stream: Option<BufferedStream>,
}

/// Bundled reader+writer halves of the connected socket.
struct BufferedStream {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
    /// Read timeout in effect, echoed in the timeout error message.
    read_timeout: Duration,
}

/// Failure of a single IPC round trip.
#[derive(Debug)]
enum IpcError {
    /// Socket write or read failed.
    Io(std::io::Error),
    /// The daemon did not reply within the read timeout.
    Timeout(Duration),
    /// The daemon closed the connection (EOF on read).
    Closed,
    /// The command could not be serialised.
    Serialize(serde_json::Error),
    /// The response line is not a valid [`Response`].
    Parse(serde_json::Error),
}

impl std::fmt::Display for IpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "socket IO failed: {e}"),
            Self::Timeout(d) if d.subsec_millis() == 0 => {
                write!(f, "daemon did not reply within {} s", d.as_secs())
            }
            Self::Timeout(d) => write!(f, "daemon did not reply within {} ms", d.as_millis()),
            Self::Closed => f.write_str("daemon closed the connection"),
            Self::Serialize(e) => write!(f, "serialise command failed: {e}"),
            Self::Parse(e) => write!(f, "parse response failed: {e}"),
        }
    }
}

impl BufferedStream {
    fn connect(path: &std::path::Path) -> std::io::Result<Self> {
        let stream = UnixStream::connect(path)?;
        tracing::info!(path = %path.display(), "socket connected, starting handshake");
        Self::from_stream(stream, READ_TIMEOUT)
    }

    /// Wrap an already-connected socket, applying `read_timeout`.
    fn from_stream(stream: UnixStream, read_timeout: Duration) -> std::io::Result<Self> {
        stream.set_read_timeout(Some(read_timeout))?;
        let writer = stream.try_clone()?;
        Ok(Self {
            reader: BufReader::new(stream),
            writer,
            read_timeout,
        })
    }

    /// Send one command line, read one response line, parse it.
    #[tracing::instrument(skip(self), level = "debug")]
    fn round_trip(&mut self, cmd: &Command) -> Result<Response, IpcError> {
        tracing::debug!(?cmd, "ipc round trip start");
        let payload = serde_json::to_string(cmd).map_err(IpcError::Serialize)?;
        self.writer
            .write_all(payload.as_bytes())
            .and_then(|()| self.writer.write_all(b"\n"))
            .and_then(|()| self.writer.flush())
            .map_err(IpcError::Io)?;
        let mut line = String::new();
        let n = self
            .reader
            .read_line(&mut line)
            .map_err(|e| match e.kind() {
                // `SO_RCVTIMEO` expiry surfaces as `WouldBlock` on Linux.
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => {
                    IpcError::Timeout(self.read_timeout)
                }
                _ => IpcError::Io(e),
            })?;
        if n == 0 {
            return Err(IpcError::Closed);
        }
        serde_json::from_str::<Response>(line.trim_end()).map_err(IpcError::Parse)
    }
}

impl Worker for IpcWorker {
    /// `init` receives the socket path. It performs no IO: relm4 runs
    /// `init` on the caller's (GTK main) thread, so the connect and
    /// handshake are deferred to a self-sent [`WorkerInput::Connect`]
    /// handled on the worker thread. Failures still surface as
    /// [`WorkerOutput::Disconnected`] before any `Send` is served,
    /// because `Connect` is the first message in the queue.
    type Init = PathBuf;
    type Input = WorkerInput;
    type Output = WorkerOutput;

    fn init(socket_path: Self::Init, sender: ComponentSender<Self>) -> Self {
        sender.input(WorkerInput::Connect);
        Self {
            socket_path,
            stream: None,
        }
    }

    fn update(&mut self, msg: Self::Input, sender: ComponentSender<Self>) {
        let (tag, command) = match msg {
            WorkerInput::Connect => {
                self.connect(&sender);
                return;
            }
            WorkerInput::Send { tag, command } => (tag, command),
        };
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
            Err(err) => {
                let reason = err.to_string();
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

impl IpcWorker {
    /// Open the socket and run the handshake, emitting
    /// [`WorkerOutput::Connected`] or [`WorkerOutput::Disconnected`].
    /// Runs on the worker thread; no-op when already connected.
    fn connect(&mut self, sender: &ComponentSender<Self>) {
        if self.stream.is_some() {
            tracing::debug!("ipc worker already connected; ignoring Connect");
            return;
        }
        match BufferedStream::connect(&self.socket_path) {
            Ok(mut stream) => match handshake(&mut stream) {
                Ok(initial) => {
                    let _ = sender.output(WorkerOutput::Connected(Box::new(initial)));
                    self.stream = Some(stream);
                }
                Err(reason) => {
                    let _ = sender.output(WorkerOutput::Disconnected { reason });
                }
            },
            Err(e) => {
                let hint = match e.kind() {
                    std::io::ErrorKind::ConnectionRefused => {
                        "the daemon is not running or not listening on this socket"
                    }
                    std::io::ErrorKind::NotFound => {
                        "the socket file does not exist — check the daemon's [control] config"
                    }
                    std::io::ErrorKind::PermissionDenied => {
                        "permission denied — check the socket file's owner/mode"
                    }
                    _ => "connection failed",
                };
                let _ = sender.output(WorkerOutput::Disconnected {
                    reason: format!("{hint} ({e})"),
                });
            }
        }
    }
}

/// Run the Stage-14 startup handshake: `list_effects` →
/// `list_presets` → `current_preset` → `get_config(None)` →
/// `config_path` (optional).
///
/// Pulled out as a free function so tests drive it over a
/// `UnixStream::pair` without spinning up the relm4 runtime.
#[tracing::instrument(skip(stream), level = "info")]
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
    // Optional handshake step — older daemons that do not yet expose
    // `ConfigPath` reply with `Err(unknown command)`. Surface that as
    // "no writable path" instead of failing the whole handshake, so
    // the GUI gracefully degrades to read-only mode and the operator
    // still sees the editor.
    let config_path = match stream.round_trip(&Command::ConfigPath) {
        Ok(Response::Ok { data }) => parse_config_path(&data),
        Ok(Response::Err { error, hint }) => {
            tracing::warn!(
                error,
                ?hint,
                "config_path command refused; Save will be disabled"
            );
            None
        }
        Ok(_) => None,
        Err(reason) => return Err(format!("config_path failed: {reason}")),
    };
    Ok(InitialState {
        inventory,
        presets,
        active_preset,
        active_config,
        config_path,
    })
}

/// Decode the `ConfigPath` payload: `{"path":"…"}` → `Some(path)`,
/// JSON `null` → `None`. Anything else is treated as `None` with a
/// debug-level breadcrumb so a daemon shape change does not silently
/// hide a real path.
fn parse_config_path(data: &serde_json::Value) -> Option<PathBuf> {
    if data.is_null() {
        return None;
    }
    if let Some(p) = data.get("path").and_then(serde_json::Value::as_str) {
        Some(PathBuf::from(p))
    } else {
        tracing::debug!(
            ?data,
            "config_path: unexpected payload shape; treating as None"
        );
        None
    }
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
    match stream.round_trip(cmd).map_err(|e| e.to_string())? {
        Response::Ok { data } => parse(data),
        Response::Err { error, hint } => Err(format!("{label} failed: {error}{}", fmt_hint(hint))),
        // `Response` is `#[non_exhaustive]`; surface unknown variants
        // as a hard handshake failure rather than panicking.
        _ => Err(format!("{label} failed: unrecognised Response variant")),
    }
}

fn fmt_hint(hint: Option<String>) -> String {
    hint.map_or_else(String::new, |h| format!(" (hint: {h})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipc_error_display_distinguishes_timeout_from_closed() {
        assert_eq!(
            IpcError::Timeout(READ_TIMEOUT).to_string(),
            "daemon did not reply within 5 s"
        );
        assert_eq!(IpcError::Closed.to_string(), "daemon closed the connection");
    }

    #[test]
    fn round_trip_reports_closed_peer() {
        let (ours, peer) = UnixStream::pair().expect("socket pair");
        let mut stream = BufferedStream::from_stream(ours, READ_TIMEOUT).expect("wrap");
        // Read the command line, then close without replying.
        let mut req = String::new();
        let reader = std::thread::spawn(move || {
            BufReader::new(peer)
                .read_line(&mut req)
                .expect("read request");
        });
        let err = stream
            .round_trip(&Command::ListPresets)
            .expect_err("closed peer must fail");
        reader.join().expect("peer thread");
        assert!(matches!(err, IpcError::Closed), "got {err:?}");
    }

    #[test]
    fn round_trip_reports_silent_peer_as_timeout() {
        let (ours, _peer) = UnixStream::pair().expect("socket pair");
        let mut stream =
            BufferedStream::from_stream(ours, Duration::from_millis(50)).expect("wrap");
        let err = stream
            .round_trip(&Command::ListPresets)
            .expect_err("silent peer must time out");
        assert!(matches!(err, IpcError::Timeout(_)), "got {err:?}");
        assert_eq!(err.to_string(), "daemon did not reply within 50 ms");
    }

    /// Spawn a mock daemon on `peer`: for each `(expected, reply)` pair,
    /// read one request line, assert it decodes to `expected`, and
    /// write `reply` back. The peer socket is dropped afterwards.
    fn spawn_mock_daemon(
        peer: UnixStream,
        script: Vec<(Command, Response)>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let mut writer = peer.try_clone().expect("clone peer");
            let mut reader = BufReader::new(peer);
            for (expected, reply) in script {
                let mut line = String::new();
                reader.read_line(&mut line).expect("read request");
                let got: Command = serde_json::from_str(line.trim_end()).expect("decode request");
                assert_eq!(got, expected, "handshake request order");
                let payload = serde_json::to_string(&reply).expect("encode reply");
                writer
                    .write_all(format!("{payload}\n").as_bytes())
                    .expect("write reply");
            }
        })
    }

    fn inventory_payload() -> serde_json::Value {
        serde_json::json!({
            "mask": [],
            "background": [],
            "foreground": [],
            "post": [],
            "build_features": ["ml"]
        })
    }

    /// Full five-step script; `config_path_reply` is the last answer.
    fn handshake_script(config_path_reply: Response) -> Vec<(Command, Response)> {
        vec![
            (Command::ListEffects, Response::ok_with(inventory_payload())),
            (
                Command::ListPresets,
                Response::ok_with(serde_json::json!(["default", "office"])),
            ),
            (
                Command::CurrentPreset,
                Response::ok_with(serde_json::json!("office")),
            ),
            (
                Command::GetConfig { path: None },
                Response::ok_with(serde_json::json!({"marker": 42})),
            ),
            (Command::ConfigPath, config_path_reply),
        ]
    }

    fn run_handshake(script: Vec<(Command, Response)>) -> Result<InitialState, String> {
        let (ours, peer) = UnixStream::pair().expect("socket pair");
        let mut stream = BufferedStream::from_stream(ours, READ_TIMEOUT).expect("wrap");
        let daemon = spawn_mock_daemon(peer, script);
        let result = handshake(&mut stream);
        daemon.join().expect("mock daemon thread");
        result
    }

    #[test]
    fn handshake_collects_all_five_replies() {
        let initial = run_handshake(handshake_script(Response::ok_with(
            serde_json::json!({"path": "/etc/fluxframe.toml"}),
        )))
        .expect("handshake succeeds");
        assert_eq!(initial.presets, vec!["default", "office"]);
        assert_eq!(initial.active_preset, "office");
        assert_eq!(initial.active_config, serde_json::json!({"marker": 42}));
        assert_eq!(initial.inventory.build_features, vec!["ml".to_string()]);
        assert_eq!(
            initial.config_path,
            Some(PathBuf::from("/etc/fluxframe.toml"))
        );
    }

    /// An old daemon refusing `config_path` must degrade to read-only
    /// mode, not fail the handshake.
    #[test]
    fn handshake_tolerates_refused_config_path() {
        let initial = run_handshake(handshake_script(Response::err("unknown command", None)))
            .expect("refused config_path must not fail the handshake");
        assert_eq!(initial.config_path, None);
        assert_eq!(initial.active_preset, "office");
    }

    /// A refused mandatory step aborts the handshake with a labelled
    /// error; no further requests are sent.
    #[test]
    fn handshake_fails_on_refused_list_effects() {
        let Err(err) = run_handshake(vec![(
            Command::ListEffects,
            Response::err("boom", Some("restart".to_string())),
        )]) else {
            panic!("refused list_effects must fail the handshake");
        };
        assert!(err.contains("list_effects"), "got: {err}");
        assert!(err.contains("boom"), "got: {err}");
    }

    #[test]
    fn parse_config_path_decodes_daemon_shapes() {
        assert_eq!(parse_config_path(&serde_json::Value::Null), None);
        assert_eq!(
            parse_config_path(&serde_json::json!({"path": "/x"})),
            Some(PathBuf::from("/x"))
        );
        assert_eq!(parse_config_path(&serde_json::json!({})), None);
    }
}
