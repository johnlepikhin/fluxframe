//! UNIX-socket listener thread for the control surface.
//!
//! The thread:
//! 1. Binds the socket on a sibling temp path, chmods it to `0600`,
//!    then atomically `rename()`s it into the final `socket_path`.
//!    The two-step bind+chmod sequence is race-free: while the
//!    socket lives at the temp path it is not advertised, and the
//!    mode-0600 invariant is in place before the rename publishes
//!    it. A stale file at the final path is unlinked first.
//! 2. Accept-loops in non-blocking mode, polling the `running` flag
//!    every 100 ms so a Ctrl-C in the parent process makes the
//!    listener exit promptly.
//! 3. For each connection: read one JSON line (capped at
//!    [`MAX_COMMAND_LENGTH`] bytes — oversized lines are rejected
//!    with a structured error and the connection is closed),
//!    dispatch via the `handler` closure, write the JSON response,
//!    loop until the client closes or the daemon shuts down.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use tracing::{debug, info, warn};

use super::commands::{Command, Response};

/// Upper bound on a single command line, in bytes. Lines longer than
/// this are rejected with a structured error and the connection is
/// closed — defends the daemon against a same-user OOM attempt that
/// sends a multi-GiB line.
///
/// 64 KiB is well past any realistic JSON command in this protocol
/// (the largest builtin is a chain swap with a few effect names);
/// the cap exists strictly as a safety valve.
pub const MAX_COMMAND_LENGTH: usize = 64 * 1024;

/// Trait implemented by the worker-side command dispatcher. The
/// listener thread keeps no state — every parsed command goes through
/// `handle`. Implementations send the command to the worker and
/// block on the response (typically via a channel + oneshot).
pub trait CommandHandler: Send + Sync + 'static {
    /// Process one command. Must return a `Response` — the listener
    /// thread cannot drop a parsed command without acknowledging the
    /// client.
    fn handle(&self, cmd: Command) -> Response;
}

/// Owning handle returned by [`spawn`]. Drop signals the listener to
/// stop and removes the socket file (best-effort).
pub struct ListenerHandle {
    running: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
    socket_path: PathBuf,
}

impl ListenerHandle {
    /// Signal the accept loop to stop. The thread joins on `Drop`.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }
}

impl Drop for ListenerHandle {
    fn drop(&mut self) {
        self.stop();
        if let Some(j) = self.join.take() {
            // The thread polls `running` every ~100 ms inside the
            // accept loop (set_nonblocking + sleep), so this join
            // completes promptly under normal shutdown. Unbounded
            // join is safe in practice; a capped join would require
            // a channel-ack or `JoinHandle::is_finished()` polling.
            let _ = j.join();
        }
        // Best-effort unlink. The file may already be gone if the
        // process double-shutdowns.
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

/// Spawn the listener thread. Returns the bound path so the daemon
/// can log it.
///
/// # Errors
///
/// Returns an error if the socket path is unusable (cannot unlink
/// stale file, parent directory missing, permissions, bind fails).
pub fn spawn<H: CommandHandler>(
    socket_path: PathBuf,
    handler: H,
) -> std::io::Result<ListenerHandle> {
    let listener = bind_socket_secure(&socket_path)?;
    listener.set_nonblocking(true)?;
    info!(path = %socket_path.display(), "control socket listening");

    let running = Arc::new(AtomicBool::new(true));
    let running_thread = Arc::clone(&running);
    let handler = Arc::new(handler);
    let path_for_thread = socket_path.clone();
    let join = thread::Builder::new()
        .name("fluxframe-control".into())
        .spawn(move || {
            accept_loop(&listener, &running_thread, &handler, &path_for_thread);
        })?;

    Ok(ListenerHandle {
        running,
        join: Some(join),
        socket_path,
    })
}

/// Race-free bind: bind on a sibling temp path, chmod to mode `0o600`
/// (owner read/write only — denies any other local user from
/// `connect()`-ing), then atomically `rename()` into the final path.
///
/// `UnixListener::bind` honours the process umask, so a naive
/// `bind` + later `set_permissions(0o600)` leaves a window during
/// which another local user could open the socket. The temp+rename
/// dance closes that window without touching the global umask
/// (which would require `unsafe`/libc and a process-wide side
/// effect).
///
/// Any stale socket file at the final path is unlinked first; a
/// non-socket file is left alone (and surfaces as the final
/// `rename()` failing, which is the right outcome).
fn bind_socket_secure(socket_path: &Path) -> std::io::Result<UnixListener> {
    // Unlink stale socket if any. A non-socket file at the path is
    // left alone — surfaces as `rename` error which is the right
    // outcome.
    if let Ok(meta) = std::fs::symlink_metadata(socket_path)
        && meta.file_type().is_socket()
    {
        std::fs::remove_file(socket_path)?;
    }

    let tmp_path = temp_socket_path(socket_path);
    // Best-effort: a previous crash may have left the temp path
    // behind. We do not care if removal fails because of ENOENT.
    let _ = std::fs::remove_file(&tmp_path);

    let listener = UnixListener::bind(&tmp_path)?;
    // 0o600 = owner read/write only; denies other local users from
    // connect()-ing to the control surface.
    let perms = std::fs::Permissions::from_mode(0o600);
    if let Err(e) = std::fs::set_permissions(&tmp_path, perms) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp_path, socket_path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    Ok(listener)
}

/// Build the sibling temp path used by [`bind_socket_secure`].
/// Includes the PID so two daemons crashing on top of each other do
/// not collide on the temp file.
fn temp_socket_path(socket_path: &Path) -> PathBuf {
    let file_name = socket_path.file_name().map_or_else(
        || "fluxframe.sock".to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let temp_name = format!(".{}.{}.tmp", file_name, std::process::id());
    match socket_path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(temp_name),
        _ => PathBuf::from(temp_name),
    }
}

fn accept_loop<H: CommandHandler>(
    listener: &UnixListener,
    running: &Arc<AtomicBool>,
    handler: &Arc<H>,
    path: &Path,
) {
    while running.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((stream, _addr)) => {
                debug!(path = %path.display(), "control client connected");
                handle_connection(stream, handler, running);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No pending connection — sleep briefly to keep the
                // poll responsive without burning CPU.
                thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                warn!(error = %e, "control accept failed");
                thread::sleep(Duration::from_millis(500));
            }
        }
    }
    debug!(path = %path.display(), "control accept loop exited");
}

fn handle_connection<H: CommandHandler>(
    stream: UnixStream,
    handler: &Arc<H>,
    running: &Arc<AtomicBool>,
) {
    // Per-connection I/O timeouts. We read with a finite timeout so a
    // hanging client cannot tie up the listener thread permanently —
    // when the client stalls, we loop back, check `running`, and
    // either retry or exit.
    if let Err(e) = stream.set_read_timeout(Some(Duration::from_millis(500))) {
        warn!(error = %e, "could not set read timeout; dropping connection");
        return;
    }
    // Cap the per-connection read buffer at MAX_COMMAND_LENGTH. We
    // use `read_until(b'\n', _)` (not `lines()`) so we can enforce a
    // hard byte budget — `BufRead::lines()` would happily buffer a
    // multi-GiB line and OOM the daemon.
    let mut reader = BufReader::with_capacity(
        MAX_COMMAND_LENGTH,
        match stream.try_clone() {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "could not clone stream; dropping");
                return;
            }
        },
    );
    let mut writer = stream;
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    while running.load(Ordering::Acquire) {
        buf.clear();
        match read_capped_line(&mut reader, &mut buf, MAX_COMMAND_LENGTH) {
            ReadOutcome::Eof => break,
            ReadOutcome::WouldBlock => {}
            ReadOutcome::Io(e) => {
                warn!(error = %e, "control read failed");
                break;
            }
            ReadOutcome::TooLarge => {
                let response = Response::err(
                    format!("command exceeds {MAX_COMMAND_LENGTH}-byte limit"),
                    Some(
                        "shorten the JSON payload; the control protocol caps a single command line"
                            .into(),
                    ),
                );
                let _ = write_response(&mut writer, &response);
                warn!(
                    limit = MAX_COMMAND_LENGTH,
                    "control command exceeded byte limit; closing connection"
                );
                break;
            }
            ReadOutcome::Ok => {
                let line = std::str::from_utf8(&buf).map(str::trim).unwrap_or("");
                if line.is_empty() {
                    continue;
                }
                let response = match serde_json::from_str::<Command>(line) {
                    Ok(cmd) => handler.handle(cmd),
                    Err(e) => Response::err(
                        format!("invalid command: {e}"),
                        Some("expected line-delimited JSON; see README".into()),
                    ),
                };
                if let Err(e) = write_response(&mut writer, &response) {
                    warn!(error = %e, "control write failed; closing");
                    break;
                }
            }
        }
    }
}

/// Outcome of one capped read attempt.
enum ReadOutcome {
    /// A complete line was read (with or without the trailing `\n`).
    Ok,
    /// The peer closed the connection cleanly.
    Eof,
    /// The read timed out and the caller should re-poll `running`.
    WouldBlock,
    /// The line exceeded the byte budget before a newline was seen.
    TooLarge,
    /// Any other I/O error.
    Io(std::io::Error),
}

/// Read up to one newline-terminated line into `buf`, refusing to
/// grow `buf` past `limit` bytes. Returns [`ReadOutcome::TooLarge`]
/// when the limit is hit before a `\n` arrives — callers are
/// expected to close the connection in that case.
fn read_capped_line<R: BufRead>(reader: &mut R, buf: &mut Vec<u8>, limit: usize) -> ReadOutcome {
    loop {
        let available = match reader.fill_buf() {
            Ok(slice) => slice,
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                return ReadOutcome::WouldBlock;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return ReadOutcome::Io(e),
        };
        if available.is_empty() {
            return if buf.is_empty() {
                ReadOutcome::Eof
            } else {
                // Final line without trailing newline — treat as a
                // complete command.
                ReadOutcome::Ok
            };
        }
        let (consume, done) = match available.iter().position(|&b| b == b'\n') {
            Some(idx) => (idx + 1, true),
            None => (available.len(), false),
        };
        // Refuse to grow past `limit`. This is the OOM guard.
        if buf.len().saturating_add(consume) > limit {
            // Drain enough to drop the offender from the BufReader
            // window, then bail.
            let take = limit.saturating_sub(buf.len()).min(consume);
            buf.extend_from_slice(&available[..take]);
            reader.consume(take);
            return ReadOutcome::TooLarge;
        }
        buf.extend_from_slice(&available[..consume]);
        reader.consume(consume);
        if done {
            return ReadOutcome::Ok;
        }
    }
}

/// Serialise a [`Response`] and write it as a single line on `writer`.
/// Returns the underlying I/O error from the write, never from the
/// serialisation (which is logged and turned into an empty success).
fn write_response<W: Write>(writer: &mut W, response: &Response) -> std::io::Result<()> {
    let payload = match serde_json::to_string(response) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "response serialise failed");
            return Ok(());
        }
    };
    writeln!(writer, "{payload}")
}

/// Derive the default socket path from an `XDG_RUNTIME_DIR` value
/// (typically read from the environment). When `xdg` is `None` or
/// empty, falls back to `/tmp/fluxframe.sock`.
///
/// The split between this pure helper and the env-reading wrapper
/// [`default_socket_path`] keeps the workspace's `forbid(unsafe_code)`
/// invariant intact — env mutation in tests would require unsafe.
#[must_use]
pub fn resolve_socket_path(xdg: Option<&str>) -> PathBuf {
    match xdg {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir).join("fluxframe.sock"),
        _ => PathBuf::from("/tmp/fluxframe.sock"),
    }
}

/// Resolve the default socket path from `$XDG_RUNTIME_DIR`.
#[must_use]
pub fn default_socket_path() -> PathBuf {
    resolve_socket_path(std::env::var("XDG_RUNTIME_DIR").ok().as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream as ClientStream;
    use std::sync::atomic::AtomicUsize;

    /// Build a unique-per-test path under the system temp dir so
    /// parallel `cargo test` runs do not race on the same file name.
    fn unique_socket_path(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "fluxframe-test-{tag}-{}-{n}.sock",
            std::process::id()
        ))
    }

    /// Handler that immediately echoes Ok for any command — enough
    /// for the socket-level wiring tests.
    struct EchoHandler;
    impl CommandHandler for EchoHandler {
        fn handle(&self, _cmd: Command) -> Response {
            Response::ok()
        }
    }

    #[test]
    fn resolve_uses_xdg_when_set() {
        assert_eq!(
            resolve_socket_path(Some("/run/user/1000")),
            PathBuf::from("/run/user/1000/fluxframe.sock")
        );
    }

    #[test]
    fn resolve_falls_back_when_xdg_empty() {
        assert_eq!(
            resolve_socket_path(Some("")),
            PathBuf::from("/tmp/fluxframe.sock")
        );
        assert_eq!(
            resolve_socket_path(None),
            PathBuf::from("/tmp/fluxframe.sock")
        );
    }

    #[test]
    fn temp_socket_path_uses_sibling_directory() {
        let p = temp_socket_path(Path::new("/run/user/1000/fluxframe.sock"));
        assert_eq!(p.parent(), Some(Path::new("/run/user/1000")));
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with(".fluxframe.sock."));
        assert!(
            std::path::Path::new(&name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("tmp"))
        );
        assert!(name.contains(&std::process::id().to_string()));
    }

    #[test]
    fn temp_socket_path_handles_bare_filename() {
        let p = temp_socket_path(Path::new("fluxframe.sock"));
        // No parent → temp path lives in the current dir, which is
        // fine for the rename (same directory).
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with(".fluxframe.sock."));
    }

    #[test]
    fn bind_rename_leaves_socket_at_target_with_mode_0600() {
        let path = unique_socket_path("bind");
        let handle = spawn(path.clone(), EchoHandler).expect("spawn");

        // Socket file is present at the target path…
        let meta = std::fs::symlink_metadata(&path).expect("socket exists");
        assert!(meta.file_type().is_socket(), "expected UNIX socket");
        // …and has owner-only mode bits.
        let mode = meta.permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "expected mode 0o600, got {mode:o}");

        // No stray temp file left behind.
        let tmp = temp_socket_path(&path);
        assert!(
            !tmp.exists(),
            "temp path should have been renamed away: {tmp:?}"
        );

        drop(handle);
        // The drop unlinks the socket file (best-effort).
        std::thread::sleep(Duration::from_millis(150));
        assert!(!path.exists(), "socket should be removed on Drop");
    }

    #[test]
    fn oversized_line_is_rejected_and_connection_closed() {
        let path = unique_socket_path("oversize");
        let handle = spawn(path.clone(), EchoHandler).expect("spawn");

        let mut client = ClientStream::connect(&path).expect("connect");
        // Send a payload comfortably larger than the limit, no
        // trailing newline (the limit must trip before we even reach
        // a newline).
        let big = vec![b'A'; MAX_COMMAND_LENGTH + 16];
        client.write_all(&big).expect("write payload");
        // Half-close to make sure the server stops waiting for more
        // bytes from us.
        client
            .shutdown(std::net::Shutdown::Write)
            .expect("shutdown write");

        // Read whatever the server sends back. Expect a structured
        // error response containing the limit text, followed by EOF
        // (the server must close the connection after rejecting).
        let mut response = String::new();
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("read timeout");
        client.read_to_string(&mut response).expect("read response");
        assert!(
            response.contains("exceeds"),
            "expected error message about limit, got: {response:?}"
        );
        assert!(
            response.contains(&MAX_COMMAND_LENGTH.to_string()),
            "expected error to mention the byte limit, got: {response:?}"
        );

        drop(handle);
    }

    #[test]
    fn small_command_round_trips() {
        let path = unique_socket_path("rt");
        let handle = spawn(path.clone(), EchoHandler).expect("spawn");

        let mut client = ClientStream::connect(&path).expect("connect");
        client
            .write_all(b"{\"cmd\":\"list_presets\"}\n")
            .expect("write");

        let mut buf = [0u8; 256];
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("read timeout");
        let n = client.read(&mut buf).expect("read");
        let response = std::str::from_utf8(&buf[..n]).unwrap();
        assert!(response.contains("\"ok\":\"true\""), "got: {response}");

        drop(handle);
    }

    #[test]
    fn read_capped_line_reads_short_line() {
        let mut data = std::io::Cursor::new(b"hello\nworld\n".to_vec());
        let mut buf = Vec::new();
        match read_capped_line(&mut data, &mut buf, 64) {
            ReadOutcome::Ok => {}
            other => panic!("expected Ok, got {:?}", std::mem::discriminant(&other)),
        }
        assert_eq!(&buf, b"hello\n");
    }

    #[test]
    fn read_capped_line_rejects_oversize() {
        let payload = vec![b'X'; 1024];
        let mut data = std::io::Cursor::new(payload);
        let mut buf = Vec::new();
        match read_capped_line(&mut data, &mut buf, 128) {
            ReadOutcome::TooLarge => {}
            other => panic!(
                "expected TooLarge, got discriminant {:?}",
                std::mem::discriminant(&other)
            ),
        }
        // Buf is bounded by the limit.
        assert!(buf.len() <= 128, "buf grew past limit: {}", buf.len());
    }

    #[test]
    fn read_capped_line_eof_on_empty() {
        let mut data = std::io::Cursor::new(Vec::<u8>::new());
        let mut buf = Vec::new();
        match read_capped_line(&mut data, &mut buf, 64) {
            ReadOutcome::Eof => {}
            other => panic!(
                "expected Eof, got discriminant {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn read_capped_line_accepts_final_line_without_newline() {
        let mut data = std::io::Cursor::new(b"trailing".to_vec());
        let mut buf = Vec::new();
        match read_capped_line(&mut data, &mut buf, 64) {
            ReadOutcome::Ok => {}
            other => panic!(
                "expected Ok, got discriminant {:?}",
                std::mem::discriminant(&other)
            ),
        }
        assert_eq!(&buf, b"trailing");
    }
}
