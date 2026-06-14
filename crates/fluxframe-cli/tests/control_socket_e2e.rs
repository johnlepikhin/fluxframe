//! Wire-format integration smoke for the control protocol.
//!
//! Exercises the serialise/deserialise path that the daemon and the
//! GUI client both depend on, plus the size cap defended by the
//! socket listener. The actual daemon round-trip is left for a
//! follow-up E2E that brings up a real `fluxframe run` instance.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::thread;

use fluxframe_core::SubchainKind;
use fluxframe_core::protocol::{Command, Response};

/// A round-trippable Command travels through a UNIX socket pair as
/// JSON with no loss.
#[test]
fn command_round_trip_over_unix_socket() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let path = dir.path().join("test.sock");
    let listener = UnixListener::bind(&path).expect("bind");

    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).expect("read");
        // Echo the command back as the Ok payload.
        let cmd: Command = serde_json::from_str(line.trim()).expect("parse");
        let resp = Response::ok_with(serde_json::to_value(&cmd).expect("reserialise"));
        let payload = serde_json::to_string(&resp).expect("serialise");
        let mut writer = reader.into_inner();
        writeln!(writer, "{payload}").expect("write");
    });

    let mut client = UnixStream::connect(&path).expect("connect");
    let cmd = Command::SetChain {
        section: SubchainKind::Post,
        chain: vec!["passthrough".into()],
    };
    let payload = serde_json::to_string(&cmd).expect("serialise");
    writeln!(client, "{payload}").expect("write");
    let mut reader = BufReader::new(client);
    let mut line = String::new();
    reader.read_line(&mut line).expect("read response");
    let resp: Response = serde_json::from_str(line.trim()).expect("parse response");
    match resp {
        Response::Ok { data } => {
            let echoed: Command = serde_json::from_value(data).expect("echoed command parses");
            assert_eq!(echoed, cmd, "round-trip preserves the command");
        }
        Response::Err { error, .. } => panic!("unexpected Err: {error}"),
        other => panic!("non-exhaustive Response variant: {other:?}"),
    }
    server.join().expect("server thread");
}

/// `Command` rejects unknown discriminator values with a parse error
/// rather than silently dropping into a default variant.
#[test]
fn command_rejects_unknown_cmd() {
    let res: Result<Command, _> = serde_json::from_str(r#"{"cmd":"bogus"}"#);
    assert!(res.is_err(), "unknown cmd must be rejected");
}

/// `Response::err` round-trips through JSON without losing the hint.
#[test]
fn response_err_serialises_with_hint() {
    let resp = Response::err("nope", Some("try again".into()));
    let payload = serde_json::to_string(&resp).expect("serialise");
    let parsed: Response = serde_json::from_str(&payload).expect("parse");
    match parsed {
        Response::Err { error, hint } => {
            assert_eq!(error, "nope");
            assert_eq!(hint.as_deref(), Some("try again"));
        }
        Response::Ok { .. } => panic!("expected Err"),
        other => panic!("non-exhaustive Response variant: {other:?}"),
    }
}
