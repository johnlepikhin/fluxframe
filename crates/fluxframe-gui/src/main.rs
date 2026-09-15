//! Entry point for `fluxframe-gui`.
//!
//! Parses CLI arguments, initialises tracing, and hands control to
//! the relm4 application loop.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod app;
mod components;
mod debounce;
mod ipc;
mod persistence;
mod reconnect;
mod reply;
mod shortcuts;
mod state;

use std::path::PathBuf;

use clap::Parser;
use fluxframe_core::protocol::default_socket_path;
use relm4::RelmApp;
use tracing_subscriber::EnvFilter;

use crate::app::AppModel;

/// `fluxframe-gui` — GTK4 control panel for the FluxFrame daemon.
///
/// Connects to the daemon over a UNIX socket and offers a slider /
/// drop-down editor for the active preset's effect chain.
#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Override the daemon control socket path. Defaults to
    /// `$XDG_RUNTIME_DIR/fluxframe.sock` (or `/tmp/fluxframe.sock`
    /// when XDG_RUNTIME_DIR is not set).
    #[arg(long, value_name = "PATH")]
    socket: Option<PathBuf>,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("info".parse().expect("static 'info' directive parses")),
        )
        .init();

    // GStreamer is used by the embedded preview pane (see
    // crates/fluxframe-gui/src/components/preview/). Failure is logged
    // but does not abort: the outcome is handed to the preview, which
    // then stays an inert placeholder without touching any gst API,
    // and the rest of the GUI keeps working.
    let gst_ready = match gstreamer::init() {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(error = %e, "gstreamer::init failed; embedded preview will be inert");
            false
        }
    };

    let args = Args::parse();
    let socket_path = args.socket.unwrap_or_else(default_socket_path);
    tracing::info!(path = %socket_path.display(), "starting fluxframe-gui");

    // clap already consumed argv; without `with_args` relm4 forwards
    // the process arguments to GApplication, which rejects `--socket`.
    let app = RelmApp::new("io.fluxframe.gui").with_args(Vec::new());
    app.run::<AppModel>(app::AppInit {
        socket_path,
        gst_ready,
    });
}
