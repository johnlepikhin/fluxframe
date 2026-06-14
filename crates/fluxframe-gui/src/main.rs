//! Entry point for `fluxframe-gui`.
//!
//! Parses CLI arguments, initialises tracing, and hands control to
//! the relm4 application loop.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod app;
mod debounce;
mod ipc;
mod persistence;
mod shortcuts;
mod state;
mod components {
    pub mod chain_page;
    pub mod param_row;
    pub mod preset_bar;
    pub mod status_page;
}

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

    let args = Args::parse();
    let socket_path = args.socket.unwrap_or_else(default_socket_path);
    tracing::info!(path = %socket_path.display(), "starting fluxframe-gui");

    let app = RelmApp::new("io.fluxframe.gui");
    app.run::<AppModel>(socket_path);
}
