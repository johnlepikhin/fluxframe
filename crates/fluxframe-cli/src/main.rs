//! `fluxframe` CLI entry point.
//!
//! Parses arguments, initialises tracing, dispatches to a subcommand and
//! renders the §27 "Error / Hint" diagnostic format on stderr when a
//! command fails.

#![warn(missing_docs)]

use std::process::ExitCode;

use clap::Parser;
use tracing::error;

mod cli;
mod commands;
mod config_merge;
mod logging;

fn main() -> ExitCode {
    let args = cli::Cli::parse();

    if let Err(e) = logging::init(args.verbose) {
        eprintln!("failed to initialise logging: {e}");
        return ExitCode::from(2);
    }

    match commands::dispatch(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            use fluxframe_core::Diagnostic;
            // Print §27 canonical format on stderr independent of RUST_LOG.
            eprintln!("Error: {}", e.reason());
            if let Some(hint) = e.hint() {
                eprintln!("Hint: {hint}");
            }
            error!(error = %e, "command failed");
            ExitCode::FAILURE
        }
    }
}
