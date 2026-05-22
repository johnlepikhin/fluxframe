//! Subcommand dispatcher.

use fluxframe_core::FluxError;

use crate::cli::{Cli, Command};

pub mod benchmark;
pub mod check;
pub mod list;
pub mod run;

/// Entry point for command dispatch.
///
/// Matches the parsed [`Cli`] onto the corresponding subcommand handler.
///
/// # Errors
///
/// Propagates any [`FluxError`] returned by the selected subcommand.
pub fn dispatch(cli: Cli) -> Result<(), FluxError> {
    match cli.command {
        Command::List(args) => list::run(args),
        Command::Check(args) => check::run(args),
        Command::Run(args) => run::run(args),
        Command::Benchmark(args) => benchmark::run(args),
    }
}
