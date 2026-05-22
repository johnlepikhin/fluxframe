//! Subcommand dispatcher.

use fluxframe_core::FluxError;

use crate::cli::{Cli, Command};

// `benchmark` is an inference-only path (Stage 3 scope); the whole
// subcommand depends on `fluxframe_effects::ml`, so we gate the module
// itself behind the `ml` feature.  Without `ml` the dispatcher returns
// a structured §27 error instead of compiling out the CLI surface
// (clap arguments stay stable across feature configurations).
#[cfg(feature = "ml")]
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
        #[cfg(feature = "ml")]
        Command::Benchmark(args) => benchmark::run(args),
        #[cfg(not(feature = "ml"))]
        Command::Benchmark(_) => Err(FluxError::Config {
            reason: "this build was compiled without the `ml` feature".into(),
            hint: Some(
                "rebuild with `cargo build -p fluxframe-cli --features ml` \
                 to enable the inference benchmark"
                    .into(),
            ),
        }),
    }
}
