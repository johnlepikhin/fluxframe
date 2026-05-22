//! `clap` definitions for the `fluxframe` binary.
//!
//! The CLI surface mirrors §§9-12 of the spec.  Stage 0 only wires the
//! argument structures and dispatch — the actual commands print a
//! `not implemented yet` notice and exit 0.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Default benchmark duration in seconds when `--duration` is omitted.
pub const DEFAULT_BENCHMARK_SECONDS: u32 = 30;

/// Top-level `fluxframe` command-line interface.
#[derive(Debug, Parser)]
#[command(
    name = "fluxframe",
    version,
    about = "Realtime video processing layer for Linux (Stage 0 scaffolding).",
    long_about = None,
)]
pub struct Cli {
    /// Increase log verbosity (repeatable: -v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// The subcommand to execute.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level subcommands exposed by the `fluxframe` binary.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// List available input and output video devices.
    List(ListArgs),
    /// Verify that GStreamer, devices and the configured effect chain are usable.
    Check(CheckArgs),
    /// Run the realtime processing pipeline.
    Run(RunArgs),
    /// Benchmark capture, processing and output stages.
    Benchmark(BenchmarkArgs),
}

/// Arguments accepted by `fluxframe list`.
#[derive(Debug, Clone, clap::Args)]
pub struct ListArgs {
    /// Optional config file (used to resolve default backends).
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,
}

/// Subset of arguments shared by `check`, `run` and `benchmark`.
///
/// Kept in one struct so that adding a new override (e.g. `--profile`) lands
/// in every consumer atomically.  Subcommand structs use
/// `#[command(flatten)] pub common: CommonRunArgs;` to expose the flags
/// without duplicating each `#[arg]` declaration.
#[derive(Debug, Clone, clap::Args)]
pub struct CommonRunArgs {
    /// Configuration file (TOML).
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,
    /// Override input device.
    #[arg(long, value_name = "PATH_OR_NAME")]
    pub input: Option<String>,
    /// Override output device.
    #[arg(long, value_name = "PATH_OR_NAME")]
    pub output: Option<String>,
    /// Effect to apply (kebab-case, e.g. `background-blur`).
    #[arg(long, value_name = "NAME")]
    pub effect: Option<String>,
    /// Model file to load.
    #[arg(long, value_name = "PATH")]
    pub model: Option<PathBuf>,
}

/// Arguments accepted by `fluxframe check`.
#[derive(Debug, Clone, clap::Args)]
pub struct CheckArgs {
    /// Arguments shared with `run`/`benchmark`.
    #[command(flatten)]
    pub common: CommonRunArgs,
}

/// Arguments accepted by `fluxframe run`.
#[derive(Debug, Clone, clap::Args)]
pub struct RunArgs {
    /// Arguments shared with `check`/`benchmark`.
    #[command(flatten)]
    pub common: CommonRunArgs,
    /// Override frame width in pixels.
    #[arg(long, value_name = "PIXELS")]
    pub width: Option<u32>,
    /// Override frame height in pixels.
    #[arg(long, value_name = "PIXELS")]
    pub height: Option<u32>,
    /// Override frame rate in frames per second.
    #[arg(long, value_name = "RATE")]
    pub fps: Option<u32>,
}

/// Arguments accepted by `fluxframe benchmark`.
#[derive(Debug, Clone, clap::Args)]
pub struct BenchmarkArgs {
    /// Arguments shared with `check`/`run`.
    #[command(flatten)]
    pub common: CommonRunArgs,
    /// Benchmark duration in seconds.
    #[arg(long, value_name = "SECONDS", default_value_t = DEFAULT_BENCHMARK_SECONDS)]
    pub duration: u32,
}
