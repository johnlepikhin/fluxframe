//! Merge a file-based `FluxConfig` with CLI overrides.
//!
//! Single source of truth for the §11.3 rule: "CLI options must override
//! values from the config file."  Keeping this in its own module makes it
//! straightforward to unit-test without spinning up clap.

use std::io::Read;
use std::path::{Path, PathBuf};

use fluxframe_core::{FluxConfig, FluxError, InputDevice};

/// Subset of CLI-overridable values shared between `run` and `benchmark`.
///
/// Preset-specific knobs (`--preset NAME`, per-effect parameters,
/// model paths) live in the TOML config now — the CLI only steers
/// device routing and capture dimensions.
#[derive(Debug, Default, Clone)]
pub struct CliOverrides {
    /// Override input device path or logical name.
    pub input: Option<String>,
    /// Override output device path or logical name.
    pub output: Option<String>,
    /// Override frame width in pixels.
    pub width: Option<u32>,
    /// Override frame height in pixels.
    pub height: Option<u32>,
    /// Override frame rate in frames per second.
    pub fps: Option<u32>,
}

/// Parse a CLI string into a typed [`InputDevice`].  Mirrors the TOML
/// adapter in `fluxframe_core::config`: `"auto"` and `"testsrc"` are
/// case-insensitive sentinels, anything else is taken as a literal
/// device path.  Kept inline here so the CLI override path applies the
/// same vocabulary as the file loader without going through serde.
fn parse_input_device(s: &str) -> InputDevice {
    if s.eq_ignore_ascii_case("auto") {
        InputDevice::Auto
    } else if s.eq_ignore_ascii_case("testsrc") {
        InputDevice::Testsrc
    } else {
        InputDevice::Path(PathBuf::from(s))
    }
}

/// Hard upper bound on the on-disk size of a config file.
///
/// Guards against accidentally pointing `--config` at `/dev/zero`,
/// `/proc/self/mem`, or a multi-gigabyte log file.  Real FluxFrame
/// configs are typically a few kilobytes.
const MAX_CONFIG_BYTES: u64 = 1 << 20; // 1 MiB hard ceiling.

/// Default sub-path under `$XDG_CONFIG_HOME` (or `$HOME/.config`)
/// where the daemon looks for an operator-managed config when
/// `--config` is omitted. Same path the GUI's Save button writes to
/// for the first time, so the resolver answer is symmetric on read
/// and write.
const XDG_CONFIG_SUBPATH: &[&str] = &["fluxframe", "fluxframe.toml"];

/// Compute the default config path under XDG conventions. The path
/// is purely derived from environment variables — no filesystem
/// access happens here, so the same answer is safe to use for both
/// "where would I read from?" and "where would I write to?".
fn default_xdg_path() -> PathBuf {
    let base = if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|s| !s.is_empty()) {
        PathBuf::from(xdg)
    } else {
        let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
        let mut p = PathBuf::from(home);
        p.push(".config");
        p
    };
    let mut p = base;
    for component in XDG_CONFIG_SUBPATH {
        p.push(component);
    }
    p
}

/// Resolve the path the daemon will use as the source of truth for
/// Save / Reload. Returns `None` only when the operator explicitly
/// opted out of default-path lookup (`--no-default-config`) AND did
/// not provide `--config`. The returned path is **not** required to
/// exist on disk — the GUI's first Save bootstraps it.
#[must_use]
pub fn resolve_writable_path(explicit: Option<&Path>, no_default: bool) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p.to_path_buf());
    }
    if no_default {
        return None;
    }
    Some(default_xdg_path())
}

/// Resolve the path to open at daemon startup. Differs from
/// [`resolve_writable_path`] only in returning `None` when the
/// resolved default path is not yet on disk — fresh installs should
/// boot from built-in defaults without surfacing an `Io` error.
/// Explicit `--config` paths still error out if the file is missing
/// (handled by [`load`]); the silent-fallback is XDG-default-only.
#[must_use]
pub fn resolve_load_path(explicit: Option<&Path>, no_default: bool) -> Option<PathBuf> {
    if explicit.is_some() {
        return explicit.map(Path::to_path_buf);
    }
    if no_default {
        return None;
    }
    let p = default_xdg_path();
    if p.exists() { Some(p) } else { None }
}

/// Load a config file if present, otherwise start from defaults.
///
/// The file is opened once and its `metadata` is fetched from the same
/// `File` handle (not the path) to avoid a TOCTOU window between the
/// size check and the read.  The reader is then capped at
/// [`MAX_CONFIG_BYTES`] so a file that grows between `metadata()` and
/// `read_to_string` still cannot OOM the CLI.
///
/// # Errors
///
/// Returns [`FluxError::Io`] if the file cannot be opened, its metadata
/// cannot be read, or reading its contents fails.  Returns
/// [`FluxError::Config`] when the target is not a regular file or
/// exceeds [`MAX_CONFIG_BYTES`].  Returns [`FluxError::ConfigParse`]
/// when the TOML body is syntactically invalid.
pub fn load(path: Option<&Path>) -> Result<FluxConfig, FluxError> {
    match path {
        Some(p) => {
            let file = std::fs::File::open(p).map_err(|e| FluxError::io(p.to_path_buf(), e))?;
            let meta = file
                .metadata()
                .map_err(|e| FluxError::io(p.to_path_buf(), e))?;
            if !meta.is_file() {
                return Err(FluxError::Config {
                    reason: format!("{} is not a regular file", p.display()),
                    hint: Some("pass a TOML file path, not a directory or device node".into()),
                });
            }
            if meta.len() > MAX_CONFIG_BYTES {
                return Err(FluxError::Config {
                    reason: format!(
                        "config file {} exceeds {MAX_CONFIG_BYTES}-byte limit ({} bytes)",
                        p.display(),
                        meta.len()
                    ),
                    hint: Some("FluxFrame config files are expected to be a few KiB".into()),
                });
            }
            let mut text = String::with_capacity(meta.len() as usize);
            file.take(MAX_CONFIG_BYTES)
                .read_to_string(&mut text)
                .map_err(|e| FluxError::io(p.to_path_buf(), e))?;
            FluxConfig::from_toml_str(&text)
        }
        None => Ok(FluxConfig::default()),
    }
}

/// Apply CLI overrides on top of a `FluxConfig`.
///
/// Effects, models, and preset selection are NOT routed through this
/// merger — the preset is named via `--preset` and resolved against
/// `cfg.presets` directly in `commands::run`.
///
/// # Errors
///
/// This function is infallible and does not validate the merged config;
/// call [`FluxConfig::validate`] afterwards to enforce structural rules.
#[must_use]
pub fn apply(mut cfg: FluxConfig, overrides: &CliOverrides) -> FluxConfig {
    if let Some(input) = &overrides.input {
        cfg.input.device = parse_input_device(input);
    }
    if let Some(output) = &overrides.output {
        cfg.output.device.clone_from(output);
    }
    // Output dimensions and fps are derived from the input by
    // construction (see `OutputConfig` doc), so CLI `--width` /
    // `--height` / `--fps` only steer the input — the supervisor
    // computes the output from `output.scale` and `input.fps`.
    if let Some(width) = overrides.width {
        cfg.input.width = width;
    }
    if let Some(height) = overrides.height {
        cfg.input.height = height;
    }
    if let Some(fps) = overrides.fps {
        cfg.input.fps = fps;
    }
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_input_overrides_file_value() {
        let file_cfg = FluxConfig::from_toml_str(
            r#"
[input]
device = "/dev/video2"
width = 640
height = 480
fps = 15
"#,
        )
        .expect("toml parses");

        let merged = apply(
            file_cfg,
            &CliOverrides {
                input: Some("/dev/video9".into()),
                width: Some(1920),
                ..Default::default()
            },
        );

        assert_eq!(
            merged.input.device,
            InputDevice::Path(PathBuf::from("/dev/video9"))
        );
        assert_eq!(merged.input.width, 1920);
        assert_eq!(merged.input.height, 480, "non-overridden field stays");
        assert_eq!(merged.input.fps, 15);
    }

    #[test]
    fn cli_input_override_parses_auto_and_testsrc_sentinels() {
        for (raw, expected) in [
            ("auto", InputDevice::Auto),
            ("AUTO", InputDevice::Auto),
            ("testsrc", InputDevice::Testsrc),
            ("TestSrc", InputDevice::Testsrc),
        ] {
            let merged = apply(
                FluxConfig::default(),
                &CliOverrides {
                    input: Some(raw.to_string()),
                    ..Default::default()
                },
            );
            assert_eq!(merged.input.device, expected, "raw={raw}");
        }
        let merged = apply(
            FluxConfig::default(),
            &CliOverrides {
                input: Some("/dev/video7".into()),
                ..Default::default()
            },
        );
        assert_eq!(
            merged.input.device,
            InputDevice::Path(PathBuf::from("/dev/video7"))
        );
    }

    #[test]
    fn missing_file_path_falls_back_to_defaults() {
        let cfg = load(None).expect("defaults always load");
        assert_eq!(cfg.input.width, 1280);
        assert_eq!(cfg.input.height, 720);
        assert_eq!(cfg.input.fps, 30);
    }

    #[test]
    fn explicit_config_path_passes_through_resolver() {
        let p = PathBuf::from("/tmp/explicit.toml");
        assert_eq!(
            resolve_writable_path(Some(p.as_path()), false),
            Some(p.clone())
        );
        assert_eq!(
            resolve_writable_path(Some(p.as_path()), true),
            Some(p.clone()),
            "explicit --config wins over --no-default-config"
        );
        assert_eq!(resolve_load_path(Some(p.as_path()), true), Some(p));
    }

    #[test]
    fn no_default_returns_none_when_no_explicit() {
        assert_eq!(resolve_writable_path(None, true), None);
        assert_eq!(resolve_load_path(None, true), None);
    }

    #[test]
    fn default_xdg_writable_path_is_returned_even_when_missing() {
        // resolve_writable_path is environment-derived; we just check
        // the shape rather than asserting a specific path (the test
        // host's $XDG_CONFIG_HOME / $HOME drives the answer).
        let path = resolve_writable_path(None, false).expect("path resolvable");
        assert!(
            path.ends_with("fluxframe/fluxframe.toml"),
            "unexpected default writable path: {}",
            path.display()
        );
    }
}
