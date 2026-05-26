//! Merge a file-based `FluxConfig` with CLI overrides.
//!
//! Single source of truth for the §11.3 rule: "CLI options must override
//! values from the config file."  Keeping this in its own module makes it
//! straightforward to unit-test without spinning up clap.

use std::io::Read;
use std::path::{Path, PathBuf};

use fluxframe_core::{FluxConfig, FluxError, InputDevice, PipelineSection, normalise_effect_name};

/// Subset of CLI-overridable values shared between `run` and `benchmark`.
#[derive(Debug, Default, Clone)]
pub struct CliOverrides {
    /// Override input device path or logical name.
    pub input: Option<String>,
    /// Override output device path or logical name.
    pub output: Option<String>,
    /// Override the effect chain with a single effect (kebab-case name).
    pub effect: Option<String>,
    /// Override frame width in pixels.
    pub width: Option<u32>,
    /// Override frame height in pixels.
    pub height: Option<u32>,
    /// Override frame rate in frames per second.
    pub fps: Option<u32>,
    /// Override the ONNX model path used by the segmentation stage of
    /// the composite pipeline. Injected into the merged config's
    /// `[mask].model` field so the builder picks it up. When the file
    /// config has no `[mask]` section yet, a minimal one is synthesised
    /// so the override still takes effect.
    pub model: Option<PathBuf>,
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

/// Apply CLI overrides on top of a `FluxConfig`.  Effects supplied via
/// `--effect` replace the entire chain (matching §11.1/11.2 usage) and
/// are normalised from kebab-case to snake_case for registry lookup.
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
    if let Some(effect) = &overrides.effect {
        cfg.effects.chain = vec![normalise_effect_name(effect)];
    }
    // `--model PATH` re-targets the segmentation model used by the
    // composite pipeline. It writes through to `[mask].model`; if the
    // file config has no `[mask]` section we synthesise a minimal one
    // so the override is not silently dropped. The composite builder
    // still requires the operator to supply the rest of the pipeline
    // (chain, background/foreground sections); `--model` only swaps
    // the model path.
    if let Some(model_path) = &overrides.model {
        match cfg.mask.as_mut() {
            Some(section) => section.model = Some(model_path.clone()),
            None => {
                cfg.mask = Some(PipelineSection {
                    model: Some(model_path.clone()),
                    ..PipelineSection::default()
                });
            }
        }
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
    fn cli_effect_replaces_chain_and_normalises_name() {
        let file_cfg = FluxConfig::from_toml_str(
            r#"
[effects]
chain = ["color_adjust", "overlay"]
"#,
        )
        .expect("toml parses");

        let merged = apply(
            file_cfg,
            &CliOverrides {
                effect: Some("background-blur".into()),
                ..Default::default()
            },
        );

        assert_eq!(merged.effects.chain, vec!["background_blur".to_string()]);
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
    fn cli_model_overrides_existing_mask_model() {
        // Pre-existing `[mask]` section: `--model` must replace the
        // model path while preserving the rest of the section
        // (chain, sub-tables, fallback_threshold, …).
        let file_cfg = FluxConfig::from_toml_str(
            r#"
[mask]
model = "/etc/wrong.onnx"
chain = ["threshold"]
fallback_threshold = 5

[mask.threshold]
level = 0.6
"#,
        )
        .expect("toml parses");

        let merged = apply(
            file_cfg,
            &CliOverrides {
                model: Some(PathBuf::from("/tmp/seg.onnx")),
                ..Default::default()
            },
        );

        let mask = merged.mask.expect("mask section survives");
        assert_eq!(
            mask.model.as_deref(),
            Some(std::path::Path::new("/tmp/seg.onnx")),
            "--model replaced the previous model path",
        );
        assert_eq!(
            mask.chain,
            vec!["threshold".to_string()],
            "chain is preserved",
        );
        assert_eq!(
            mask.fallback_threshold,
            Some(5),
            "fallback_threshold is preserved",
        );
        let threshold = mask
            .per_effect
            .get("threshold")
            .and_then(toml::Value::as_table)
            .expect("threshold sub-table preserved");
        assert!(
            (threshold
                .get("level")
                .and_then(toml::Value::as_float)
                .unwrap()
                - 0.6)
                .abs()
                < 1e-6,
            "threshold knobs survive the override",
        );
    }

    #[test]
    fn cli_model_synthesises_mask_section_when_absent() {
        // Without a file-level `[mask]` section the CLI override
        // creates a minimal one so the model path is not silently
        // dropped. The synthesised section is otherwise empty — the
        // operator still owns the chain/background/foreground wiring.
        let cfg = FluxConfig::default();
        let overrides = CliOverrides {
            model: Some(PathBuf::from("/tmp/seg.onnx")),
            ..Default::default()
        };
        let merged = apply(cfg, &overrides);
        let mask = merged
            .mask
            .expect("mask section synthesised by --model override");
        assert_eq!(
            mask.model.as_deref(),
            Some(std::path::Path::new("/tmp/seg.onnx")),
        );
        assert!(
            mask.chain.is_empty(),
            "synthesised section has empty chain (operator must populate)",
        );
    }
}
