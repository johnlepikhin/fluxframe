//! `fluxframe check` — verify GStreamer init, devices and effect chain.
//!
//! Stage 2 implementation: pre-flight checks before `fluxframe run` so
//! users see structured §27 Error/Hint diagnostics instead of opaque
//! GStreamer failures.  Required-elements introspection lands in
//! Stage 5 (needs a helper in `fluxframe-gst`).

use std::path::Path;

use fluxframe_core::error::PipelineError;
use fluxframe_core::{Diagnostic, FluxConfig, FluxError, normalise_effect_name};
use fluxframe_gst::{V4l2DeviceKind, enumerate_devices};
use tracing::{info, warn};

use crate::cli::CheckArgs;
use crate::config_merge::{CliOverrides, apply, load};
use crate::runtime::default_registry;

/// Entry point for `fluxframe check`.
///
/// # Errors
///
/// Returns the first failed check as a [`FluxError`].  All checks run
/// (subsequent failures are also printed) before returning.
pub fn run(args: CheckArgs) -> Result<(), FluxError> {
    let overrides = CliOverrides {
        input: args.common.input,
        output: args.common.output,
        effect: args.common.effect,
        width: None,
        height: None,
        fps: None,
    };
    let cfg = load(args.common.config.as_deref())?;
    let cfg = apply(cfg, &overrides);
    cfg.validate()?;

    let mut failures: Vec<FluxError> = Vec::new();

    if let Err(e) = check_gstreamer_init() {
        failures.push(e);
    }
    if let Err(e) = check_input_device(&cfg) {
        failures.push(e);
    }
    if let Err(e) = check_output_device(&cfg) {
        failures.push(e);
    }
    if let Err(e) = check_effect_chain(&cfg) {
        failures.push(e);
    }
    if let Some(model) = args.common.model.as_deref() {
        if let Err(e) = check_model_file(model) {
            failures.push(e);
        }
    }

    if failures.is_empty() {
        info!("all checks passed");
        Ok(())
    } else {
        for failure in &failures {
            eprintln!("Error: {}", failure.reason());
            if let Some(hint) = failure.hint() {
                eprintln!("Hint: {hint}");
            }
        }
        Err(failures.into_iter().next().expect("non-empty by branch"))
    }
}

fn check_gstreamer_init() -> Result<(), FluxError> {
    fluxframe_gst::init()?;
    info!("gstreamer init: ok");
    Ok(())
}

fn check_input_device(cfg: &FluxConfig) -> Result<(), FluxError> {
    let device = &cfg.input.device;
    if device == "testsrc" {
        info!(device, "input: synthetic source (no device check needed)");
        return Ok(());
    }
    if !device.starts_with("/dev/") {
        return Err(FluxError::Config {
            reason: format!("input '{device}' is neither 'testsrc' nor a /dev/* path"),
            hint: Some("use --input testsrc or --input /dev/video<N>".into()),
        });
    }
    let path = Path::new(device);
    if !path.exists() {
        return Err(FluxError::from(PipelineError::InputDeviceUnavailable {
            device: device.clone(),
            reason: "device does not exist".into(),
            hint: "run 'fluxframe list' for available devices".into(),
        }));
    }
    match std::fs::OpenOptions::new().read(true).open(path) {
        Ok(_) => {
            info!(device, "input device: readable");
            Ok(())
        }
        Err(e) => Err(FluxError::from(PipelineError::InputDeviceUnavailable {
            device: device.clone(),
            reason: e.to_string(),
            hint: input_open_hint(&e),
        })),
    }
}

fn check_output_device(cfg: &FluxConfig) -> Result<(), FluxError> {
    let device = &cfg.output.device;
    if device == "auto" {
        info!("output: autovideosink (no device check needed)");
        return Ok(());
    }
    if !device.starts_with("/dev/") {
        info!(device, "output: fakesink (no device check needed)");
        return Ok(());
    }
    let path = Path::new(device);
    if !path.exists() {
        return Err(FluxError::from(PipelineError::OutputDeviceUnavailable {
            device: device.clone(),
            reason: "device does not exist".into(),
            hint: "load v4l2loopback (see README) or run 'fluxframe list'".into(),
        }));
    }
    match std::fs::OpenOptions::new().write(true).open(path) {
        Ok(_) => {
            warn_if_not_loopback(device);
            info!(device, "output device: writable");
            Ok(())
        }
        Err(e) => Err(FluxError::from(PipelineError::OutputDeviceUnavailable {
            device: device.clone(),
            reason: e.to_string(),
            hint: output_open_hint(&e),
        })),
    }
}

fn warn_if_not_loopback(device: &str) {
    let devices = enumerate_devices();
    if let Some(d) = devices.iter().find(|d| d.path.to_string_lossy() == device) {
        if !matches!(d.kind, V4l2DeviceKind::Virtual) {
            warn!(
                device = device,
                kind = ?d.kind,
                "output device does not look like a v4l2loopback; you may be writing to a real camera"
            );
        }
    }
}

fn input_open_hint(e: &std::io::Error) -> String {
    use std::io::ErrorKind;
    match e.kind() {
        ErrorKind::PermissionDenied => "add the user to the 'video' group or check udev rules".into(),
        _ if e.raw_os_error() == Some(16) => {
            "another application is holding the device; close it (e.g. browser tab, OBS)".into()
        }
        _ => "see 'dmesg' or 'v4l2-ctl --device=... --all' for kernel diagnostics".into(),
    }
}

fn output_open_hint(e: &std::io::Error) -> String {
    use std::io::ErrorKind;
    match e.kind() {
        ErrorKind::PermissionDenied => "add the user to the 'video' group".into(),
        _ if e.raw_os_error() == Some(16) => {
            "another application is using the loopback device".into()
        }
        _ => "see 'dmesg' or v4l2loopback module state".into(),
    }
}

fn check_effect_chain(cfg: &FluxConfig) -> Result<(), FluxError> {
    if cfg.effects.chain.is_empty() {
        return Err(FluxError::Config {
            reason: "effect chain is empty".into(),
            hint: Some("set --effect <name> or fill [effects].chain in the config".into()),
        });
    }
    let registry = default_registry();
    for name in &cfg.effects.chain {
        let normalised = normalise_effect_name(name);
        if !registry.contains(&normalised) {
            return Err(FluxError::Config {
                reason: format!("effect '{name}' is not registered (normalised to '{normalised}')"),
                hint: Some("available effects: passthrough (Stage 1) — more land in Stage 4".into()),
            });
        }
    }
    info!(chain = ?cfg.effects.chain, "effect chain: ok");
    Ok(())
}

fn check_model_file(path: &Path) -> Result<(), FluxError> {
    if !path.exists() {
        return Err(FluxError::Config {
            reason: format!("model file not found: {}", path.display()),
            hint: Some("pass --model <path> pointing at a valid ONNX model".into()),
        });
    }
    info!(model = %path.display(), "model file: exists");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_effect_chain_empty_fails() {
        let mut cfg = FluxConfig::default();
        cfg.effects.chain.clear();
        let err = check_effect_chain(&cfg).expect_err("empty chain must fail");
        let msg = format!("{err}");
        assert!(msg.contains("empty"), "got: {msg}");
    }

    #[test]
    fn check_effect_chain_unknown_fails() {
        let mut cfg = FluxConfig::default();
        cfg.effects.chain = vec!["bogus-effect-name".into()];
        let err = check_effect_chain(&cfg).expect_err("unknown effect must fail");
        let msg = format!("{err}");
        assert!(msg.contains("bogus"), "got: {msg}");
    }

    #[test]
    fn check_effect_chain_passthrough_ok() {
        let mut cfg = FluxConfig::default();
        cfg.effects.chain = vec!["passthrough".into()];
        check_effect_chain(&cfg).expect("passthrough is registered");
    }

    #[test]
    fn check_input_device_accepts_testsrc() {
        let mut cfg = FluxConfig::default();
        cfg.input.device = "testsrc".into();
        check_input_device(&cfg).expect("testsrc always passes");
    }

    #[test]
    fn check_input_device_rejects_unknown_scheme() {
        let mut cfg = FluxConfig::default();
        cfg.input.device = "http://example.com/stream".into();
        let err = check_input_device(&cfg).expect_err("unsupported scheme must fail");
        let msg = format!("{err}");
        assert!(msg.contains("testsrc") || msg.contains("/dev"), "got: {msg}");
    }
}
