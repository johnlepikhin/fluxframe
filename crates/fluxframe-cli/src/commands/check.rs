//! `fluxframe check` — verify GStreamer init, devices and effect chain.
//!
//! Stage 2 implementation: pre-flight checks before `fluxframe run` so
//! users see structured §27 Error/Hint diagnostics instead of opaque
//! GStreamer failures.  Required-elements introspection lands in
//! Stage 5 (needs a helper in `fluxframe-gst`).

use std::path::Path;

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
        // Print every failure here, in §27 Error/Hint form, so the
        // operator sees the full picture (not just the first failure).
        for failure in &failures {
            eprintln!("Error: {}", failure.reason());
            if let Some(hint) = failure.hint() {
                eprintln!("Hint: {hint}");
            }
        }
        // Return a *summary* error (without a Hint) so `main.rs` prints
        // one extra line — "Error: N pre-flight check(s) failed; see
        // messages above" — and not the first failure again.  Without
        // this collapse, every failure would be printed three times
        // (once here, once by `main.rs`, once by the `tracing::error!`
        // there).
        let n = failures.len();
        Err(FluxError::Config {
            reason: format!("{n} pre-flight check(s) failed; see messages above"),
            hint: None,
        })
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
    // Delegate the actual canonicalisation + open-probe + hint-rendering
    // to `fluxframe_gst::check_v4l2_input_access` so the §27 Error/Hint
    // text is rendered in exactly one place across the workspace.
    let canon = fluxframe_gst::check_v4l2_input_access(Path::new(device))?;
    info!(device, canonical = %canon.display(), "input device: readable");
    Ok(())
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
    // Delegate to `fluxframe_gst::check_v4l2_output_access` for the same
    // reason as on the input side — keep hint phrasing canonical.
    let canon = fluxframe_gst::check_v4l2_output_access(Path::new(device))?;
    warn_if_not_loopback(&canon);
    info!(device, canonical = %canon.display(), "output device: writable");
    Ok(())
}

fn warn_if_not_loopback(canon: &Path) {
    let devices = enumerate_devices();
    if let Some(d) = devices.iter().find(|d| d.path == canon) {
        if !matches!(d.kind, V4l2DeviceKind::Virtual) {
            warn!(
                device = %canon.display(),
                kind = ?d.kind,
                "output device does not look like a v4l2loopback; you may be writing to a real camera"
            );
        }
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
