//! `fluxframe check` — verify GStreamer init, devices and effect chain.
//!
//! Stage 2 implementation: pre-flight checks before `fluxframe run` so
//! users see structured §27 Error/Hint diagnostics instead of opaque
//! GStreamer failures.  Required-elements introspection lands in
//! Stage 5 (needs a helper in `fluxframe-gst`).

use std::path::Path;

use fluxframe_core::{FluxConfig, FluxError, normalise_effect_name};
use fluxframe_effects::ml::OnnxEngine;
use fluxframe_gst::{V4l2DeviceKind, enumerate_devices};
use tracing::{info, warn};

use crate::cli::CheckArgs;
use crate::config_merge::{CliOverrides, apply, load};
use crate::runtime::{InputSpec, OutputSpec, classify_input, classify_output, default_registry};

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
            use fluxframe_core::Diagnostic;
            eprintln!("Error: {}", failure.reason());
            if let Some(hint) = failure.hint() {
                eprintln!("Hint: {hint}");
            }
        }
        // Surface an `Aggregated` error so `main.rs` recognises that we
        // already rendered each underlying failure and skips its own
        // §27 print.  Preserving `primary` keeps `matches!()` consumers
        // (tests in particular) working unchanged.
        let count = failures.len();
        let primary = failures.into_iter().next().expect("non-empty");
        Err(FluxError::Aggregated {
            primary: Box::new(primary),
            count,
        })
    }
}

fn check_gstreamer_init() -> Result<(), FluxError> {
    fluxframe_gst::init()?;
    info!("gstreamer init: ok");
    Ok(())
}

fn check_input_device(cfg: &FluxConfig) -> Result<(), FluxError> {
    // Triage lives in [`classify_input`] (shared with `commands::run`)
    // so both code paths cannot drift apart on which device strings are
    // valid.  Delegate the actual canonicalisation + open-probe +
    // hint-rendering to `fluxframe_gst::check_v4l2_input_access` so the
    // §27 Error/Hint text is rendered in exactly one place.
    match classify_input(cfg) {
        InputSpec::Testsrc => {
            info!(device = %cfg.input.device, "input: synthetic source (no device check needed)");
            Ok(())
        }
        InputSpec::V4l2(path) => {
            let canon = fluxframe_gst::check_v4l2_input_access(&path)?;
            info!(
                device = %path.display(),
                canonical = %canon.display(),
                "input device: readable",
            );
            Ok(())
        }
        InputSpec::Unsupported(d) => Err(FluxError::Config {
            reason: format!("input '{d}' is not supported"),
            hint: Some("use --input testsrc or --input /dev/video<N>".into()),
        }),
    }
}

fn check_output_device(cfg: &FluxConfig) -> Result<(), FluxError> {
    // Same rationale as `check_input_device`: route through
    // [`classify_output`] so `commands::check` and `commands::run` agree
    // on which output strings are valid.
    match classify_output(cfg) {
        OutputSpec::Auto => {
            info!("output: autovideosink (no device check needed)");
            Ok(())
        }
        OutputSpec::Fake => {
            info!("output: fakesink (no device check needed)");
            Ok(())
        }
        OutputSpec::V4l2(path) => {
            let canon = fluxframe_gst::check_v4l2_output_access(&path)?;
            warn_if_not_loopback(&canon);
            info!(
                device = %path.display(),
                canonical = %canon.display(),
                "output device: writable",
            );
            Ok(())
        }
        OutputSpec::Unsupported(d) => Err(FluxError::Config {
            reason: format!("output '{d}' is not supported"),
            hint: Some("supported outputs: auto, fakesink, /dev/video<N>".into()),
        }),
    }
}

fn warn_if_not_loopback(canonical: &Path) {
    let devices = enumerate_devices();
    let Some(d) = devices.iter().find(|d| d.path == canonical) else {
        // Device passed the open-probe but isn't in the sysfs enumeration
        // (e.g. the user pointed at a path outside `/dev/video*`).  We
        // intentionally do not warn the operator here — they already
        // know they are writing to a non-standard path — but record a
        // debug breadcrumb for support.
        tracing::debug!(
            device = %canonical.display(),
            "loopback classification skipped (device not in enumeration)",
        );
        return;
    };
    match d.kind {
        V4l2DeviceKind::Virtual => {} // expected
        V4l2DeviceKind::Unknown => {
            warn!(
                device = %canonical.display(),
                "device kind unknown; could not verify loopback. If this is a real camera, output may be wasted.",
            );
        }
        _ => {
            warn!(
                device = %canonical.display(),
                kind = ?d.kind,
                "output device does not look like a v4l2loopback; you may be writing to a real camera",
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
                hint: Some(
                    "available effects: passthrough (Stage 1) — more land in Stage 4".into(),
                ),
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
    let config = fluxframe_effects::ml::load_sidecar_or_placeholder(path)?;
    let engine = OnnxEngine::load(path, config)?;
    info!(
        inputs = engine.input_count(),
        outputs = engine.output_count(),
        "model file: loads OK",
    );
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
        use fluxframe_core::Diagnostic;
        let mut cfg = FluxConfig::default();
        cfg.input.device = "http://example.com/stream".into();
        let err = check_input_device(&cfg).expect_err("unsupported scheme must fail");
        let reason = err.reason();
        assert!(
            reason.contains("not supported") || reason.contains("http://"),
            "got reason: {reason}",
        );
        let hint = err.hint().expect("Unsupported should carry a hint");
        assert!(
            hint.contains("testsrc") || hint.contains("/dev"),
            "got hint: {hint}",
        );
    }
}
