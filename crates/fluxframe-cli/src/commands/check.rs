//! `fluxframe check` — verify GStreamer init, devices and the selected
//! preset's pipeline before `fluxframe run` so users see structured §27
//! Error/Hint diagnostics instead of opaque GStreamer failures.

use std::path::Path;

use fluxframe_core::{FluxConfig, FluxError, InputDevice, PipelineSection, Preset};
#[cfg(feature = "ml")]
use fluxframe_effects::ml::OnnxEngine;
use fluxframe_effects::{mask_effects, plane_effects};
use fluxframe_gst::{V4l2DeviceKind, enumerate_devices};
use tracing::{info, warn};

use crate::cli::CheckArgs;
use crate::config_merge::{CliOverrides, apply, load};
use crate::preset;
use crate::runtime::{InputSpec, OutputSpec, classify_input, classify_output};

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
    if let Err(e) = check_preset(&cfg, args.preset.as_deref()) {
        failures.push(e);
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
    // Auto-pick mode: nothing to verify ahead of time. Just enumerate
    // and report what would be picked right now so the operator can
    // sanity-check the candidate list. An empty list is a warning,
    // not an error — runtime will wait for a device to appear.
    if matches!(cfg.input.device, InputDevice::Auto) {
        let candidates = fluxframe_gst::v4l2_caps::enumerate_capture_devices();
        if candidates.is_empty() {
            tracing::warn!(
                "input: auto — no capture devices available right now \
                 (runtime will keep polling at `[input.auto].poll_interval_secs`)"
            );
        } else {
            info!(
                devices = ?candidates,
                "input: auto — first available will be picked at startup",
            );
        }
        return Ok(());
    }
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
        OutputSpec::Pipewire(node_name) => {
            // No pre-flight to run: PipeWire availability and the
            // `pipewiresink` element are validated at pipeline build
            // time, which surfaces a structured MissingElement error
            // when `gst-plugin-pipewire` is not installed.
            info!(node = ?node_name, "output: pipewire (validated at pipeline build)");
            Ok(())
        }
        OutputSpec::Unsupported(d) => Err(FluxError::Config {
            reason: format!("output '{d}' is not supported"),
            hint: Some("supported outputs: auto, fakesink, /dev/video<N>, pipewire[:name]".into()),
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

/// Validate the preset selected on the command line (or the implicit
/// [`preset::DEFAULT_PRESET_NAME`] preset).
///
/// Checks performed:
///   1. The named preset exists ([`preset::resolve`]).
///   2. Every effect name in `mask.chain` is registered in the mask
///      registry.
///   3. Every effect name in `background.chain` / `foreground.chain`
///      is registered in the plane registry.
///   4. The preset can be turned into an [`EffectChain`]
///      ([`preset::build_chain`]) — this surfaces the same
///      feature-gated and bg-without-mask errors that `run` would
///      raise, so `check` and `run` cannot disagree.
///   5. (`ml` feature only) The model path referenced by the mask
///      section exists on disk and loads as an ONNX model.
///
/// [`EffectChain`]: fluxframe_effects::EffectChain
fn check_preset(cfg: &FluxConfig, requested: Option<&str>) -> Result<(), FluxError> {
    let (name, preset) = preset::resolve(cfg, requested)?;

    // Registry-level effect-name validation lives here (and only here)
    // because `preset::build_chain` already surfaces these as
    // `EffectError::UnknownEffect` via the composite builder when a
    // `[mask]` section is present — but `check` should also reject
    // unknown names in `[background]`/`[foreground]` chains when no
    // mask is configured (in which case those sections are config
    // bugs that `build_chain` rejects, but the operator deserves the
    // sharper "unknown effect" message first).
    check_preset_sections(name, preset)?;

    // Surface the same Ok/Err that `run` would see when constructing
    // its chain.  Catches: mask-without-ml, bg/fg-without-mask, and
    // composite-builder failures (per-effect TOML, model missing).
    let _ = preset::build_chain(name, preset)?;

    #[cfg(feature = "ml")]
    if let Some(mask) = preset.mask.as_ref() {
        if let Some(model) = mask.model.as_deref() {
            check_model_file(model)?;
        }
    }

    info!(preset = %name, "preset: ok");
    Ok(())
}

/// Validate each effect name appearing in the preset's sub-chains is
/// known to the corresponding registry.
fn check_preset_sections(preset_name: &str, preset: &Preset) -> Result<(), FluxError> {
    let mask_registry = mask_effects::default_registry();
    let plane_registry = plane_effects::default_registry();

    if let Some(mask) = preset.mask.as_ref() {
        check_mask_chain(preset_name, mask, &mask_registry)?;
    }
    if let Some(bg) = preset.background.as_ref() {
        check_plane_chain(preset_name, "background", bg, &plane_registry)?;
    }
    if let Some(fg) = preset.foreground.as_ref() {
        check_plane_chain(preset_name, "foreground", fg, &plane_registry)?;
    }
    Ok(())
}

fn check_mask_chain(
    preset_name: &str,
    mask: &PipelineSection,
    registry: &mask_effects::MaskEffectRegistry,
) -> Result<(), FluxError> {
    for name in &mask.chain {
        if !registry.contains(name) {
            return Err(FluxError::Config {
                reason: format!(
                    "preset '{preset_name}': mask chain references unknown effect '{name}'"
                ),
                hint: Some(format!(
                    "available mask effects: {}",
                    registry.names().join(", "),
                )),
            });
        }
    }
    Ok(())
}

fn check_plane_chain(
    preset_name: &str,
    section: &str,
    plane: &PipelineSection,
    registry: &plane_effects::PlaneEffectRegistry,
) -> Result<(), FluxError> {
    for name in &plane.chain {
        if !registry.contains(name) {
            return Err(FluxError::Config {
                reason: format!(
                    "preset '{preset_name}': {section} chain references unknown effect '{name}'"
                ),
                hint: Some(format!(
                    "available plane effects: {}",
                    registry.names().join(", "),
                )),
            });
        }
    }
    Ok(())
}

#[cfg(feature = "ml")]
fn check_model_file(path: &Path) -> Result<(), FluxError> {
    if !path.exists() {
        return Err(FluxError::Config {
            reason: format!("model file not found: {}", path.display()),
            hint: Some(
                "set the `model = \"...\"` field in the preset's [mask] section \
                 to a valid ONNX path"
                    .into(),
            ),
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
    use fluxframe_core::Preset;

    fn cfg_with(toml: &str) -> FluxConfig {
        FluxConfig::from_toml_str(toml).expect("toml parses")
    }

    #[test]
    fn check_preset_fails_when_no_default_and_no_flag() {
        let cfg = cfg_with("[presets.custom]\n");
        let err = check_preset(&cfg, None).expect_err("missing default → error");
        let msg = format!("{err}");
        assert!(msg.contains("default"), "got: {msg}");
    }

    #[test]
    fn check_preset_fails_when_named_missing() {
        let cfg = cfg_with("[presets.default]\n");
        let err = check_preset(&cfg, Some("nope")).expect_err("missing requested → error");
        let msg = format!("{err}");
        assert!(msg.contains("'nope'"), "got: {msg}");
    }

    #[test]
    fn check_preset_default_with_empty_sections_passes() {
        let cfg = cfg_with("[presets.default]\n");
        check_preset(&cfg, None).expect("empty default preset is valid");
    }

    #[test]
    fn check_preset_rejects_unknown_mask_effect() {
        let cfg = cfg_with(
            r#"
[presets.default]

[presets.default.mask]
model = "/tmp/nonexistent.onnx"
chain = ["nonexistent_mask_effect"]
"#,
        );
        // We intentionally do not test the model-file branch here —
        // the chain validation runs before the model existence check,
        // so the unknown-effect error is what surfaces first.
        let err = check_preset_sections(
            "default",
            cfg.presets.get("default").expect("preset present"),
        )
        .expect_err("unknown mask effect must fail");
        let msg = format!("{err}");
        assert!(msg.contains("nonexistent_mask_effect"), "got: {msg}");
    }

    #[test]
    fn check_preset_rejects_unknown_plane_effect() {
        let cfg = cfg_with(
            r#"
[presets.default]

[presets.default.background]
chain = ["nonexistent_plane_effect"]
"#,
        );
        let err = check_preset_sections(
            "default",
            cfg.presets.get("default").expect("preset present"),
        )
        .expect_err("unknown plane effect must fail");
        let msg = format!("{err}");
        assert!(msg.contains("nonexistent_plane_effect"), "got: {msg}");
    }

    #[test]
    fn check_preset_accepts_known_effects() {
        let cfg = cfg_with(
            r#"
[presets.default]

[presets.default.background]
chain = ["blur"]

[presets.default.foreground]
chain = ["passthrough"]
"#,
        );
        check_preset_sections(
            "default",
            cfg.presets.get("default").expect("preset present"),
        )
        .expect("blur+passthrough are registered");
    }

    #[test]
    fn check_input_device_accepts_testsrc() {
        let mut cfg = FluxConfig::default();
        cfg.input.device = InputDevice::Testsrc;
        check_input_device(&cfg).expect("testsrc always passes");
    }

    #[test]
    fn check_input_device_reports_auto_without_devices_as_warning_not_error() {
        // `Auto` enters the polling branch, which logs a warning and
        // returns `Ok(())` when nothing is plugged in. The unit harness
        // is unlikely to have a real V4L2 camera available; either
        // outcome counts as "no error surfaced", because absence of a
        // device is not a check-time failure.
        let mut cfg = FluxConfig::default();
        cfg.input.device = InputDevice::Auto;
        check_input_device(&cfg).expect("auto mode never errors at check time");
    }

    #[test]
    fn check_preset_sections_with_default_preset_is_noop() {
        // An empty Preset (no mask/bg/fg) trivially passes — there are
        // no chains to validate.
        let preset = Preset::default();
        check_preset_sections("empty", &preset).expect("empty preset has no chains to validate");
    }
}
