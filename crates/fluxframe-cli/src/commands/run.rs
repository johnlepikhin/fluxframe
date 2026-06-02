//! `fluxframe run` — dispatch the configured input backend onto the
//! shared runtime supervisor (`testsrc` and V4L2 are both supported as
//! of Stage 2).
//!
//! `[input] device = "auto"` enters a polling wait state: the
//! supervisor enumerates `/dev/video*`, picks the first openable
//! capture device (skipping the output node), and starts a normal
//! run. When the run ends with an error (camera unplugged, busy,
//! permissions hiccup) the loop re-enumerates and tries again until
//! either a clean shutdown or Ctrl-C — the binary is intended to
//! sit in the background and outlive any individual camera session.

use std::path::PathBuf;
use std::time::Duration;

use fluxframe_core::{FluxConfig, FluxError, InputDevice};
use tracing::{info, warn};

use crate::cli::RunArgs;
use crate::config_merge::{CliOverrides, apply, load};
use crate::preset;
use crate::runtime::{
    InputSpec, OutputSpec, classify_input, classify_output, ensure_ctrlc_handler,
    is_shutdown_requested, run_testsrc_chain, run_v4l2_chain, wait_for_shutdown,
};

/// Entry point for `fluxframe run`.
///
/// # Errors
///
/// Returns a [`FluxError`] when the config file cannot be loaded, the
/// merged configuration fails validation, the requested preset cannot
/// be resolved or built, or the runtime pipeline fails with an error
/// the auto loop cannot recover from.
pub fn run(args: RunArgs) -> Result<(), FluxError> {
    let overrides = CliOverrides {
        input: args.common.input,
        output: args.common.output,
        width: args.width,
        height: args.height,
        fps: args.fps,
    };

    let cfg = load(args.common.config.as_deref())?;
    let cfg = apply(cfg, &overrides);
    cfg.validate()?;

    // Resolve the preset once up-front so a misnamed preset fails fast
    // (before we touch any GStreamer state) and the auto input loop
    // does not re-enter a doomed configuration on each
    // device-appearance event. We store the name as `String` so the
    // borrow does not span the per-iteration `cfg.clone()` in
    // `run_auto`.
    let (resolved_name, _preset) = preset::resolve(&cfg, args.preset.as_deref())?;
    let preset_name = resolved_name.to_string();
    info!(
        input = %cfg.input.device,
        output = %cfg.output.device,
        width = cfg.input.width,
        height = cfg.input.height,
        fps = cfg.input.fps,
        preset = %preset_name,
        "merged configuration"
    );

    match &cfg.input.device {
        InputDevice::Auto => run_auto(&cfg, &preset_name, args.common.config.as_deref()),
        _ => run_once(&cfg, &preset_name, args.common.config.as_deref()),
    }
}

/// Pick the first available V4L2 capture device, build the pipeline,
/// run until exit. On any *transient* pipeline error (camera unplugged,
/// busy, IO flake) re-enter the polling loop and try again; a
/// *permanent* error (missing GStreamer plugin, invalid config) is
/// surfaced immediately. Returns `Ok(())` only on a clean shutdown
/// (Ctrl-C or pipeline EOS).
///
/// The wait loop deduplicates log lines so a long stretch with no
/// candidate device or a recurring transient failure does not flood
/// the operator's journal (1800 lines / hour at the default 2 s poll).
fn run_auto(
    cfg: &FluxConfig,
    preset_name: &str,
    config_path: Option<&std::path::Path>,
) -> Result<(), FluxError> {
    ensure_ctrlc_handler();
    let interval = Duration::from_secs(u64::from(cfg.input.auto.poll_interval_secs.max(1)));
    let exclude = compute_excludes(cfg);
    info!(
        poll_interval_secs = interval.as_secs(),
        exclude = ?exclude,
        "entering auto-input wait loop"
    );

    // Log-dedup state. `last_failure_kind` keys on the FluxError's
    // discriminant so a recurring failure mode (e.g. /dev/video0 keeps
    // returning EBUSY) only logs once until something else happens.
    // `last_no_devices_logged` collapses the empty-candidate state
    // into a single line until a device shows up.
    let mut last_failure_kind: Option<String> = None;
    let mut last_no_devices_logged = false;

    loop {
        if is_shutdown_requested() {
            info!("auto-input loop: shutdown requested before any device was found");
            return Ok(());
        }
        let candidates = fluxframe_gst::v4l2_caps::enumerate_capture_devices();
        if let Some(path) = fluxframe_gst::v4l2_caps::pick_input_device(&candidates, &exclude) {
            last_no_devices_logged = false;
            info!(device = %path.display(), "auto-input picked");
            let mut resolved = cfg.clone();
            resolved.input.device = InputDevice::Path(path.clone());
            match run_once(&resolved, preset_name, config_path) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if !e.is_transient() {
                        warn!(error = %e, "auto-input: permanent error, surfacing");
                        return Err(e);
                    }
                    let kind = format!("{:?}", std::mem::discriminant(&e));
                    if last_failure_kind.as_deref() != Some(kind.as_str()) {
                        warn!(
                            error = %e,
                            "auto-input transient error (new); returning to wait"
                        );
                        last_failure_kind = Some(kind);
                    }
                }
            }
        } else {
            last_failure_kind = None;
            if !last_no_devices_logged {
                info!(
                    candidates = ?candidates,
                    "auto-input: no openable capture devices, will retry silently",
                );
                last_no_devices_logged = true;
            }
        }
        if wait_for_shutdown(interval) {
            info!("auto-input loop: shutdown during wait");
            return Ok(());
        }
    }
}

/// Build the effect chain from the named preset and dispatch it to the
/// appropriate per-backend runtime entry point. One execution; does
/// not poll or retry.
///
/// Re-resolving the preset on each iteration of `run_auto` keeps the
/// lifetime contract simple (no `&Preset` borrow spanning the
/// per-iteration `cfg.clone()`) and the lookup is O(log n) over the
/// presets map — negligible compared to GStreamer pipeline setup.
fn run_once(
    cfg: &FluxConfig,
    preset_name: &str,
    config_path: Option<&std::path::Path>,
) -> Result<(), FluxError> {
    let (name, preset) = preset::resolve(cfg, Some(preset_name))?;
    let chain = preset::build_chain(name, preset)?;

    // Dispatch via `classify_input` so the testsrc-vs-V4L2 triage lives in
    // exactly one place (shared with `commands::check`).  Adding a variant
    // to `InputSpec` turns this into a compile error and prompts the
    // developer to teach the dispatch about it.
    match classify_input(cfg) {
        InputSpec::Testsrc => run_testsrc_chain(cfg, name, config_path, chain),
        InputSpec::V4l2(_) => run_v4l2_chain(cfg, name, config_path, chain),
        InputSpec::Unsupported(d) => Err(FluxError::Config {
            reason: format!("input '{d}' is not supported"),
            hint: Some("supported inputs: testsrc, /dev/video* (V4L2)".into()),
        }),
    }
}

/// Build the device-exclusion list for `pick_input_device`. Combines
/// the operator-supplied `[input.auto].exclude_devices` with the
/// output device path when it resolves to a V4L2 sink — we never want
/// auto-pick to feed our own loopback back into capture.
///
/// Triage goes through [`classify_output`] (the same dispatch used by
/// `commands::check` and `runtime::resolve_output_sink`) so the
/// stringly-typed `starts_with("/dev/")` heuristic stays in one place
/// and `pipewire` / `fakesink` / `auto` outputs cannot accidentally
/// collide with the input scan.
fn compute_excludes(cfg: &FluxConfig) -> Vec<PathBuf> {
    let mut excludes = cfg.input.auto.exclude_devices.clone();
    if let OutputSpec::V4l2(p) = classify_output(cfg) {
        if !excludes.contains(&p) {
            excludes.push(p);
        }
    }
    excludes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_excludes_includes_output_dev_path() {
        let mut cfg = FluxConfig::default();
        cfg.output.device = "/dev/video10".into();
        let ex = compute_excludes(&cfg);
        assert!(ex.contains(&PathBuf::from("/dev/video10")));
    }

    #[test]
    fn compute_excludes_skips_non_dev_output() {
        let mut cfg = FluxConfig::default();
        cfg.output.device = "fakesink".into();
        assert!(compute_excludes(&cfg).is_empty());
    }

    #[test]
    fn compute_excludes_merges_operator_excludes_with_output() {
        let mut cfg = FluxConfig::default();
        cfg.output.device = "/dev/video10".into();
        cfg.input.auto.exclude_devices = vec![PathBuf::from("/dev/video20")];
        let ex = compute_excludes(&cfg);
        assert!(ex.contains(&PathBuf::from("/dev/video10")));
        assert!(ex.contains(&PathBuf::from("/dev/video20")));
        assert_eq!(ex.len(), 2);
    }

    #[test]
    fn compute_excludes_no_duplicate_when_output_already_listed() {
        let mut cfg = FluxConfig::default();
        cfg.output.device = "/dev/video10".into();
        cfg.input.auto.exclude_devices = vec![PathBuf::from("/dev/video10")];
        let ex = compute_excludes(&cfg);
        assert_eq!(ex.len(), 1);
    }
}
