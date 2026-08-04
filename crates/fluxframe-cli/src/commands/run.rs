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
use std::time::{Duration, Instant};

use fluxframe_core::{FluxConfig, FluxError, InputDevice};
use tracing::{info, warn};

use crate::cli::RunArgs;
use crate::config_merge::{CliOverrides, apply, load, resolve_load_path, resolve_writable_path};
use crate::preset;
use crate::runtime::{
    InputSpec, OutputSpec, classify_input, classify_output, ensure_ctrlc_handler,
    is_shutdown_requested, mark_output_down, run_testsrc_chain, run_v4l2_chain, wait_for_shutdown,
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

    // Two paths derived from the same flags:
    //   - load_path: what to open at startup (None on a fresh install
    //     with no XDG file → daemon boots from built-in defaults
    //     silently);
    //   - writable_path: where Save / Reload will target later (the
    //     resolved XDG default even when the file does not yet exist,
    //     so the GUI's first Save can bootstrap it).
    let load_path = resolve_load_path(args.common.config.as_deref(), args.common.no_default_config);
    let writable_path =
        resolve_writable_path(args.common.config.as_deref(), args.common.no_default_config);
    let cfg = load(load_path.as_deref())?;
    let cfg = apply(cfg, &overrides);
    cfg.validate()?;

    // The subscriber was installed from `-v` / `RUST_LOG` before the
    // config existed; now that we have a validated one, let its
    // `[logging] level` take effect for the rest of the daemon's life.
    // Everything logged above this line (config-load failures included)
    // still had to go through the bootstrap filter.
    crate::logging::apply_config_level(&cfg.logging.level);

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
        config = ?writable_path,
        "merged configuration"
    );

    match &cfg.input.device {
        InputDevice::Auto => run_auto(&cfg, &preset_name, writable_path.as_deref()),
        _ => run_once(&cfg, &preset_name, writable_path.as_deref()),
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
    // Backoff for *repeated* transient failures only. A camera that
    // reports EBUSY on every attempt used to be retried at the flat
    // `poll_interval_secs`, producing ~18 full pipeline rebuilds a
    // minute for as long as the condition lasted; each rebuild is a
    // window where `/dev/video10` has no producer and therefore
    // advertises no CAPTURE caps, so clients could not see the camera
    // for minutes at a time.
    //
    // Only the failure branch backs off. The "no candidate devices"
    // branch keeps the flat poll: a camera being plugged in should be
    // picked up promptly, and two competing timers would make that
    // latency unpredictable.
    let backoff_base = Duration::from_millis(u64::from(cfg.input.acquire_backoff_base_ms.max(1)));
    let backoff_max = Duration::from_millis(u64::from(cfg.input.acquire_backoff_max_ms.max(1)));
    let mut failure_backoff: Option<Duration> = None;

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
            let run_started = Instant::now();
            match run_once(&resolved, preset_name, config_path) {
                Ok(()) => {
                    mark_output_down(None);
                    return Ok(());
                }
                Err(e) => {
                    // The run took the whole pipeline down with it,
                    // loopback output included: from here until the next
                    // successful `output.start()` the node advertises no
                    // CAPTURE caps and clients cannot see the camera.
                    mark_output_down(Some(&e));
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
                    // A run that stayed up for a while and then failed is
                    // a fresh incident, not a tight failure loop — start
                    // over from the base delay so an isolated flap a day
                    // later is not punished with the maximum wait.
                    failure_backoff = Some(next_failure_backoff(
                        failure_backoff,
                        run_started.elapsed(),
                        backoff_base,
                        backoff_max,
                    ));
                }
            }
        } else {
            last_failure_kind = None;
            failure_backoff = None;
            if !last_no_devices_logged {
                info!(
                    candidates = ?candidates,
                    "auto-input: no openable capture devices, will retry silently",
                );
                last_no_devices_logged = true;
            }
        }
        let wait = failure_backoff.unwrap_or(interval);
        if wait_for_shutdown(wait) {
            info!("auto-input loop: shutdown during wait");
            return Ok(());
        }
    }
}

/// Multiplier for deciding a run "held" long enough to count as a fresh
/// incident rather than part of an ongoing failure loop.
const BACKOFF_RESET_FACTOR: u32 = 4;

/// Next delay before retrying after a transient run failure.
///
/// Doubles while failures keep arriving quickly, and resets to `base`
/// once a run managed to stay up for `BACKOFF_RESET_FACTOR × max`.
/// Without the reset an isolated flap a day into an otherwise healthy
/// session would still be met with the maximum delay; without the
/// growth, a persistently busy camera would be retried in a tight loop.
///
/// Pure so the escalate/reset behaviour is testable without sleeping.
fn next_failure_backoff(
    current: Option<Duration>,
    run_uptime: Duration,
    base: Duration,
    max: Duration,
) -> Duration {
    if run_uptime >= max * BACKOFF_RESET_FACTOR {
        return base.min(max);
    }
    match current {
        None => base.min(max),
        Some(prev) => (prev * 2).min(max),
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

    const BASE: Duration = Duration::from_millis(500);
    const MAX: Duration = Duration::from_secs(5);

    #[test]
    fn first_failure_waits_the_base_delay() {
        assert_eq!(
            next_failure_backoff(None, Duration::from_millis(10), BASE, MAX),
            BASE
        );
    }

    #[test]
    fn repeated_fast_failures_double_up_to_the_ceiling() {
        let mut d = next_failure_backoff(None, Duration::ZERO, BASE, MAX);
        assert_eq!(d, Duration::from_millis(500));
        d = next_failure_backoff(Some(d), Duration::ZERO, BASE, MAX);
        assert_eq!(d, Duration::from_secs(1));
        d = next_failure_backoff(Some(d), Duration::ZERO, BASE, MAX);
        assert_eq!(d, Duration::from_secs(2));
        d = next_failure_backoff(Some(d), Duration::ZERO, BASE, MAX);
        assert_eq!(d, Duration::from_secs(4));
        // Ceiling holds.
        d = next_failure_backoff(Some(d), Duration::ZERO, BASE, MAX);
        assert_eq!(d, MAX);
        d = next_failure_backoff(Some(d), Duration::ZERO, BASE, MAX);
        assert_eq!(d, MAX);
    }

    #[test]
    fn a_long_lived_run_resets_the_backoff() {
        // An isolated flap after a healthy session must not inherit the
        // maximum delay accumulated hours earlier.
        let long_enough = MAX * BACKOFF_RESET_FACTOR;
        assert_eq!(
            next_failure_backoff(Some(MAX), long_enough, BASE, MAX),
            BASE
        );
        // Just under the threshold still counts as the same incident.
        assert_eq!(
            next_failure_backoff(
                Some(BASE),
                long_enough - Duration::from_millis(1),
                BASE,
                MAX
            ),
            BASE * 2
        );
    }

    #[test]
    fn backoff_never_exceeds_max_even_when_base_is_larger() {
        // Misconfiguration (base > max) must not produce a delay beyond
        // the operator's stated ceiling.
        let base = Duration::from_secs(30);
        let max = Duration::from_secs(5);
        assert_eq!(next_failure_backoff(None, Duration::ZERO, base, max), max);
    }

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
