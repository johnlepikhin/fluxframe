//! End-to-end runtime supervisor.
//!
//! Owns the lifetime of input/output pipelines, the processing worker
//! thread and the Ctrl-C signal handler.  Stage 1 wired `videotestsrc`
//! → effect chain → fakesink/autovideosink only; Stage 2 adds the V4L2
//! capture and v4l2loopback sink paths.
//!
//! GStreamer is intentionally absent from this module's surface: bus
//! events arrive through the typed [`fluxframe_gst::BusEvent`] enum,
//! draining is owned by [`fluxframe_gst::BusListener`], and the supervisor
//! itself only juggles `std::sync` primitives.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::PipelineError;
use fluxframe_core::{FluxConfig, FluxError, InputDevice, Preset, SubchainKind};
use fluxframe_effects::EffectChain;
use fluxframe_gst::input::{InputParams, InputPipeline};
use fluxframe_gst::output::{OutputParams, OutputPipeline, OutputSink};
use fluxframe_gst::{BusEvent, BusListener, BusSource, LatestFrameSlot, WatchedPipeline};
use tracing::{debug, error, info, warn};

use crate::control::{
    self, Command as ControlCommand, ListenerHandle, Response as ControlResponse,
};
use crate::metrics_reporter::MetricsReporter;
use crate::preset;
use crate::runtime_metrics::RuntimeMetrics;

/// Worker-response cap for the listener thread. The worker only polls
/// between frames; at 30 fps one cycle is ~33 ms, so 5 s is generous
/// for a `set` and tight enough to surface a wedged worker.
const WORKER_RESPONSE_TIMEOUT: Duration = Duration::from_secs(5);

/// Depth of the listener → worker command channel. Sized to absorb a
/// short burst of commands without blocking the listener.
const CONTROL_CHANNEL_DEPTH: usize = 16;

/// Depth of the worker → listener reply channel. Each command gets a
/// fresh oneshot, so depth 1 is sufficient.
const REPLY_CHANNEL_DEPTH: usize = 1;

/// Envelope passed from the listener thread to the worker: a parsed
/// command plus the channel the worker uses to send the response
/// back. Reply channel acts as a oneshot ([`REPLY_CHANNEL_DEPTH`] = 1).
pub(crate) struct ControlEnvelope {
    pub cmd: ControlCommand,
    pub reply_tx: crossbeam_channel::Sender<ControlResponse>,
}

/// Bridge from the listener's `CommandHandler` trait into the
/// worker's command channel.
struct ChannelHandler {
    tx: crossbeam_channel::Sender<ControlEnvelope>,
}

impl control::CommandHandler for ChannelHandler {
    fn handle(&self, cmd: ControlCommand) -> ControlResponse {
        let (reply_tx, reply_rx) =
            crossbeam_channel::bounded::<ControlResponse>(REPLY_CHANNEL_DEPTH);
        if self.tx.send(ControlEnvelope { cmd, reply_tx }).is_err() {
            return ControlResponse::err("worker channel closed", None);
        }
        match reply_rx.recv_timeout(WORKER_RESPONSE_TIMEOUT) {
            Ok(r) => r,
            Err(_) => ControlResponse::err(
                "worker did not respond within timeout",
                Some("the pipeline may be wedged; check the supervisor log".into()),
            ),
        }
    }
}

/// Spawn the control listener when `[control].enabled = true`.
/// Returns the listener handle (kept alive for the run) and the
/// receiver the worker reads commands from. Returns `(None, None)`
/// when the feature is disabled or the listener fails to bind
/// (logged but not fatal).
fn maybe_spawn_control(
    cfg: &FluxConfig,
) -> (
    Option<ListenerHandle>,
    Option<crossbeam_channel::Receiver<ControlEnvelope>>,
) {
    if !cfg.control.enabled {
        return (None, None);
    }
    let socket_path = cfg
        .control
        .socket_path
        .clone()
        .unwrap_or_else(control::default_socket_path);
    let (tx, rx) = crossbeam_channel::bounded::<ControlEnvelope>(CONTROL_CHANNEL_DEPTH);
    let handler = ChannelHandler { tx };
    match control::spawn(socket_path, handler) {
        Ok(handle) => (Some(handle), Some(rx)),
        Err(e) => {
            warn!(error = %e, "control socket failed to bind; live-reconfig disabled");
            (None, None)
        }
    }
}

/// Short stable label for a control [`ControlCommand`] variant. Used
/// by the worker loop to tag the per-command telemetry line, so
/// elapsed time can be filtered by command kind in log queries
/// without binding payload values into the log target.
fn command_label(cmd: &ControlCommand) -> &'static str {
    match cmd {
        ControlCommand::ListPresets => "list_presets",
        ControlCommand::CurrentPreset => "current_preset",
        ControlCommand::ListEffects => "list_effects",
        ControlCommand::GetConfig { .. } => "get_config",
        ControlCommand::SetPreset { .. } => "set_preset",
        ControlCommand::Set { .. } => "set",
        ControlCommand::SetChain { .. } => "set_chain",
        ControlCommand::Reload => "reload",
        ControlCommand::SavePreset => "save_preset",
        ControlCommand::SavePresetAs { .. } => "save_preset_as",
        ControlCommand::ConfigPath => "config_path",
        // `ControlCommand` is `#[non_exhaustive]` for forward-compat
        // with future wire variants. Tag unknowns explicitly so log
        // queries surface "we received something we cannot label yet".
        _ => "unknown",
    }
}

/// Apply a single control command. Returns the response the worker
/// will hand back to the listener thread.
#[allow(
    clippy::too_many_arguments,
    reason = "command dispatcher; refactor deferred"
)]
fn apply_control_command(
    cfg: &mut FluxConfig,
    config_path: Option<&std::path::Path>,
    processing_ctx: &ProcessingContext,
    chain: &mut EffectChain,
    active_preset_name: &mut String,
    active_preset: &mut Preset,
    cmd: ControlCommand,
) -> ControlResponse {
    match cmd {
        ControlCommand::ListPresets => {
            let names: Vec<&str> = cfg.presets.keys().map(String::as_str).collect();
            ControlResponse::ok_with(serde_json::json!(names))
        }
        ControlCommand::CurrentPreset => {
            ControlResponse::ok_with(serde_json::json!(active_preset_name.clone()))
        }
        ControlCommand::SetPreset { name } => {
            let Some(new_preset) = cfg.presets.get(&name) else {
                let available: Vec<&str> = cfg.presets.keys().map(String::as_str).collect();
                return ControlResponse::err(
                    format!("preset '{name}' not defined"),
                    Some(format!("available: {}", available.join(", "))),
                );
            };
            let mut new_chain = match preset::build_chain(&name, new_preset) {
                Ok(c) => c,
                Err(e) => return ControlResponse::err(format!("build chain failed: {e}"), None),
            };
            if let Err(e) = new_chain.prepare_all(processing_ctx) {
                return ControlResponse::err(
                    format!("prepare failed: {e}"),
                    Some(format!(
                        "preset '{name}' parsed but did not initialise at runtime"
                    )),
                );
            }
            // Swap in the new chain; old chain is shut down on the
            // local variable.
            std::mem::swap(chain, &mut new_chain);
            if let Err(e) = new_chain.shutdown_all() {
                warn!(error = %e, "old chain shutdown reported an error");
            }
            active_preset_name.clone_from(&name);
            // Refresh the working preset copy from the source-of-truth
            // config so `current_config` / `set` see the new pipeline.
            *active_preset = new_preset.clone();
            info!(preset = %name, "control: swapped active preset");
            ControlResponse::ok()
        }
        ControlCommand::Set { path, value } => {
            apply_set_command(chain, active_preset, &path, value)
        }
        ControlCommand::SetChain {
            section,
            chain: new_names,
        } => apply_set_chain_command(chain, active_preset, section, &new_names),
        ControlCommand::GetConfig { path } => {
            apply_get_config_command(active_preset, path.as_deref())
        }
        ControlCommand::Reload => apply_reload_command(
            cfg,
            config_path,
            processing_ctx,
            chain,
            active_preset_name,
            active_preset,
        ),
        ControlCommand::ListEffects => apply_list_effects_command(),
        ControlCommand::SavePreset => {
            apply_save_preset_command(cfg, config_path, active_preset_name, active_preset)
        }
        ControlCommand::SavePresetAs { name } => {
            apply_save_preset_as_command(cfg, config_path, active_preset, &name)
        }
        ControlCommand::ConfigPath => apply_config_path_command(config_path),
        // `ControlCommand` is `#[non_exhaustive]`; a future variant
        // that this build does not yet handle gets a structured
        // error rather than panicking the worker thread.
        _ => ControlResponse::err(
            "unknown command — daemon was built without support for this wire variant",
            Some("update the daemon, or check the client is not ahead of the daemon".into()),
        ),
    }
}

/// Persist the in-memory active preset back into the operator's
/// TOML config file. The runtime mirror in `cfg.presets` is refreshed
/// so a subsequent `Reload` sees consistent state (otherwise a Reload
/// right after a Save would silently revert the unsaved in-memory
/// edits that landed via `Set` since the last load).
fn apply_save_preset_command(
    cfg: &mut FluxConfig,
    config_path: Option<&std::path::Path>,
    active_preset_name: &str,
    active_preset: &Preset,
) -> ControlResponse {
    let Some(path) = config_path else {
        return ControlResponse::err(
            "daemon has no writable config path",
            Some(
                "restart the daemon with --config <PATH> or without --no-default-config to enable Save"
                    .into(),
            ),
        );
    };
    if let Err(e) = crate::persist::save_preset(path, active_preset_name, active_preset) {
        return ControlResponse::err(
            format!("save failed: {e}"),
            Some(format!("path: {}", path.display())),
        );
    }
    cfg.presets
        .insert(active_preset_name.to_string(), active_preset.clone());
    info!(
        path = %path.display(),
        preset = %active_preset_name,
        "control: saved active preset to disk",
    );
    ControlResponse::ok()
}

/// Persist the active preset under a new name. Inserts the new
/// preset into the in-memory `cfg.presets` mirror so a follow-up
/// `ListPresets` reflects the new entry immediately.
fn apply_save_preset_as_command(
    cfg: &mut FluxConfig,
    config_path: Option<&std::path::Path>,
    active_preset: &Preset,
    new_name: &str,
) -> ControlResponse {
    let Some(path) = config_path else {
        return ControlResponse::err(
            "daemon has no writable config path",
            Some(
                "restart the daemon with --config <PATH> or without --no-default-config to enable Save"
                    .into(),
            ),
        );
    };
    // Race with the in-memory mirror: a preset that already exists in
    // the running daemon (e.g. one defined in the TOML at startup, or
    // created earlier via another SavePresetAs) must not be silently
    // overwritten. `persist::save_preset_as` already checks the
    // on-disk file, but we mirror that check against the in-memory
    // state so the operator gets the same answer when the file was
    // edited externally between the load and this command.
    if cfg.presets.contains_key(new_name) {
        return ControlResponse::err(
            format!("preset '{new_name}' already exists"),
            Some("pick a different name or use Save to overwrite the active preset".into()),
        );
    }
    if let Err(e) = crate::persist::save_preset_as(path, new_name, active_preset) {
        return ControlResponse::err(
            format!("save-as failed: {e}"),
            Some(format!("path: {}", path.display())),
        );
    }
    cfg.presets
        .insert(new_name.to_string(), active_preset.clone());
    info!(
        path = %path.display(),
        preset = %new_name,
        "control: saved active preset under new name",
    );
    ControlResponse::ok()
}

/// Report the daemon's writable config path (or `null`) so the GUI
/// can show "Saves to: …" and grey out the Save button when the
/// daemon was started without a resolvable config target.
fn apply_config_path_command(config_path: Option<&std::path::Path>) -> ControlResponse {
    match config_path {
        Some(p) => ControlResponse::ok_with(serde_json::json!({ "path": p })),
        None => ControlResponse::ok_with(serde_json::Value::Null),
    }
}

/// Walk the three sub-chain registries and return their full metadata
/// inventory keyed by section. The GUI uses this on startup to build
/// param widgets without hard-coding which effects exist.
///
/// The response also includes a `build_features` array of strings
/// (e.g. `["ml", "image-fill"]`) so the GUI can distinguish a slim
/// daemon (which yields `post = []` and may omit `image_fill` from
/// the plane registry) from a full daemon that happens to have those
/// entries unconfigured. The JSON shape would otherwise be ambiguous.
///
/// Under a slim build (`--no-default-features`) `image_fill` is absent
/// from the plane registry and therefore from the response, which is
/// the correct behaviour: the GUI should only offer effects the daemon
/// can actually instantiate.
///
/// The payload is built once per process and cached in a file-local
/// `OnceLock<serde_json::Value>` — effect metadata is fully static,
/// and this handler runs on the hot worker thread between frames
/// (~33 ms budget at 30 fps).  Subsequent calls clone the cached
/// `Value` instead of rebuilding three `Vec<&EffectMetadata>` and
/// re-serialising them.  Exactly one cfg branch is active per build,
/// so the cache covers both ml and slim configurations.
fn apply_list_effects_command() -> ControlResponse {
    static CACHE: OnceLock<serde_json::Value> = OnceLock::new();
    let payload = CACHE.get_or_init(|| {
        use fluxframe_effects::{mask_effects, plane_effects};

        let mask: Vec<_> = mask_effects::default_registry().iter_metadata().collect();
        let background: Vec<_> = plane_effects::default_registry().iter_metadata().collect();
        // foreground and background currently share the same plane
        // registry; if that changes (e.g. foreground-only effects),
        // split the call site and update the GUI contract.
        let foreground = background.clone();
        // Post effects exist only when the `ml` feature is enabled
        // (the composite path itself is ml-gated). Under a slim build
        // the section is reported empty so the GUI knows not to offer
        // post-effect editing.
        #[cfg(feature = "ml")]
        let post: Vec<_> = fluxframe_effects::post_effects::default_registry()
            .iter_metadata()
            .collect();
        #[cfg(not(feature = "ml"))]
        let post: Vec<&'static fluxframe_core::EffectMetadata> = Vec::new();

        // Build-feature inventory: lets the GUI distinguish a slim
        // daemon from a full one without inferring it from the
        // (possibly empty) effect lists. `mut` is conditional because
        // under a fully-slim build no `push` is reachable.
        #[allow(
            unused_mut,
            reason = "slim build leaves `features` empty; mut is needed when any feature is enabled"
        )]
        let mut features: Vec<&'static str> = Vec::new();
        #[cfg(feature = "ml")]
        features.push("ml");
        #[cfg(feature = "image-fill")]
        features.push("image-fill");

        serde_json::json!({
            "mask": mask,
            "background": background,
            "foreground": foreground,
            "post": post,
            "build_features": features,
        })
    });
    ControlResponse::ok_with(payload.clone())
}

/// Apply `reload`. Re-reads the TOML file from disk, validates it,
/// re-binds the daemon's `cfg`, then rebuilds the currently-active
/// preset (looking up the same name in the new config). CLI
/// overrides are NOT re-applied — they were captured at startup and
/// remain in effect through `working_cfg`.
fn apply_reload_command(
    cfg: &mut FluxConfig,
    config_path: Option<&std::path::Path>,
    processing_ctx: &ProcessingContext,
    chain: &mut EffectChain,
    active_preset_name: &mut String,
    active_preset: &mut Preset,
) -> ControlResponse {
    let Some(path) = config_path else {
        return ControlResponse::err(
            "no config path provided at startup",
            Some("pass --config <PATH> on the command line to enable reload".into()),
        );
    };
    let new_cfg = match crate::config_merge::load(Some(path)) {
        Ok(c) => c,
        Err(e) => {
            return ControlResponse::err(
                format!("re-read failed: {e}"),
                Some(format!("path: {}", path.display())),
            );
        }
    };
    if let Err(e) = new_cfg.validate() {
        return ControlResponse::err(format!("validation failed: {e}"), None);
    }
    let Some(new_preset) = new_cfg.presets.get(active_preset_name) else {
        let available: Vec<&str> = new_cfg.presets.keys().map(String::as_str).collect();
        return ControlResponse::err(
            format!("active preset '{active_preset_name}' is gone from the reloaded config"),
            Some(format!(
                "available presets after reload: {}",
                available.join(", ")
            )),
        );
    };
    let mut new_chain = match preset::build_chain(active_preset_name, new_preset) {
        Ok(c) => c,
        Err(e) => return ControlResponse::err(format!("build chain failed: {e}"), None),
    };
    if let Err(e) = new_chain.prepare_all(processing_ctx) {
        return ControlResponse::err(
            format!("prepare on reloaded chain failed: {e}"),
            Some("the live chain is unchanged".into()),
        );
    }
    std::mem::swap(chain, &mut new_chain);
    if let Err(e) = new_chain.shutdown_all() {
        warn!(error = %e, "old chain shutdown reported an error during reload");
    }
    *active_preset = new_preset.clone();
    *cfg = new_cfg;
    info!(path = %path.display(), preset = %active_preset_name, "control: reloaded config from disk");
    ControlResponse::ok()
}

/// Apply `get_config { path }`. With `path = None`, serialise the
/// whole active preset; otherwise walk the dot-path into the JSON
/// representation and return that subtree.
fn apply_get_config_command(active_preset: &Preset, path: Option<&str>) -> ControlResponse {
    let full = match serde_json::to_value(active_preset) {
        Ok(v) => v,
        Err(e) => return ControlResponse::err(format!("serialise failed: {e}"), None),
    };
    let Some(path) = path else {
        return ControlResponse::ok_with(full);
    };
    // Walk dot-path into the serialised value. Components must exist
    // as object keys at each step.
    let mut current = &full;
    for component in path.split('.') {
        if component.is_empty() {
            return ControlResponse::err(format!("empty component in path '{path}'"), None);
        }
        match current.get(component) {
            Some(v) => current = v,
            None => {
                return ControlResponse::err(
                    format!("path '{path}' has no value at '{component}'"),
                    None,
                );
            }
        }
    }
    ControlResponse::ok_with(current.clone())
}

/// Apply a `set <path> <value>` command. Parses the dot-path,
/// converts the JSON value to TOML, merges the field into the active
/// preset's `per_effect[effect]` table, builds a fresh `RawEffectParams`
/// from the merged table, and calls into the chain's named-effect
/// reconfiguration.
fn apply_set_command(
    chain: &mut EffectChain,
    active_preset: &mut Preset,
    path: &str,
    value: serde_json::Value,
) -> ControlResponse {
    let set_path = match crate::control::commands::parse_set_path(path) {
        Ok(p) => p,
        Err(e) => return ControlResponse::err(e, None),
    };
    let toml_value = match crate::control::commands::json_to_toml(value) {
        Ok(v) => v,
        Err(e) => return ControlResponse::err(e, None),
    };
    // Snapshot whether the targeted section existed BEFORE we touch
    // the preset; the Err arm below uses this to revert any
    // materialisation we performed on behalf of a doomed Set.
    let was_absent = section_is_absent(active_preset, set_path.section);
    // Mask cannot be conjured (the composite needs an ONNX model that
    // the control protocol cannot synthesise) — return a typed error
    // pointing the operator at a preset with `[mask]`. Plane/Post
    // sections are materialised on demand via
    // `ensure_plane_or_post_section_mut`.
    if set_path.section == SubchainKind::Mask && was_absent {
        return ControlResponse::err(
            "preset does not have a [mask] sub-section",
            Some(
                "this preset has no segmentation; switch to a preset with [mask] to add post-process effects"
                    .into(),
            ),
        );
    }
    let section_mut = if set_path.section == SubchainKind::Mask {
        // Safe to unwrap: `was_absent == false` here (we returned above otherwise).
        let Some(s) = mask_section_mut(active_preset) else {
            debug_assert!(
                false,
                "mask section disappeared between absence check and access"
            );
            return ControlResponse::err(
                "internal: mask section missing",
                Some("this is a bug; please report".into()),
            );
        };
        s
    } else {
        ensure_plane_or_post_section_mut(active_preset, set_path.section)
    };
    // Read the existing per-effect table for this effect (defaulting
    // to empty), insert the new field, then hand the merged table to
    // the chain for reconfiguration. On failure we revert the in-memory
    // copy AND re-call reconfigure with the ORIGINAL params so the
    // live effect's `self.config` matches what we kept in memory —
    // important for effects with lazy reload (image_fill, blur) that
    // pick up the new config on the next `process()`.
    let original_entry = section_mut.per_effect.get(&set_path.effect).cloned();
    let mut effect_table = match original_entry.clone() {
        Some(toml::Value::Table(t)) => t,
        _ => toml::Table::new(),
    };
    effect_table.insert(set_path.field.clone(), toml_value);
    let merged_params: fluxframe_core::traits::RawEffectParams =
        toml::Value::Table(effect_table.clone());
    section_mut
        .per_effect
        .insert(set_path.effect.clone(), toml::Value::Table(effect_table));
    match chain.reconfigure_named_effect(set_path.section, &set_path.effect, merged_params) {
        Ok(()) => {
            info!(
                section = %set_path.section,
                effect = %set_path.effect,
                field = %set_path.field,
                "control: live-reconfigured effect"
            );
            ControlResponse::ok()
        }
        Err(e) => {
            tracing::debug!(
                section = %set_path.section,
                effect = %set_path.effect,
                field = %set_path.field,
                error = %e,
                "control: set rejected",
            );
            // Revert the in-memory copy.
            match &original_entry {
                Some(v) => {
                    section_mut
                        .per_effect
                        .insert(set_path.effect.clone(), v.clone());
                }
                None => {
                    section_mut.per_effect.remove(&set_path.effect);
                }
            }
            // Best-effort revert of the live effect: reconfigure with
            // the ORIGINAL params so an effect that already swapped
            // its `self.config` rolls back. Build a TOML table from
            // `original_entry` (or an empty one) and replay. If this
            // second call also fails, log a warn — we cannot do
            // better without tearing the chain down.
            let revert_params: fluxframe_core::traits::RawEffectParams = match original_entry {
                Some(toml::Value::Table(t)) => toml::Value::Table(t),
                _ => toml::Value::Table(toml::Table::new()),
            };
            if let Err(revert_err) =
                chain.reconfigure_named_effect(set_path.section, &set_path.effect, revert_params)
            {
                warn!(
                    section = %set_path.section,
                    effect = %set_path.effect,
                    error = %revert_err,
                    "control: revert reconfigure also failed; live effect may be in an inconsistent state",
                );
            }
            // If we materialised the section purely to host this doomed
            // Set, undo the materialisation. We only do this when the
            // section is now empty (no chain, no per_effect entries
            // besides the one we just removed) — a non-empty section
            // means someone else populated it concurrently or the Set
            // partially succeeded, and we should leave it alone.
            revert_absent_section_if_empty(active_preset, set_path.section, was_absent);
            let (reason, hint) = match &e {
                fluxframe_core::EffectError::InvalidConfig { reason, hint, .. } => {
                    (reason.clone(), hint.clone())
                }
                other => (format!("{other}"), None),
            };
            ControlResponse::err(reason, hint)
        }
    }
}

/// Mutable handle to the `[mask]` sub-section of `preset`. Returns
/// `None` when the preset declares no mask — the mask section is the
/// only sub-section that **cannot** be materialised on demand because
/// the composite requires an ONNX model that the control protocol
/// has no way to synthesise.
fn mask_section_mut(preset: &mut Preset) -> Option<&mut fluxframe_core::PipelineSection> {
    preset.mask.as_mut()
}

/// Mutable handle to a plane (Background/Foreground) or Post
/// sub-section, creating it from `PipelineSection::default()` if
/// absent. Used by [`apply_set_chain_command`] so the control protocol
/// can extend a preset with a section it did not originally declare in
/// TOML (e.g. add a `[post]` chain to a preset that only had
/// `[mask]` + `[background]`).
///
/// **Caller MUST be prepared to revert the materialisation** on
/// subsequent failure (e.g. effect-name typo, prepare error). See
/// [`apply_set_chain_command`] for the snapshot/revert pattern.
///
/// `tracing::info!` fires when a new section is created so the
/// operator can correlate "preset now has [post]" with the originating
/// command.
fn ensure_plane_or_post_section_mut(
    preset: &mut Preset,
    section: SubchainKind,
) -> &mut fluxframe_core::PipelineSection {
    let slot: &mut Option<fluxframe_core::PipelineSection> = match section {
        SubchainKind::Background => &mut preset.background,
        SubchainKind::Foreground => &mut preset.foreground,
        SubchainKind::Post => &mut preset.post,
        SubchainKind::Mask => {
            // `Mask` is filtered out by the `PlaneOrPostKind` typestate
            // approach we considered, but for now a runtime check with
            // debug_assert! suffices — callers that hit this arm are
            // programming errors, not user-facing.
            debug_assert!(
                false,
                "ensure_plane_or_post_section_mut called with Mask — use mask_section_mut"
            );
            // Fall back to mask in release builds; this restores the
            // pre-fix behaviour (silent None propagation) rather than
            // panicking on operators.
            return preset.mask.get_or_insert_with(|| {
                tracing::info!(
                    section = "mask",
                    "materialised absent mask section on demand (fallback)"
                );
                fluxframe_core::PipelineSection::default()
            });
        }
    };
    if slot.is_none() {
        tracing::info!(section = %section, "materialised absent sub-section on demand");
    }
    slot.get_or_insert_with(Default::default)
}

/// Apply `set_chain <section> [...]`. Rebuilds the named sub-chain
/// against the live registry, configures each effect with the
/// matching `per_effect` payload from the active preset (defaults
/// when absent), prepares them against the composite's stored
/// `ProcessingContext`, and swaps them in.
///
/// Available only with the `ml` feature: `set_chain` rebuilds the
/// composite's sub-chains, which depend on the composite effect (and
/// the post/mask/plane registries that the composite owns).
#[cfg(feature = "ml")]
fn apply_set_chain_command(
    chain: &mut EffectChain,
    active_preset: &mut Preset,
    section: SubchainKind,
    new_names: &[String],
) -> ControlResponse {
    let Some(composite) = chain.composite_mut() else {
        return ControlResponse::err(
            "active chain has no composite (preset has no [mask] section)",
            Some(
                "this preset has no segmentation; switch to a preset with [mask] to add post-process effects"
                    .into(),
            ),
        );
    };
    // Snapshot whether the targeted section existed BEFORE we touch
    // the preset; failures below revert any materialisation we did.
    let was_absent = section_is_absent(active_preset, section);
    // Mask cannot be materialised on demand (the composite is bound to
    // a model that the control protocol cannot synthesise). The
    // `composite_mut()` guard above ensures the composite exists, which
    // implies the original preset had `[mask]`. If the active preset
    // somehow lacks it, that's an internal inconsistency.
    let section_data = if section == SubchainKind::Mask {
        let Some(s) = mask_section_mut(active_preset) else {
            debug_assert!(
                false,
                "internal: composite present but active preset has no [mask] section"
            );
            return ControlResponse::err(
                "internal: composite present but mask section missing",
                Some("this is a bug; please report".into()),
            );
        };
        s
    } else {
        ensure_plane_or_post_section_mut(active_preset, section)
    };
    // Build new effects + configure them. The old `per_effect` payload
    // is reused when present; otherwise an empty TOML table feeds
    // the effect its serde defaults.
    let result = build_and_configure_subchain(section, new_names, &section_data.per_effect);
    let payload = match result {
        Ok(o) => o,
        Err(e) => {
            // Revert the materialisation if we created an empty section
            // purely to host this doomed `set_chain`.
            revert_absent_section_if_empty(active_preset, section, was_absent);
            return ControlResponse::err(e.reason, e.hint);
        }
    };
    // Hand the new chain to the composite. `replace_subchain` runs
    // `prepare()` against the stored ProcessingContext and atomically
    // swaps. On failure, the old chain stays intact and we don't
    // touch active_preset.
    match composite.replace_subchain(payload) {
        Ok(()) => {
            // Re-borrow because the previous `section_data` borrow was
            // released when we handed `payload` to the composite.
            let section_data = match section {
                SubchainKind::Mask => mask_section_mut(active_preset)
                    .expect("mask presence checked above and not removed since"),
                SubchainKind::Background | SubchainKind::Foreground | SubchainKind::Post => {
                    ensure_plane_or_post_section_mut(active_preset, section)
                }
            };
            section_data.chain = new_names.to_vec();
            // Prune per-effect tables for effects no longer in the
            // chain. Without this, GUI flow "add → tune → remove"
            // would leave an orphan `[<section>.<effect>]` table in
            // `active_preset`, which then (a) survives Save into the
            // TOML file, (b) trips the composite builder's
            // `reject_unknown_table_keys` validation next time the
            // preset is loaded (the operator sees "sub-table has no
            // matching entry in chain"). Reserved keys (`model`,
            // `model_config`, `fallback_threshold`) are not effect
            // names but the `Set` handler could conceivably put them
            // into `per_effect`; keep them so a future Set on a mask
            // sub-section stays intact.
            section_data.per_effect.retain(|key, _| {
                new_names.iter().any(|n| n == key) || is_reserved_per_effect_key(key)
            });
            info!(section = %section, chain = ?new_names, "control: replaced sub-chain");
            ControlResponse::ok()
        }
        Err(e) => {
            revert_absent_section_if_empty(active_preset, section, was_absent);
            ControlResponse::err(
                format!("prepare on new chain failed: {e}"),
                Some("the previous chain is still active".into()),
            )
        }
    }
}

/// Mirror of `composite::builder::PIPELINE_RESERVED_KEYS`: keys that
/// are NOT effect names but may legitimately appear in `per_effect`
/// (e.g. `model` for the mask section). Pruning logic in
/// [`apply_set_chain_command`] keeps these even when they are not in
/// the active chain so a future Set on a mask field is preserved.
fn is_reserved_per_effect_key(key: &str) -> bool {
    matches!(
        key,
        "chain" | "model" | "model_config" | "fallback_threshold"
    )
}

/// Whether the given sub-section of `preset` is currently `None`.
/// Snapshot helper used by [`apply_set_command`] and
/// [`apply_set_chain_command`] before any code path that might
/// materialise the section via [`ensure_plane_or_post_section_mut`].
fn section_is_absent(preset: &Preset, section: SubchainKind) -> bool {
    match section {
        SubchainKind::Mask => preset.mask.is_none(),
        SubchainKind::Background => preset.background.is_none(),
        SubchainKind::Foreground => preset.foreground.is_none(),
        SubchainKind::Post => preset.post.is_none(),
    }
}

/// If `was_absent` is `true` AND the target section is now empty
/// (no chain entries, no per-effect overrides), revert it to `None`.
/// Used by [`apply_set_chain_command`] and [`apply_set_command`] to
/// undo a section materialisation when the command that triggered it
/// fails — keeps `active_preset` in sync with the TOML's structural
/// shape so the operator does not see a phantom `[post]` they never
/// created.
fn revert_absent_section_if_empty(
    active_preset: &mut Preset,
    section: SubchainKind,
    was_absent: bool,
) {
    if !was_absent {
        return;
    }
    let now_empty = match section {
        SubchainKind::Mask => active_preset
            .mask
            .as_ref()
            .is_some_and(|s| s.chain.is_empty() && s.per_effect.is_empty()),
        SubchainKind::Background => active_preset
            .background
            .as_ref()
            .is_some_and(|s| s.chain.is_empty() && s.per_effect.is_empty()),
        SubchainKind::Foreground => active_preset
            .foreground
            .as_ref()
            .is_some_and(|s| s.chain.is_empty() && s.per_effect.is_empty()),
        SubchainKind::Post => active_preset
            .post
            .as_ref()
            .is_some_and(|s| s.chain.is_empty() && s.per_effect.is_empty()),
    };
    if now_empty {
        match section {
            SubchainKind::Mask => active_preset.mask = None,
            SubchainKind::Background => active_preset.background = None,
            SubchainKind::Foreground => active_preset.foreground = None,
            SubchainKind::Post => active_preset.post = None,
        }
    }
}

/// Slim-build stub for `set_chain`: the post/mask/plane registries
/// and the composite effect are all `ml`-gated, so this command is
/// unavailable without `--features ml`.
#[cfg(not(feature = "ml"))]
fn apply_set_chain_command(
    _chain: &mut EffectChain,
    _active_preset: &mut Preset,
    _section: SubchainKind,
    _new_names: &[String],
) -> ControlResponse {
    ControlResponse::err(
        "set_chain unavailable: built without ml feature",
        Some("rebuild with `--features fluxframe-cli/ml` to enable composite live-reconfig".into()),
    )
}

/// Sub-error type the chain-build path can surface.
#[cfg(feature = "ml")]
struct SubChainError {
    reason: String,
    hint: Option<String>,
}

/// Resolve effect names against the right registry and configure each
/// with the matching `per_effect` table. Mirrors what
/// `CompositeBuilder` does at construction time, but exposed here so
/// the worker can rebuild a single sub-chain at runtime.
#[cfg(feature = "ml")]
#[allow(
    clippy::too_many_lines,
    reason = "four enum arms × build+configure pattern; trait-object types differ so each arm is its own monomorphisation"
)]
fn build_and_configure_subchain(
    section: SubchainKind,
    names: &[String],
    per_effect: &std::collections::BTreeMap<String, toml::Value>,
) -> Result<fluxframe_effects::composite::SubChainPayload, SubChainError> {
    use fluxframe_effects::composite::SubChainPayload;
    use fluxframe_effects::{mask_effects, plane_effects, post_effects};

    /// Local helper: trait object with a `configure()` method, used
    /// to drive each effect's `configure()` after registry build.
    /// Implemented in this crate for the three sub-effect dyn types
    /// because the unified `SubEffectAdapter` lives `pub(crate)` in
    /// `fluxframe-effects` and is not exposed here.
    trait Configure {
        fn configure_mut(
            &mut self,
            params: fluxframe_core::traits::RawEffectParams,
        ) -> Result<(), fluxframe_core::EffectError>;
    }
    impl Configure for dyn fluxframe_core::MaskEffect {
        fn configure_mut(
            &mut self,
            params: fluxframe_core::traits::RawEffectParams,
        ) -> Result<(), fluxframe_core::EffectError> {
            self.configure(params)
        }
    }
    impl Configure for dyn fluxframe_core::PlaneEffect {
        fn configure_mut(
            &mut self,
            params: fluxframe_core::traits::RawEffectParams,
        ) -> Result<(), fluxframe_core::EffectError> {
            self.configure(params)
        }
    }
    impl Configure for dyn fluxframe_core::PostEffect {
        fn configure_mut(
            &mut self,
            params: fluxframe_core::traits::RawEffectParams,
        ) -> Result<(), fluxframe_core::EffectError> {
            self.configure(params)
        }
    }

    /// Synthesise a default TOML config for an effect that has no
    /// explicit `per_effect` block in the incoming preset.
    ///
    /// Branches:
    /// - metadata present + every required param has a default → emit a
    ///   populated [`toml::Table`].
    /// - metadata present but at least one required param has no
    ///   default → return [`EffectError::InvalidConfig`] naming the
    ///   missing field(s) verbatim so the GUI / TUI can surface them
    ///   to the user (instead of the cryptic serde error that would
    ///   otherwise come out of `configure`).
    /// - metadata absent → programming error: registering an effect
    ///   without `pub const METADATA: EffectMetadata` is forbidden;
    ///   return an explicit `InvalidConfig` rather than silently
    ///   passing an empty table down to `configure` (which is the
    ///   exact failure mode this whole helper was added to eliminate).
    fn synthesise_default(
        name: &str,
        lookup_metadata: impl Fn(&str) -> Option<&'static fluxframe_core::EffectMetadata>,
    ) -> Result<toml::Value, fluxframe_core::EffectError> {
        match lookup_metadata(name).map(fluxframe_core::EffectMetadata::default_config) {
            Some(Ok(table)) => Ok(toml::Value::Table(table)),
            Some(Err(missing)) => Err(fluxframe_core::EffectError::InvalidConfig {
                // `name` is guaranteed by `build_chain` to be a registered
                // snake_case literal; safe to surface in error/log messages.
                name: name.to_string(),
                reason: format!(
                    "missing required field(s) {}; set them via the parameter editor \
                     (or in fluxframe.toml) before adding this effect to a chain",
                    missing
                        .iter()
                        .map(|f| format!("`{f}`"))
                        .collect::<Vec<_>>()
                        .join(", "),
                ),
                hint: Some(format!(
                    "set the following field(s) on `{name}` explicitly, \
                     e.g. via the parameter editor: {}",
                    missing
                        .iter()
                        .map(|f| format!("`{f}`"))
                        .collect::<Vec<_>>()
                        .join(", "),
                )),
            }),
            None => Err(fluxframe_core::EffectError::InvalidConfig {
                // `name` is guaranteed by `build_chain` to be a registered
                // snake_case literal; safe to surface in error/log messages.
                name: name.to_string(),
                reason: "effect is registered without metadata; cannot synthesise defaults".into(),
                hint: Some(
                    "this is a programming error — every effect must declare \
                     `pub const METADATA: EffectMetadata`. Please file a bug."
                        .into(),
                ),
            }),
        }
    }

    fn configure_each<E>(
        chain: &mut [Box<E>],
        names: &[String],
        per_effect: &std::collections::BTreeMap<String, toml::Value>,
        lookup_metadata: impl Fn(&str) -> Option<&'static fluxframe_core::EffectMetadata>,
    ) -> Result<(), fluxframe_core::EffectError>
    where
        E: ?Sized + Configure,
    {
        for (effect, name) in chain.iter_mut().zip(names.iter()) {
            // No explicit per_effect block — happens when a chain
            // entry was added via the GUI (`Command::SetChain`
            // with just names). Synthesise a default config from
            // the effect's metadata so we don't hand `configure`
            // an empty table that fails for required-field
            // structs (color_fill.rgb, image_fill.path).
            let params = if let Some(v) = per_effect.get(name).cloned() {
                v
            } else {
                tracing::debug!(effect = %name, "synthesising default config from metadata");
                synthesise_default(name, &lookup_metadata)?
            };
            effect.configure_mut(params)?;
        }
        Ok(())
    }
    match section {
        SubchainKind::Mask => {
            let registry = mask_effects::default_registry();
            let mut built = registry.build_chain(names).map_err(|e| SubChainError {
                reason: format!("mask chain build failed: {e}"),
                hint: None,
            })?;
            configure_each::<dyn fluxframe_core::MaskEffect>(&mut built, names, per_effect, |n| {
                registry.metadata(n)
            })
            .map_err(|e| match &e {
                fluxframe_core::EffectError::InvalidConfig { reason, hint, .. } => SubChainError {
                    reason: reason.clone(),
                    hint: hint.clone(),
                },
                _ => SubChainError {
                    reason: format!("configure failed: {e}"),
                    hint: None,
                },
            })?;
            Ok(SubChainPayload::Mask(built))
        }
        SubchainKind::Background => {
            let registry = plane_effects::default_registry();
            let mut built = registry.build_chain(names).map_err(|e| SubChainError {
                reason: format!("background chain build failed: {e}"),
                hint: None,
            })?;
            configure_each::<dyn fluxframe_core::PlaneEffect>(&mut built, names, per_effect, |n| {
                registry.metadata(n)
            })
            .map_err(|e| match &e {
                fluxframe_core::EffectError::InvalidConfig { reason, hint, .. } => SubChainError {
                    reason: reason.clone(),
                    hint: hint.clone(),
                },
                _ => SubChainError {
                    reason: format!("configure failed: {e}"),
                    hint: None,
                },
            })?;
            Ok(SubChainPayload::Background(built))
        }
        SubchainKind::Foreground => {
            let registry = plane_effects::default_registry();
            let mut built = registry.build_chain(names).map_err(|e| SubChainError {
                reason: format!("foreground chain build failed: {e}"),
                hint: None,
            })?;
            configure_each::<dyn fluxframe_core::PlaneEffect>(&mut built, names, per_effect, |n| {
                registry.metadata(n)
            })
            .map_err(|e| match &e {
                fluxframe_core::EffectError::InvalidConfig { reason, hint, .. } => SubChainError {
                    reason: reason.clone(),
                    hint: hint.clone(),
                },
                _ => SubChainError {
                    reason: format!("configure failed: {e}"),
                    hint: None,
                },
            })?;
            Ok(SubChainPayload::Foreground(built))
        }
        SubchainKind::Post => {
            let registry = post_effects::default_registry();
            let mut built = registry.build_chain(names).map_err(|e| SubChainError {
                reason: format!("post chain build failed: {e}"),
                hint: None,
            })?;
            configure_each::<dyn fluxframe_core::PostEffect>(&mut built, names, per_effect, |n| {
                registry.metadata(n)
            })
            .map_err(|e| match &e {
                fluxframe_core::EffectError::InvalidConfig { reason, hint, .. } => SubChainError {
                    reason: reason.clone(),
                    hint: hint.clone(),
                },
                _ => SubChainError {
                    reason: format!("configure failed: {e}"),
                    hint: None,
                },
            })?;
            Ok(SubChainPayload::Post(built))
        }
    }
}

/// How long the worker waits on an empty frame slot before re-checking the
/// shutdown flag.  Short enough to be responsive to Ctrl-C, long enough to
/// avoid spinning when the source briefly stalls.
const WORKER_POLL_TIMEOUT: Duration = Duration::from_millis(50);

/// Reconcile slot drop counter into [`RuntimeMetrics`] every Nth frame.
/// Picked low enough that a brief burst surfaces within a second at
/// 30 fps; high enough to keep the supervisor loop free of per-frame
/// atomic chatter on the slot's counter.
const DROPPED_SYNC_INTERVAL: u64 = 30;

/// Process-wide registry of live [`RunToken`]s.  The Ctrl-C signal handler
/// walks this list and broadcasts a shutdown request to every concurrent
/// run.  Tokens are stored as [`Weak`] references so a run that has
/// already torn down does not keep its slot alive.
static REGISTERED_TOKENS: OnceLock<Mutex<Vec<Weak<RunToken>>>> = OnceLock::new();

/// Process-wide shutdown flag.  Set to `true` by the Ctrl-C handler;
/// drained by [`is_shutdown_requested`] / [`wait_for_shutdown`] so
/// non-pipeline polling loops (e.g. the `device = "auto"` wait state)
/// can break out promptly.  Distinct from per-run `RunToken::flag`
/// because the auto wait loop lives OUTSIDE any active run.
///
/// Uses [`Ordering::Relaxed`] because this flag carries a single bit
/// with no companion data — the polling loops only need eventual
/// visibility, not happens-before with any other variable. Per-run
/// `RunToken::flag` keeps Acquire/Release because it is read alongside
/// `slot.close()` and must order frame-slot tear-down.
///
/// TODO(post-MVP): unify SHUTDOWN_REQUESTED into the tokens_registry by
/// adding an idle/global token, so the auto wait loop and the run loop
/// share one shutdown surface. The current two-tier split is honest
/// (each loop only reads what it owns) but doubles the bookkeeping.
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

fn tokens_registry() -> &'static Mutex<Vec<Weak<RunToken>>> {
    REGISTERED_TOKENS.get_or_init(|| Mutex::new(Vec::new()))
}

// ---------------------------------------------------------------------------
// Loopback visibility tracking
// ---------------------------------------------------------------------------
//
// How long `/dev/video10` had no producer holding it open. With
// `exclusive_caps=1` the node only advertises CAPTURE capabilities while
// a producer is attached, so this window is exactly the time a client
// enumerating devices does not see the camera at all — the operator-
// visible symptom behind "the app can't find FluxFrame, restart it".
//
// It is deliberately process-global rather than part of `RuntimeMetrics`:
// the window lives *between* runs, and a per-run metrics bundle is
// created after the output is already back up. A storm of 164 restarts
// would have reset a per-run counter 164 times and reported zero.

/// Wall-clock ms accumulated across every completed invisible window.
static OUTPUT_INVISIBLE_MS_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Number of times the output pipeline was (re)built from scratch.
static OUTPUT_REBUILDS_TOTAL: AtomicU64 = AtomicU64::new(0);
/// Monotonic ms since process start marking when the output went down,
/// or `0` when the output is currently live. Stored as a scalar rather
/// than an `Instant` so it can live in a plain atomic.
static OUTPUT_DOWN_SINCE_MS: AtomicU64 = AtomicU64::new(0);

/// Process start, the epoch for [`OUTPUT_DOWN_SINCE_MS`].
fn process_epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

fn elapsed_ms_since_epoch() -> u64 {
    u64::try_from(process_epoch().elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Record that the loopback output is live and advertising CAPTURE caps.
/// Closes any open invisible window and logs its duration.
pub(crate) fn mark_output_live() {
    OUTPUT_REBUILDS_TOTAL.fetch_add(1, Ordering::Relaxed);
    let since = OUTPUT_DOWN_SINCE_MS.swap(0, Ordering::Relaxed);
    if since == 0 {
        // First start of the process — there was no preceding window.
        return;
    }
    let down_ms = elapsed_ms_since_epoch().saturating_sub(since);
    OUTPUT_INVISIBLE_MS_TOTAL.fetch_add(down_ms, Ordering::Relaxed);
    info!(
        down_ms,
        invisible_ms_total = OUTPUT_INVISIBLE_MS_TOTAL.load(Ordering::Relaxed),
        rebuilds_total = OUTPUT_REBUILDS_TOTAL.load(Ordering::Relaxed),
        "loopback output back up — clients can enumerate the camera again"
    );
}

/// Record that the loopback output has been torn down, starting an
/// invisible window. `reason` is the error (or `None` for a clean exit)
/// that ended the run.
pub(crate) fn mark_output_down(reason: Option<&FluxError>) {
    // Keep the earliest down-edge if this is called twice without an
    // intervening `mark_output_live` — the window is "since the output
    // last worked", not "since the last teardown".
    let now = elapsed_ms_since_epoch().max(1);
    let _ = OUTPUT_DOWN_SINCE_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
    if let Some(e) = reason {
        warn!(error = %e, "loopback output torn down — camera unavailable to clients");
    } else {
        info!("loopback output torn down (clean exit)");
    }
}

/// Returns `true` if Ctrl-C has been received since process start.
#[must_use]
pub(crate) fn is_shutdown_requested() -> bool {
    SHUTDOWN_REQUESTED.load(Ordering::Relaxed)
}

/// Sleep for `total` or until shutdown is requested, whichever comes
/// first.  Returns `true` if the sleep was cut short by a shutdown
/// request.  Polls every ~100 ms so a Ctrl-C during a long wait is
/// reflected promptly.
pub(crate) fn wait_for_shutdown(total: Duration) -> bool {
    let deadline = Instant::now() + total;
    while Instant::now() < deadline {
        if is_shutdown_requested() {
            return true;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        std::thread::sleep(remaining.min(Duration::from_millis(100)));
    }
    is_shutdown_requested()
}

/// Make sure the process-wide Ctrl-C handler is installed.  Idempotent
/// — call from anywhere that needs the handler before opening a slot
/// (currently: per-run setup and the `device = "auto"` wait loop).
pub(crate) fn ensure_ctrlc_handler() {
    install_ctrlc_once();
}

/// Per-run shutdown state.  Held inside an [`Arc`] so the signal handler
/// can flip the flag and close the slot without owning the run.
struct RunToken {
    /// Shutdown flag: `true` while the run is active, set to `false` by
    /// the signal handler or the supervisor itself to request termination.
    flag: Arc<AtomicBool>,
    /// Frame slot to close when shutdown is requested, so the worker
    /// loop wakes from `recv_timeout` immediately.
    slot: LatestFrameSlot,
}

/// Install the Ctrl-C handler at most once for the entire process.  If the
/// handler cannot be installed (e.g. another component already claimed it)
/// we log a warning and continue without one: tests routinely run inside
/// harnesses that grab SIGINT for themselves.
fn install_ctrlc_once() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let result = ctrlc::set_handler(|| {
            info!("Ctrl-C received - broadcasting shutdown");
            SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);
            let mut reg = tokens_registry()
                .lock()
                .expect("tokens registry poisoned");
            reg.retain(|w| w.upgrade().is_some());
            for token in reg.iter().filter_map(Weak::upgrade) {
                token.flag.store(false, Ordering::Release);
                token.slot.close();
            }
        });
        if let Err(e) = result {
            warn!(
                error = %e,
                "could not install Ctrl-C handler (another one is already set); continuing without one",
            );
        }
    });
}

/// RAII guard that owns a [`RunToken`] for the duration of a single run.
///
/// On drop the guard removes its token's weak reference from the global
/// registry, which keeps the registry from growing unboundedly when many
/// runs come and go inside the same process (the integration test suite
/// is the obvious caller).
struct TokenGuard {
    token: Arc<RunToken>,
}

impl Drop for TokenGuard {
    fn drop(&mut self) {
        let reg = tokens_registry();
        let mut tokens = reg.lock().expect("tokens registry poisoned");
        tokens.retain(|w| {
            w.upgrade()
                .is_some_and(|other| !Arc::ptr_eq(&other, &self.token))
        });
    }
}

/// Register a fresh [`RunToken`] for the current run and return a guard
/// plus the per-run shutdown flag.  The guard must remain in scope until
/// the run finishes; dropping it removes the token from the registry.
fn register_token(slot: LatestFrameSlot) -> (TokenGuard, Arc<AtomicBool>) {
    install_ctrlc_once();
    let token = Arc::new(RunToken {
        flag: Arc::new(AtomicBool::new(true)),
        slot,
    });
    let flag = Arc::clone(&token.flag);
    let weak = Arc::downgrade(&token);
    tokens_registry()
        .lock()
        .expect("tokens registry poisoned")
        .push(weak);
    (TokenGuard { token }, flag)
}

/// Parsed input source resolved from a [`FluxConfig`].
///
/// Routing rules:
///
/// * [`InputDevice::Testsrc`] → [`Self::Testsrc`].
/// * [`InputDevice::Path`]    → [`Self::V4l2`] (path is *not*
///   canonicalised here — that happens inside
///   [`fluxframe_gst::input::InputPipeline::build_v4l2`]).
/// * [`InputDevice::Auto`]    → [`Self::Unsupported`]; `commands::run`
///   intercepts `Auto` and dispatches to the polling loop before this
///   function is called, so reaching `classify_input` with `Auto` is a
///   bug in the caller.
///
/// Adding a variant turns every CLI match site into a compile error,
/// which is what we want at the `pub(crate)` boundary; `#[non_exhaustive]`
/// would only matter cross-crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InputSpec {
    /// Synthetic test source (`videotestsrc`).
    Testsrc,
    /// V4L2 capture device (path is *not* canonicalised here — that
    /// happens inside [`fluxframe_gst::input::InputPipeline::build_v4l2`]).
    V4l2(PathBuf),
    /// Unrecognised — caller surfaces a structured error.
    Unsupported(String),
}

/// Classify the input section of `cfg` into an [`InputSpec`].
///
/// [`InputDevice::Auto`] should never reach this function: the
/// `commands::run` entry point intercepts it and routes the run through
/// the polling loop (`run_auto`) which substitutes the picked path back
/// into `cfg.input.device` before calling here. Returning
/// `InputSpec::Unsupported` is the safe escape hatch.
#[must_use]
pub(crate) fn classify_input(cfg: &FluxConfig) -> InputSpec {
    match &cfg.input.device {
        InputDevice::Testsrc => InputSpec::Testsrc,
        InputDevice::Path(p) => InputSpec::V4l2(p.clone()),
        InputDevice::Auto => InputSpec::Unsupported(
            "auto (should have been handled in commands::run before classify_input)".into(),
        ),
    }
}

/// Parsed output sink resolved from a [`FluxConfig`].
///
/// Mirrors [`InputSpec`] on the output side: a typed enum collapses the
/// `device == "auto" | "fakesink" | /dev/...` triage into one match site
/// so `commands::check` and `commands::run` cannot drift apart.
///
/// Adding a variant turns every CLI match site into a compile error,
/// which is what we want at the `pub(crate)` boundary; `#[non_exhaustive]`
/// would only matter cross-crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OutputSpec {
    /// `autovideosink` (manual glance verification).
    Auto,
    /// `fakesink` (CI / dev smoke without a loopback).
    Fake,
    /// `v4l2sink` to a loopback (or other writable V4L2) device.
    V4l2(PathBuf),
    /// `pipewiresink` publishing a PipeWire node.  Optional name shows
    /// up in PipeWire graph tools (`pw-cli`, `helvum`).
    Pipewire(Option<String>),
    /// Unrecognised — caller surfaces a structured error.
    Unsupported(String),
}

/// Classify the output section of `cfg` into an [`OutputSpec`].
///
/// Recognised forms:
///   * `auto` / `fakesink` — built-in GStreamer sinks.
///   * `/dev/video*` — v4l2loopback (or any writable V4L2 device).
///   * `pipewire` / `pipewire:<node-name>` — PipeWire stream.  The
///     optional name suffix labels the PipeWire client in graph tools
///     and lets multiple FluxFrame instances co-exist.
#[must_use]
pub(crate) fn classify_output(cfg: &FluxConfig) -> OutputSpec {
    let device = cfg.output.device.as_str();
    if let Some(suffix) = device.strip_prefix("pipewire") {
        if suffix.is_empty() {
            return OutputSpec::Pipewire(None);
        }
        if let Some(rest) = suffix.strip_prefix(':') {
            let trimmed = rest.trim();
            return OutputSpec::Pipewire(if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            });
        }
        // `pipewireFOO` etc — neither bare nor `pipewire:NAME`.  Fall
        // through to Unsupported so the operator sees a clear error.
    }
    match device {
        "auto" => OutputSpec::Auto,
        "fakesink" => OutputSpec::Fake,
        other if other.starts_with("/dev/") => OutputSpec::V4l2(PathBuf::from(other)),
        other => OutputSpec::Unsupported(other.to_string()),
    }
}

/// Run a passthrough-style chain on `videotestsrc`, emitting to the sink
/// resolved from the configuration.
///
/// # Errors
///
/// Propagates [`FluxError`] from pipeline construction, effect chain
/// preparation, or runtime failures.
#[tracing::instrument(skip_all, fields(sink = ?cfg.output.device, preset = preset_name))]
pub(crate) fn run_testsrc_chain(
    cfg: &FluxConfig,
    preset_name: &str,
    config_path: Option<&std::path::Path>,
    chain: EffectChain,
) -> Result<(), FluxError> {
    run_chain(
        cfg,
        preset_name,
        config_path,
        chain,
        "testsrc",
        InputPipeline::build_testsrc,
    )
}

/// Run the effect chain against a V4L2 capture device.
///
/// # Errors
///
/// Propagates [`FluxError`] from pipeline construction (including a
/// structured [`PipelineError::InputDeviceUnavailable`] when the
/// pre-open check fails), effect chain preparation, or runtime failures.
#[tracing::instrument(skip_all, fields(device = %cfg.input.device, sink = ?cfg.output.device, preset = preset_name))]
pub(crate) fn run_v4l2_chain(
    cfg: &FluxConfig,
    preset_name: &str,
    config_path: Option<&std::path::Path>,
    chain: EffectChain,
) -> Result<(), FluxError> {
    // Detect the camera's native fps so the operator's `[input] fps`
    // hint becomes an upper bound, not a hard requirement: requesting
    // 30 fps from a 15 fps camera silently capped to 15 is friendlier
    // than failing the pipeline.
    //
    // Width/height are NOT overridden — operator's `[input]
    // width/height` is the source of truth and propagates through to
    // the output dimensions (which v4l2loopback then exposes to
    // consumers).  Doing it the other way (cfg follows camera) made
    // the output size change every time the camera changed, which
    // broke the v4l2loopback `VIDIOC_S_FMT` negotiation on switch
    // and made the consumer-visible resolution non-deterministic.
    // `videoscale` in the input pipeline pads with `add-borders=true`
    // so the aspect ratio is preserved across the resize.
    let device_path = match &cfg.input.device {
        InputDevice::Path(p) => p.clone(),
        // run_v4l2_chain is only reachable from the V4l2 branch of
        // classify_input, which extracts a path. Auto / Testsrc would
        // never get here, but surface a structured error rather than
        // panic if someone wires this up incorrectly later.
        other => {
            return Err(FluxError::Config {
                reason: format!(
                    "run_v4l2_chain invoked with non-path input device '{other}' (this is a CLI \
                     dispatch bug)",
                ),
                hint: None,
            });
        }
    };
    // Stage 16: detection now lives INSIDE the input builder so it runs
    // on every (re)acquire attempt — the camera may be busy at startup
    // and become free later. `detect_native_mode` opens the device and
    // drops its `v4l::Device` before `build_v4l2` re-opens it via
    // `v4l2src`, so there is no VIDIOC_REQBUFS race within a single
    // attempt. Output fps comes from `cfg.input.fps`; only the *input*
    // adopts the detected native fps (width/height stay operator-set).
    let device_path_for_builder = device_path.clone();
    run_chain(
        cfg,
        preset_name,
        config_path,
        chain,
        "v4l2src",
        move |params| {
            let detected =
                fluxframe_gst::v4l2_caps::detect_native_mode(&device_path_for_builder, params.fps)?;
            let resolved =
                InputParams::new(params.width, params.height, detected.fps, params.format);
            InputPipeline::build_v4l2(&device_path_for_builder, resolved)
        },
    )
}

/// Shared driver for all input backends: takes a closure that builds the
/// input pipeline so the bus-listener / processing-loop / teardown plumbing
/// lives in exactly one place.
fn run_chain<F>(
    cfg: &FluxConfig,
    preset_name: &str,
    config_path: Option<&std::path::Path>,
    mut chain: EffectChain,
    source_label: &'static str,
    input_builder: F,
) -> Result<(), FluxError>
where
    F: Fn(InputParams) -> Result<InputPipeline, PipelineError>,
{
    fluxframe_gst::init()?;

    // Build metrics BEFORE `prepare_all` so the runtime can hand the
    // shared `Arc<Counters>` to effects through `ProcessingContext`.
    // This lets, e.g., the Stage 7 sticky-fallback decorator inside
    // `BackgroundBlurEffect`'s blur backend publish transition events
    // through the same counter bundle the supervisor reads later.
    let metrics = RuntimeMetrics::new();
    // Publish the effective effect-processing pool size (set once by
    // `cap_rayon_pool` at startup) as a gauge so operators can confirm
    // the cap and watch for oversubscription against `inference_p95`.
    metrics
        .counters
        .set_processing_threads(rayon::current_num_threads() as u64);

    // Stage 16: build + start the output (loopback) producer FIRST, so
    // `/dev/video10` is enumerable by Chrome and streams a placeholder
    // even while the camera is busy/absent. The input is acquired
    // afterwards (with backoff on the supervised path).
    let (output, mut processing_ctx, sink_label) = build_output(cfg)?;
    processing_ctx.counters = Some(Arc::clone(&metrics.counters));
    // Per-effect TOML configuration happens inside the composite
    // builder (preset path) and at construction for [`PassthroughEffect`],
    // so the runtime only needs to drive `prepare_all` here. The chain
    // never sees standalone effects with unconfigured state.
    chain
        .prepare_all(&processing_ctx)
        .map_err(FluxError::from)?;

    let (effective_out_w, effective_out_h) = cfg
        .output
        .effective_dimensions(cfg.input.width, cfg.input.height);
    info!(
        source = source_label,
        sink = sink_label,
        effects = ?chain.names(),
        input_w = cfg.input.width,
        input_h = cfg.input.height,
        input_fps = cfg.input.fps,
        output_w = effective_out_w,
        output_h = effective_out_h,
        output_scale = cfg.output.scale.value(),
        input_format = ?cfg.input.format,
        output_format = ?cfg.output.format,
        "starting pipeline ({source_label} -> {sink_label})",
    );

    output.start()?;
    // From here the node advertises CAPTURE caps again; close any
    // invisible window opened by the previous run's teardown.
    mark_output_live();

    // Stage 16: acquire the input AFTER the output is live. On the
    // supervised path a busy/absent camera streams the placeholder and
    // retries with backoff instead of tearing the loopback down.
    // `prepare_input` tears the output + chain down itself on a clean
    // shutdown (`None`) or a permanent error (`Err`).
    let Some(input) = prepare_input(cfg, &input_builder, &output, &metrics, &mut chain)? else {
        // Shutdown requested while waiting for the camera; prepare_input
        // already tore the output + chain down.
        return Ok(());
    };

    let slot = input.slot();
    // The token guard MUST live until the end of this function: when it
    // drops it removes the run's weak entry from the global registry.
    let (_token_guard, running) = register_token(slot.clone());

    // Bus listener relays GStreamer fatal errors and EOS into the shared
    // shutdown flag, and surfaces any captured error back to the caller.
    // Input-side failures are split off into `input_failure` so they can
    // be recovered in place instead of taking the loopback down.
    let bus_error: Arc<Mutex<Option<FluxError>>> = Arc::new(Mutex::new(None));
    let input_failure = Arc::new(InputFailureSignal::new(supervised_acquire(cfg)));
    let _bus_listener = build_bus_listener(
        &input,
        &output,
        Arc::clone(&running),
        Arc::clone(&bus_error),
        slot.clone(),
        Arc::clone(&input_failure),
    );

    // `metrics` was constructed above (before `prepare_all`) so the
    // counters Arc could be threaded into `ProcessingContext`.  Same
    // bundle owns the histograms used by the periodic reporter below.

    // Periodic reporter: emits per-window fps + percentiles via `info!`
    // at the cadence configured in `[realtime] metrics_interval_secs`.
    // `metrics_interval_secs = 0` disables it (teardown summary still
    // runs). `_reporter` is dropped at the end of this function — its
    // `Drop` impl joins the thread before we tear down counters.
    let _reporter = spawn_metrics_reporter(cfg, &metrics, &running);

    // Stage 13 control socket. Spawned only when explicitly enabled.
    // The handle owns the listener thread; dropping it unlinks the
    // socket file at end-of-run.
    let (control_handle, control_rx) = maybe_spawn_control(cfg);

    // Stage 15 idle runtime: detector spawn + state machine + cached
    // placeholder. Returns `None` (Stage 14 fallthrough) when idle
    // mode is disabled, the sink is non-V4L2, or the host is non-Linux.
    let mut idle_runtime = build_idle_runtime(cfg, &input, &output, &running, &metrics.counters);

    let process_result = run_supervised_loop(
        WorkerDeps {
            cfg,
            config_path,
            initial_preset_name: preset_name,
            processing_ctx: &processing_ctx,
            metrics: &metrics,
            control_rx: control_rx.as_ref(),
        },
        &running,
        &slot,
        &mut chain,
        &output,
        &mut idle_runtime,
        SupervisedInput {
            pipeline: &input,
            failure: &input_failure,
        },
    );

    drop(control_handle);

    // Ensure the bus listener wakes and exits.  Dropping `_bus_listener`
    // at the end of the function calls `BusListener::stop` via Drop, but
    // we also flip the flag here so the listener observes shutdown even
    // before the drop runs.
    running.store(false, Ordering::Release);
    slot.close();

    teardown(&input, &output, &mut chain);

    // Final reconciliation: pull whatever the slot dropped between the
    // last sync inside the loop and shutdown.
    metrics.sync_dropped(slot.dropped_count());
    let snap = metrics.snapshot();
    // Absolute totals at teardown. Shares the single-source-of-truth
    // field list with the periodic reporter via `emit_metrics_line`
    // (passing `None` — no rolling-window deltas here), so every counter
    // is emitted from one place under the `fluxframe::metrics` target.
    fluxframe_core::emit_metrics_line(&snap, None);

    // Surface bus-reported errors when the processing loop itself was
    // clean, so the operator sees the real cause of shutdown.
    let bus_err = bus_error.lock().expect("bus_error mutex poisoned").take();
    match (process_result, bus_err) {
        (Ok(()), Some(e)) | (Err(e), _) => Err(e),
        (Ok(()), None) => Ok(()),
    }
}

/// Stop the already-started output and shut the effect chain down —
/// the partial-teardown path when input acquisition aborts (shutdown)
/// or fails permanently, before the full [`teardown`] wiring exists.
fn teardown_partial(output: &OutputPipeline, chain: &mut EffectChain) {
    if let Err(e) = output.stop() {
        warn!(error = %e, "output.stop failed during acquire teardown");
    }
    if let Err(e) = chain.shutdown_all() {
        warn!(error = %e, "chain shutdown reported an error during acquire teardown");
    }
}

/// Build the idle placeholder (supervised path only) and acquire the
/// input. Returns `Some(input)` to proceed, `None` if shutdown was
/// requested while waiting for the camera (output + chain already torn
/// down here), or `Err` on a permanent acquire failure (also torn down
/// here, so the caller just propagates).
fn prepare_input<F>(
    cfg: &FluxConfig,
    input_builder: &F,
    output: &OutputPipeline,
    metrics: &RuntimeMetrics,
    chain: &mut EffectChain,
) -> Result<Option<Arc<InputPipeline>>, FluxError>
where
    F: Fn(InputParams) -> Result<InputPipeline, PipelineError>,
{
    let (out_w, out_h) = cfg
        .output
        .effective_dimensions(cfg.input.width, cfg.input.height);
    let placeholder: Option<Box<dyn crate::idle::Placeholder>> = if supervised_acquire(cfg) {
        match crate::idle::build_placeholder(&cfg.idle, out_w, out_h) {
            Ok(p) => Some(p),
            Err(e) => {
                warn!(error = %e, "failed to build acquire placeholder — falling back to single-attempt input acquisition");
                None
            }
        }
    } else {
        None
    };
    match acquire_input(cfg, input_builder, output, metrics, placeholder.as_deref()) {
        Ok(Some(input)) => Ok(Some(Arc::new(input))),
        Ok(None) => {
            teardown_partial(output, chain);
            Ok(None)
        }
        Err(e) => {
            teardown_partial(output, chain);
            Err(e)
        }
    }
}

/// Build the output (loopback) pipeline and the processing context,
/// independent of the input. Stage 16 brings the output up *first* so
/// `/dev/video10` advertises CAPTURE caps and streams a placeholder
/// while the camera is acquired (or retried) — see [`acquire_input`].
fn build_output(
    cfg: &FluxConfig,
) -> Result<(OutputPipeline, ProcessingContext, &'static str), FluxError> {
    let sink = resolve_output_sink(cfg)?;
    let sink_label = output_sink_label(&sink);
    // Sink dimensions = input × `output.scale`, rounded to even pixels.
    // fps is inherited from `cfg.input.fps` verbatim; the camera's
    // detected native fps is applied to the *input* at acquire time, and
    // the output writer thread decouples the two cadences (it re-pushes
    // the latest composite at the output rate, deduped by sequence).
    let (sink_w, sink_h) = cfg
        .output
        .effective_dimensions(cfg.input.width, cfg.input.height);
    let output_params = OutputParams::new(sink_w, sink_h, cfg.input.fps, cfg.input.format, sink)
        .with_sink_format(cfg.output.format);

    let output = OutputPipeline::build(output_params)?;

    let processing_ctx = ProcessingContext {
        width: cfg.input.width,
        height: cfg.input.height,
        format: cfg.input.format,
        fps: cfg.input.fps,
        // `run_chain` fills in the supervisor's `Arc<Counters>` after
        // this returns.  `None` is the correct default for any
        // standalone caller that doesn't run a full supervisor.
        counters: None,
    };

    Ok((output, processing_ctx, sink_label))
}

/// Whether the supervised (Stage 16) always-on path applies: idle mode
/// on, a v4l2loopback sink (so a placeholder keeps the device visible),
/// and a v4l2 camera input (the only source that can be busy/absent).
/// testsrc never contends, so it keeps the plain single-attempt path.
fn supervised_acquire(cfg: &FluxConfig) -> bool {
    cfg.idle.enabled
        && matches!(classify_output(cfg), OutputSpec::V4l2(_))
        && matches!(classify_input(cfg), InputSpec::V4l2(_))
}

/// `true` for input errors that are device-contention (camera busy or
/// temporarily absent) — the only class the supervisor retries forever.
/// Everything else (missing element, bad caps/format, config) is treated
/// as permanent and propagated so a real misconfiguration fails fast.
fn is_input_contention(e: &FluxError) -> bool {
    matches!(
        e,
        FluxError::Pipeline(fluxframe_core::PipelineError::InputDeviceUnavailable { .. })
    )
}

/// Acquire (build + start) the input pipeline.
///
/// Non-supervised path: a single attempt; any error propagates (the
/// `run_auto` layer decides whether to retry).
///
/// Supervised path (Stage 16, [`supervised_acquire`]): the output is
/// already streaming, so on a *contention* error this keeps pushing the
/// placeholder to `output` and retries with exponential backoff
/// (`input.acquire_backoff_base_ms` → … → `_max_ms`), forever. A
/// permanent error still propagates. `Ok(None)` signals shutdown was
/// requested while waiting — the caller exits cleanly.
fn acquire_input<F>(
    cfg: &FluxConfig,
    input_builder: &F,
    output: &OutputPipeline,
    metrics: &RuntimeMetrics,
    placeholder: Option<&dyn crate::idle::Placeholder>,
) -> Result<Option<InputPipeline>, FluxError>
where
    F: Fn(InputParams) -> Result<InputPipeline, PipelineError>,
{
    let (ph_w, ph_h) = cfg
        .output
        .effective_dimensions(cfg.input.width, cfg.input.height);
    let base = Duration::from_millis(u64::from(cfg.input.acquire_backoff_base_ms));
    let max = Duration::from_millis(u64::from(cfg.input.acquire_backoff_max_ms));
    let mut backoff = base;
    let mut logged_kind: Option<String> = None;

    loop {
        if is_shutdown_requested() {
            return Ok(None);
        }
        metrics.counters.inc_input_acquire_attempts();
        let params = InputParams::new(
            cfg.input.width,
            cfg.input.height,
            cfg.input.fps,
            cfg.input.format,
        );
        // build + start in one shot; both can surface a busy device.
        //
        // On a failed `start` the pipeline has usually reached Ready or
        // Paused, which means `v4l2src` already holds a device fd and
        // its streaming threads exist. `InputPipeline` has no `Drop`, so
        // simply dropping it here would leak both. That was survivable
        // when this ran once at startup; the supervised path retries in
        // a loop, so a camera missing for minutes would leak one
        // pipeline per attempt.
        let attempt = input_builder(params).and_then(|input| match input.start() {
            Ok(()) => Ok(input),
            Err(e) => {
                let _ = input.set_state_null();
                Err(e)
            }
        });
        match attempt {
            Ok(input) => return Ok(Some(input)),
            Err(e) => {
                let fe = FluxError::from(e);
                let Some(placeholder) = placeholder.filter(|_| is_input_contention(&fe)) else {
                    // Not supervised, or a permanent error → propagate.
                    return Err(fe);
                };
                let cont = backoff_after_contention(
                    &fe,
                    &mut logged_kind,
                    backoff,
                    placeholder,
                    ph_w,
                    ph_h,
                    cfg.input.format,
                    output,
                    metrics,
                )?;
                if !cont {
                    // Shutdown requested during the backoff sleep.
                    return Ok(None);
                }
                backoff = (backoff * 2).min(max);
            }
        }
    }
}

/// Handle one contention failure during [`acquire_input`]: count it, log
/// it (rate-limited — `warn` once per distinct error text, `debug` on
/// repeats), push a placeholder so the loopback stays visible, then
/// sleep the current `backoff` in Ctrl-C-responsive chunks. Returns
/// `Ok(false)` if shutdown was requested during the sleep, `Ok(true)` to
/// keep retrying, `Err` only if the output itself is broken.
#[expect(
    clippy::too_many_arguments,
    reason = "extracted from acquire_input's loop purely to satisfy too-many-lines; \
              the arguments are the acquire context and bundling them adds no clarity"
)]
fn backoff_after_contention(
    fe: &FluxError,
    logged_kind: &mut Option<String>,
    backoff: Duration,
    placeholder: &dyn crate::idle::Placeholder,
    ph_w: u32,
    ph_h: u32,
    format: fluxframe_core::frame::PixelFormat,
    output: &OutputPipeline,
    metrics: &RuntimeMetrics,
) -> Result<bool, FluxError> {
    metrics.counters.inc_input_acquire_failures();
    let kind = fe.to_string();
    if logged_kind.as_deref() == Some(kind.as_str()) {
        tracing::debug!(
            target: "fluxframe::idle",
            backoff_ms = backoff.as_millis() as u64,
            "input still unavailable — retrying",
        );
    } else {
        warn!(
            target: "fluxframe::idle",
            error = %kind,
            backoff_ms = backoff.as_millis() as u64,
            "input device unavailable — streaming placeholder, retrying with backoff",
        );
        *logged_kind = Some(kind);
    }
    // Output itself broken → genuinely fatal (propagates).
    push_placeholder(placeholder, ph_w, ph_h, format, output, metrics)?;
    // Sleep the backoff in short chunks so Ctrl-C is honoured within
    // ~100 ms regardless of the current backoff.
    let deadline = Instant::now() + backoff;
    while Instant::now() < deadline {
        if is_shutdown_requested() {
            return Ok(false);
        }
        std::thread::sleep(
            deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(100)),
        );
    }
    Ok(true)
}

/// Build the [`BusListener`] watching both pipelines.  The listener is
/// returned so the caller can keep it alive (its `Drop` joins the thread).
fn build_bus_listener(
    input: &InputPipeline,
    output: &OutputPipeline,
    running: Arc<AtomicBool>,
    bus_error: Arc<Mutex<Option<FluxError>>>,
    slot: LatestFrameSlot,
    input_failure: Arc<InputFailureSignal>,
) -> BusListener {
    let pipelines = vec![
        WatchedPipeline {
            source: BusSource::Input,
            bus: input.bus(),
        },
        WatchedPipeline {
            source: BusSource::Output,
            bus: output.bus(),
        },
    ];
    BusListener::spawn(pipelines, move |event| {
        on_bus_event(&event, &running, &bus_error, &slot, &input_failure);
    })
}

/// Upper bound on how long we keep trying to recover the camera in
/// place before giving up and letting the run restart.
///
/// Escalation is not a defeat, it is the second half of the strategy:
/// `InputPipeline::reacquire` reopens *the same* device node, so a
/// camera that came back on a different `/dev/videoN` (very common after
/// a physical replug) can only be found by the auto-input loop
/// re-enumerating. Retrying in place forever would turn that case into a
/// permanent hang — the exact self-healing behaviour this path must not
/// regress.
///
/// It also bounds how long control-socket commands go unanswered: the
/// recovery runs on the worker thread, so nothing drains the command
/// channel until it finishes one way or the other.
const INPUT_RECOVERY_BUDGET: Duration = Duration::from_secs(60);

/// What [`run_supervised_loop`] should do after the worker loop returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoopAction {
    /// Recover the camera in place; the output pipeline stays up.
    ReacquireInput,
    /// Hand control back to the caller (and thus to the auto-input
    /// loop, which rebuilds everything and re-selects the device).
    TeardownAll,
    /// Normal shutdown — Ctrl-C or a clean end of stream.
    Exit,
}

/// Decide what to do after the worker loop returned.
///
/// Pure so the matrix can be tested without GStreamer: building an
/// `OutputPipeline` needs a live `/dev/video10`, which no unit test has.
fn next_action(running: bool, input_failed: bool, worker_failed: bool) -> LoopAction {
    if worker_failed {
        // A chain/output error is not something reacquiring the camera
        // can fix.
        return LoopAction::TeardownAll;
    }
    if !running {
        return LoopAction::Exit;
    }
    if input_failed {
        LoopAction::ReacquireInput
    } else {
        LoopAction::Exit
    }
}

/// Run the worker loop, recovering from input-side failures in place.
///
/// The loopback output, the effect chain, the control socket, the
/// metrics reporter and the consumer detector are all created by the
/// caller and deliberately live *outside* this loop: recreating the
/// detector on every camera hiccup would reset its verdict to `Unknown`
/// (read as "consumer present"), so idle would stop engaging until the
/// consumer next reconnected.
fn run_supervised_loop(
    deps: WorkerDeps<'_>,
    running: &AtomicBool,
    slot: &LatestFrameSlot,
    chain: &mut EffectChain,
    output: &OutputPipeline,
    idle: &mut Option<IdleRuntime>,
    input: SupervisedInput<'_>,
) -> Result<(), FluxError> {
    loop {
        let watch = InputFailureWatch {
            signal: input.failure,
            entry_generation: input.failure.generation(),
        };
        let worker_result = run_process_loop(
            WorkerDeps { ..deps },
            running,
            slot,
            chain,
            output,
            idle.as_mut(),
            watch,
        );
        let worker_failed = worker_result.is_err();
        let action = next_action(
            running.load(Ordering::Acquire),
            watch.tripped(),
            worker_failed,
        );
        match action {
            LoopAction::Exit | LoopAction::TeardownAll => return worker_result,
            LoopAction::ReacquireInput => {
                recover_input(
                    deps.cfg,
                    input.pipeline,
                    output,
                    idle.as_mut(),
                    deps.metrics,
                    slot,
                )?;
            }
        }
    }
}

/// Bring the camera back without touching the output pipeline.
///
/// Uses `InputPipeline::reacquire` — the same object, so the worker's
/// frame slot and the bus watch stay valid. Building a fresh
/// `InputPipeline` here would create a new slot and a new bus that
/// nothing is subscribed to, orphaning both.
///
/// Returns `Err` once [`INPUT_RECOVERY_BUDGET`] is spent, so the caller
/// escalates to a full restart and the auto-input loop can re-select the
/// device.
fn recover_input(
    cfg: &FluxConfig,
    input: &Arc<InputPipeline>,
    output: &OutputPipeline,
    idle: Option<&mut IdleRuntime>,
    metrics: &RuntimeMetrics,
    slot: &LatestFrameSlot,
) -> Result<(), FluxError> {
    let started = Instant::now();

    // Snapshot what the retry loop needs from the idle runtime, so the
    // `&mut` borrow ends here rather than spanning the whole loop.
    let placeholder = idle.as_ref().map(|rt| {
        (
            Arc::clone(&rt.placeholder),
            rt.output_w,
            rt.output_h,
            rt.output_format,
        )
    });

    if let Some(rt) = idle {
        // Join any in-flight idle-resume reload before touching the
        // pipeline: both paths drive `set_state(Null/Playing)` on the
        // same `gst::Pipeline`, and racing those transitions wedges the
        // input. This is why recovery has a single owner.
        if let Some(handle) = rt.reload_handle.take() {
            debug!("joining an in-flight reload before recovering the input");
            let _ = handle.join();
        }
        // The engine itself is fine, but no frames are coming; make sure
        // no tick observes "active and ready" while the input is down.
        rt.engine_ready.store(false, Ordering::Release);
    }

    // Publish a placeholder before anything else. Until we do, the
    // output writer thread keeps re-pushing the last composite it saw,
    // so the consumer stares at a frozen frame that looks exactly like a
    // working camera.
    let push_fill = |metrics: &RuntimeMetrics| {
        if let Some((ph, w, h, fmt)) = placeholder.as_ref() {
            if let Err(e) = push_placeholder(ph.as_ref(), *w, *h, *fmt, output, metrics) {
                warn!(error = %e, "placeholder push failed during input recovery");
            }
        }
    };
    push_fill(metrics);

    if let Err(e) = input.set_state_null() {
        warn!(error = %e, "set_state_null failed while recovering the input");
    }
    // Drop whatever the dying device published last — a torn or
    // half-written frame would otherwise be the first thing the
    // consumer sees after recovery.
    slot.clear();

    let base = Duration::from_millis(u64::from(cfg.input.acquire_backoff_base_ms));
    let max = Duration::from_millis(u64::from(cfg.input.acquire_backoff_max_ms));
    let mut backoff = base;
    let mut attempt: u32 = 0;

    loop {
        if is_shutdown_requested() {
            return Ok(());
        }
        attempt += 1;
        metrics.counters.inc_input_reacquire();
        match input.reacquire() {
            Ok(()) => {
                let down = started.elapsed();
                metrics
                    .counters
                    .add_input_down_ms(u64::try_from(down.as_millis()).unwrap_or(u64::MAX));
                // Clear again: `reacquire` may have let a first frame
                // land while we were still deciding.
                slot.clear();
                info!(
                    attempt,
                    down_ms = down.as_millis() as u64,
                    "camera reacquired — output was never interrupted"
                );
                return Ok(());
            }
            Err(e) => {
                metrics.counters.inc_input_reacquire_failures();
                if started.elapsed() >= INPUT_RECOVERY_BUDGET {
                    warn!(
                        attempt,
                        elapsed_secs = started.elapsed().as_secs(),
                        error = %e,
                        "in-place camera recovery budget exhausted — restarting the run \
                         so the device can be re-selected"
                    );
                    return Err(FluxError::from(e));
                }
                debug!(attempt, error = %e, "reacquire failed; will retry");
                // Keep the loopback's ring buffer fresh while we retry:
                // consumers that filter on capabilities drop a node that
                // stops producing.
                push_fill(metrics);
                if wait_for_shutdown(backoff) {
                    return Ok(());
                }
                backoff = (backoff * 2).min(max);
            }
        }
    }
}

/// Shared "the input died" signal between the bus listener and the
/// worker loop.
///
/// A generation counter rather than a flag. The GStreamer bus is drained
/// on a timer, so losing a camera delivers a burst of messages: with a
/// boolean, clearing it after handling the first would immediately be
/// re-raised by the rest of the burst (an endless reacquire loop), while
/// clearing it before draining would swallow a genuinely new failure
/// that arrived during recovery. Comparing generations makes "has
/// anything failed since I started recovering?" exactly answerable.
///
/// `enabled` is false when the run has no supervised recovery path
/// (`idle.enabled = false`, or a sink with no placeholder). There an
/// input failure must stay fatal: with no placeholder to publish, the
/// writer thread would keep re-pushing the last live frame and the
/// consumer would see a frozen picture indistinguishable from a working
/// camera.
#[derive(Debug)]
pub(crate) struct InputFailureSignal {
    generation: AtomicU64,
    enabled: bool,
}

impl InputFailureSignal {
    fn new(enabled: bool) -> Self {
        Self {
            generation: AtomicU64::new(0),
            enabled,
        }
    }

    /// Does this signal take responsibility for failures from `source`?
    fn handles(&self, source: BusSource) -> bool {
        self.enabled && source == BusSource::Input
    }

    /// Record an input failure.
    fn raise(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Current generation. Snapshot it before recovering, and compare
    /// afterwards to tell "the failure I already handled" from "another
    /// one happened while I was recovering".
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }
}

/// Translate a [`BusEvent`] into supervisor side-effects.  Extracted so
/// the closure passed to [`BusListener::spawn`] stays a one-liner and so
/// the handler is unit-testable.
///
/// Fatal-error promotion (busy device → typed [`PipelineError`]) lives in
/// [`fluxframe_gst::translate_fatal`] so input-side and output-side bus
/// traffic agree on phrasing.
fn on_bus_event(
    event: &BusEvent,
    running: &AtomicBool,
    bus_error: &Mutex<Option<FluxError>>,
    slot: &LatestFrameSlot,
    input_failure: &InputFailureSignal,
) {
    match event {
        BusEvent::FatalError {
            element,
            message,
            debug: debug_payload,
            source,
        } => {
            // An input-side fatal error is recoverable in place when the
            // supervised path is active: the camera is gone, but the
            // loopback output is not, and tearing it down is what makes
            // clients lose the virtual camera entirely. Route it to the
            // reacquire path instead of killing the run.
            if input_failure.handles(*source) {
                warn!(
                    ?source,
                    %element,
                    %message,
                    debug = ?debug_payload,
                    action = "reacquire",
                    "input pipeline failed — recovering without tearing the output down",
                );
                input_failure.raise();
                // Deliberately NOT `slot.close()`: `LatestFrameSlot` has
                // no reopen, so closing it here would make every frame
                // after a successful reacquire a silent no-op. The worker
                // wakes on its own poll timeout anyway.
                //
                // Deliberately NOT recorded in `bus_error` either: that
                // latch is returned at end-of-run, so a recovered failure
                // would still surface as the run's error and could mask a
                // later, real one.
                return;
            }
            error!(
                ?source,
                %element,
                %message,
                debug = ?debug_payload,
                "bus fatal error",
            );
            if let Some(pipeline_err) = fluxframe_gst::translate_fatal(event) {
                let mut guard = bus_error.lock().expect("bus_error mutex poisoned");
                if guard.is_none() {
                    *guard = Some(FluxError::from(pipeline_err));
                }
            }
            running.store(false, Ordering::Release);
            slot.close();
        }
        BusEvent::Warning {
            source,
            element,
            message,
            debug: debug_payload,
        } => {
            warn!(
                ?source,
                %element,
                %message,
                debug = ?debug_payload,
                "bus warning",
            );
        }
        BusEvent::Eos { source } => {
            // `v4l2src` reports a hot-unplug as EOS at least as often as
            // it reports an error, so this needs the same treatment —
            // otherwise the most common way to lose the camera still
            // takes the loopback down with it.
            if input_failure.handles(*source) {
                warn!(?source, action = "reacquire", "input pipeline reached EOS");
                input_failure.raise();
                return;
            }
            info!(?source, "pipeline EOS");
            running.store(false, Ordering::Release);
            slot.close();
        }
        // `BusEvent` is `#[non_exhaustive]`: future variants land here
        // until the supervisor is taught to interpret them.
        _ => {
            warn!(?event, "unhandled bus event variant");
        }
    }
}

/// Build a [`MetricsReporter`] when periodic reporting is enabled by
/// the config, otherwise return `None`.  A `None` return means the
/// teardown summary is the only line the operator sees — useful for
/// CI / tests that do not want the per-window noise.
fn spawn_metrics_reporter(
    cfg: &FluxConfig,
    metrics: &RuntimeMetrics,
    running: &Arc<AtomicBool>,
) -> Option<MetricsReporter> {
    let secs = cfg.realtime.metrics_interval_secs;
    if secs == 0 {
        tracing::debug!("metrics reporter disabled by config (metrics_interval_secs = 0)");
        return None;
    }
    let interval = Duration::from_secs(u64::from(secs));
    match MetricsReporter::spawn(metrics.clone(), interval, Arc::clone(running)) {
        Ok(handle) => {
            tracing::debug!(secs, "metrics reporter spawned");
            Some(handle)
        }
        Err(e) => {
            warn!(error = %e, "could not spawn metrics reporter; teardown summary only");
            None
        }
    }
}

/// Mutable per-iteration state of the worker thread.
///
/// Carved out of [`run_process_loop`] so the orchestration skeleton
/// stays a thin wrapper while per-frame and control-drain logic each
/// live in their own helpers. Stage 15 idle wiring is grafted onto
/// the same struct without bloating the loop body.
///
/// Both `working_cfg` and `active_preset` are owned working copies,
/// not references into the caller's config: `reload` rebinds
/// `working_cfg` to a freshly-read `FluxConfig`, and Stage 13
/// `Set`/`SetChain` commands mutate `active_preset` alongside the live
/// chain so `current_config`/introspection stay consistent and chain
/// composition changes have a stable rebuild source.
struct WorkerState {
    frame_context: FrameContext,
    /// Track fallback edges so a single-frame visual artefact
    /// ("flicker") caused by an inference miss surfaces as a warn line
    /// tied to the exact frame_seq, instead of being lost in the
    /// per-frame debug noise.
    prev_fallback: bool,
    frames_seen: u64,
    active_preset_name: String,
    /// Owned config used for per-command lookups.
    working_cfg: FluxConfig,
    /// Working copy of the currently active preset.
    active_preset: Preset,
}

impl WorkerState {
    fn new(cfg: &FluxConfig, initial_preset_name: &str, metrics: &RuntimeMetrics) -> Self {
        let working_cfg: FluxConfig = cfg.clone();
        let active_preset = working_cfg
            .presets
            .get(initial_preset_name)
            .cloned()
            .unwrap_or_default();
        Self {
            frame_context: FrameContext {
                telemetry: metrics.effect_telemetry(),
                ..FrameContext::default()
            },
            prev_fallback: false,
            frames_seen: 0,
            active_preset_name: initial_preset_name.to_string(),
            working_cfg,
            active_preset,
        }
    }
}

/// Drain any pending control commands before processing the next
/// frame. Each command is fully applied (or rejected) before the
/// worker reads from the slot, so the swap is atomic with respect to
/// frame boundaries.
fn drain_control_commands(
    state: &mut WorkerState,
    control_rx: Option<&crossbeam_channel::Receiver<ControlEnvelope>>,
    config_path: Option<&std::path::Path>,
    processing_ctx: &ProcessingContext,
    chain: &mut EffectChain,
) {
    let Some(rx) = control_rx else {
        return;
    };
    while let Ok(envelope) = rx.try_recv() {
        // Capture variant label BEFORE the move into
        // `apply_control_command`. Slow control commands silently eat
        // frame budget (~33 ms at 30 fps), so surface elapsed_us per
        // command for diagnosis.
        let cmd_kind = command_label(&envelope.cmd);
        let start = Instant::now();
        let response = apply_control_command(
            &mut state.working_cfg,
            config_path,
            processing_ctx,
            chain,
            &mut state.active_preset_name,
            &mut state.active_preset,
            envelope.cmd,
        );
        tracing::debug!(
            command = cmd_kind,
            elapsed_us = start.elapsed().as_micros() as u64,
            "control command applied",
        );
        // Best-effort: if the listener already dropped the reply
        // receiver, the client has disconnected and we simply move on.
        let _ = envelope.reply_tx.send(response);
    }
}

/// Run the effect chain on a single frame and forward it to the
/// output sink. Returns `Err` only on unrecoverable failure — the
/// caller stops the worker loop.
fn process_one_frame(
    state: &mut WorkerState,
    mut frame: fluxframe_core::frame::VideoFrame,
    chain: &mut EffectChain,
    output: &OutputPipeline,
    metrics: &RuntimeMetrics,
    slot: &LatestFrameSlot,
    no_consumer: bool,
) -> Result<(), FluxError> {
    let recv_at = Instant::now();
    metrics.counters.inc_frames_in();
    state.frames_seen += 1;

    state.frame_context.frame_sequence = frame.meta.sequence;
    state.frame_context.frame_timestamp = frame.meta.timestamp;
    state.frame_context.fallback_active = false;

    let pre_process = Instant::now();
    if let Err(e) = chain.process(&mut frame, &mut state.frame_context) {
        metrics.counters.inc_effect_error();
        error!(error = %e, "effect chain failed; stopping");
        return Err(FluxError::from(e));
    }
    metrics.processing.record_duration(pre_process.elapsed());

    if state.frame_context.fallback_active != state.prev_fallback {
        warn!(
            frame_seq = state.frame_context.frame_sequence,
            fallback_active = state.frame_context.fallback_active,
            "effect-chain fallback state changed (passthrough frame)"
        );
        if state.frame_context.fallback_active {
            // Count edges into fallback only — a steady passthrough
            // run would otherwise inflate the counter every frame.
            metrics.counters.inc_fallback();
        }
        state.prev_fallback = state.frame_context.fallback_active;
    }

    let pre_push = Instant::now();
    if let Err(e) = output.push_frame(frame) {
        error!(error = %e, "output.push_frame failed; stopping");
        return Err(e.into());
    }
    metrics.output.record_duration(pre_push.elapsed());
    metrics.end_to_end.record_duration(recv_at.elapsed());
    metrics.counters.inc_frames_out();
    if no_consumer {
        metrics.counters.inc_frames_out_while_no_consumer();
    }

    if state.frames_seen % DROPPED_SYNC_INTERVAL == 0 {
        metrics.sync_dropped(slot.dropped_count());
    }
    Ok(())
}

/// Does the consumer detector currently say nobody is reading the
/// loopback?
///
/// A `None` idle runtime (idle disabled, or a sink with no detector)
/// means there is no presence signal at all, so frames are never
/// attributed as wasted — an absent signal is not evidence of an absent
/// consumer.
fn no_consumer_attached(idle: Option<&IdleRuntime>) -> bool {
    idle.is_some_and(|rt| {
        crate::idle::ConsumerStatus::from_u8(rt.consumer_status.load(Ordering::Acquire))
            == crate::idle::ConsumerStatus::Absent
    })
}

/// Push a pre-rendered placeholder frame to the output, bypassing
/// the effect chain. Used by the worker loop while in Idle to keep
/// v4l2loopback's ring buffer fresh (and the device enumerable by
/// Chrome) without paying the cost of the full pipeline.
fn push_placeholder(
    placeholder: &dyn crate::idle::Placeholder,
    width: u32,
    height: u32,
    format: fluxframe_core::frame::PixelFormat,
    output: &OutputPipeline,
    metrics: &RuntimeMetrics,
) -> Result<(), FluxError> {
    let frame = placeholder.render(width, height, format)?;
    if let Err(e) = output.push_frame(frame) {
        error!(error = %e, "output.push_frame failed during idle placeholder push");
        return Err(e.into());
    }
    metrics.counters.inc_idle_frames_pushed();
    Ok(())
}

/// Bundle of Stage 15 idle infrastructure handed to the worker loop.
/// `None` when idle mode is disabled or unavailable (non-V4L2 sink,
/// non-Linux host); the loop then runs the Stage 14 path verbatim.
struct IdleRuntime {
    /// Pure state machine — owned exclusively by the worker thread.
    state_machine: crate::idle::IdleStateMachine,
    /// Detector → worker status atomic. Worker `Acquire`-loads each
    /// tick; the detector thread `Release`-stores on transition.
    consumer_status: Arc<std::sync::atomic::AtomicU8>,
    /// Cached placeholder built once at startup (or supervisor
    /// reload) for the configured `(output_w, output_h, format)`.
    placeholder: Arc<dyn crate::idle::Placeholder>,
    /// Output appsrc dimensions + format — the placeholder was
    /// built for exactly these and the worker passes them on every
    /// `render` call.
    output_w: u32,
    output_h: u32,
    output_format: fluxframe_core::frame::PixelFormat,
    /// Shared input handle. The worker calls `set_state_null` on
    /// EnterIdle and the reload thread calls `start` on resume.
    input: Arc<InputPipeline>,
    /// `false` whenever the supervisor must not call
    /// `chain.process` — true in Active, cleared on resume while the
    /// reload thread restarts the input, re-armed by that thread.
    engine_ready: Arc<AtomicBool>,
    /// Live reload thread, if one is currently in flight. The
    /// worker checks `is_finished` periodically to surface failures
    /// in logs (Step 5 wires a `resume_latency_ms` histogram here).
    reload_handle: Option<std::thread::JoinHandle<crate::idle::reload::ResumeOutcome>>,
    /// Fix #7: tracks whether the previous worker iteration ran the
    /// full chain (`true`) or pushed a placeholder (`false`). The
    /// supervisor logs a one-shot `info!` line at every false→true
    /// transition so "stuck in placeholder?" troubleshooting surfaces
    /// the resume edge as an obvious line in the log stream.
    /// Defaults to `true` because a freshly-built runtime is in the
    /// Active level by construction.
    was_active: bool,
    /// Detector thread handle, kept alive for the duration of the
    /// run. Its `Drop` impl joins the detector thread.
    #[cfg(target_os = "linux")]
    _detector: crate::idle::ConsumerDetector,
}

/// Construct the idle-runtime bundle if the configuration enables
/// it, the host platform supports the detector, and the sink is a
/// V4L2 loopback. Returns `None` for non-V4L2 sinks, non-Linux
/// hosts, or when `idle.enabled = false` — the caller then runs
/// the Stage 14 worker loop unchanged.
fn build_idle_runtime(
    cfg: &FluxConfig,
    input: &Arc<InputPipeline>,
    output: &OutputPipeline,
    running: &Arc<AtomicBool>,
    counters: &Arc<fluxframe_core::Counters>,
) -> Option<IdleRuntime> {
    if !cfg.idle.enabled {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let sink = resolve_output_sink(cfg).ok()?;
        // Fix #6: surface the silent fallback so an operator who set
        // `idle.enabled = true` with a non-V4L2 sink sees why idle is
        // not taking effect, rather than silently running the Stage 14
        // path.
        let Some(device_path) = crate::idle::device_path(&sink) else {
            warn!(
                "idle.enabled = true but sink is not v4l2loopback — idle disabled \
                 (the consumer-presence detector needs a /dev/videoN device path for the \
                 inotify watch + /proc/*/fd walk)"
            );
            return None;
        };
        let my_pid = std::process::id();

        let (sink_w, sink_h) = cfg
            .output
            .effective_dimensions(cfg.input.width, cfg.input.height);
        // Placeholder format must match the appsrc's caps — that's
        // the *input* format on the supervisor's pipeline. The
        // output's sink format is reached via videoconvert downstream.
        let placeholder_format = cfg.input.format;
        let placeholder: Arc<dyn crate::idle::Placeholder> =
            match crate::idle::build_placeholder(&cfg.idle, sink_w, sink_h) {
                Ok(p) => p.into(),
                Err(e) => {
                    warn!(error = %e, "failed to build idle placeholder — idle mode disabled");
                    return None;
                }
            };

        // Ask the loopback for a capture-usage subscription. `None`
        // means this sink owns no device fd (only the direct-write path
        // does); `Err` means the driver predates the event. Neither is
        // fatal — the detector decides what to do based on
        // `presence_source`, and only `kernel_event` insists on it.
        let watch = match output.subscribe_consumer_events() {
            Some(Ok(watch)) => Some(watch),
            Some(Err(e)) => {
                warn!(
                    error = %e,
                    "could not subscribe to loopback client-usage events \
                     (v4l2loopback older than 0.13?)"
                );
                None
            }
            None => None,
        };

        let detector_running = Arc::clone(running);
        let detector = crate::idle::ConsumerDetector::spawn(
            crate::idle::DetectorParams {
                device_path,
                my_pid,
                poll_interval: std::time::Duration::from_millis(u64::from(
                    cfg.idle.poll_interval_ms,
                )),
                mode: cfg.idle.presence_source,
                // Never decay a balance-backed verdict faster than the
                // grace period the operator chose for entering idle:
                // below that, the decay would race the very hysteresis
                // it sits behind.
                stale_balance_after: std::time::Duration::from_secs(u64::from(
                    cfg.idle.teardown_secs.max(1),
                )),
            },
            detector_running,
            Arc::clone(counters),
            watch,
        );
        // The detector handle moves into the runtime so its Drop
        // runs when the worker loop exits.
        let consumer_status = detector.status_handle();

        Some(IdleRuntime {
            state_machine: crate::idle::IdleStateMachine::new(Instant::now()),
            consumer_status,
            placeholder,
            output_w: sink_w,
            output_h: sink_h,
            output_format: placeholder_format,
            input: Arc::clone(input),
            engine_ready: Arc::new(AtomicBool::new(true)),
            reload_handle: None,
            was_active: true,
            _detector: detector,
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (input, running, counters);
        warn!(
            target: "fluxframe::idle",
            "idle mode requested but host is not Linux — idle disabled (the /proc/*/fd consumer detector is Linux-only)"
        );
        None
    }
}

/// Non-blocking reap of a finished reload thread. Returns its
/// [`crate::idle::reload::ResumeOutcome`] when the thread has already
/// exited (the `join` is then instant); leaves an in-flight handle in
/// place and returns `None`. Called by [`tick_idle`] every iteration.
///
/// Takes the handle slot directly (rather than the whole `IdleRuntime`)
/// so it stays unit-testable without fabricating a full runtime. The
/// shutdown path does its own *blocking* join instead — it must wait for
/// an in-flight reload to finish before teardown, which this
/// non-blocking reaper deliberately does not.
fn reap_reload(
    reload_handle: &mut Option<std::thread::JoinHandle<crate::idle::reload::ResumeOutcome>>,
) -> Option<crate::idle::reload::ResumeOutcome> {
    let handle = reload_handle.take()?;
    if !handle.is_finished() {
        *reload_handle = Some(handle);
        return None;
    }
    if let Ok(outcome) = handle.join() {
        Some(outcome)
    } else {
        warn!(target: "fluxframe::idle", "reload thread panicked");
        None
    }
}

/// Handle the side effects of an `IdleEdge`. Updates counters,
/// reconfigures the input pipeline, and spawns the off-worker
/// reload thread when needed.
fn handle_idle_edge(
    idle: &mut IdleRuntime,
    edge: crate::idle::IdleEdge,
    slot: &LatestFrameSlot,
    output: &OutputPipeline,
    metrics: &RuntimeMetrics,
) {
    use crate::idle::IdleEdge;
    match edge {
        IdleEdge::None => {}
        IdleEdge::EnterIdle => {
            metrics.counters.inc_idle_entered();
            info!(target: "fluxframe::idle", "entering idle — tearing down input pipeline");
            if let Err(e) = idle.input.set_state_null() {
                warn!(error = %e, "set_state_null failed on idle entry");
            }
            // Discard any in-flight frame the capture thread published
            // before the pipeline transitioned to Null — resuming with
            // a stale frame on the chain would be visible to the
            // consumer as a single mis-timed image.
            slot.clear();
        }
        IdleEdge::ResumeActive => {
            info!(target: "fluxframe::idle", "consumer reconnected — spawning reload thread");
            // Push one placeholder immediately to clear
            // v4l2loopback's stale-frame replay before any real
            // frames arrive (50–300 ms warmup on UVC).
            if let Err(e) = push_placeholder(
                idle.placeholder.as_ref(),
                idle.output_w,
                idle.output_h,
                idle.output_format,
                output,
                metrics,
            ) {
                warn!(error = %e, "resume-edge placeholder push failed");
            }
            // Spawn the reload coordinator. The engine reloader is a
            // no-op here because the ONNX session is kept warm across
            // Idle; `input.reacquire()` inside `spawn_reload_thread` is
            // the meaningful work.
            //
            // Double-spawn guard: a finished reload was already reaped
            // (and acted on) in `tick_idle` before this edge dispatch,
            // so `reload_handle` is either empty or an in-flight thread.
            // Never kick a second reload over an in-flight one — racing
            // the GStreamer state transitions would wedge the input.
            if let Some(handle) = idle.reload_handle.take() {
                if !handle.is_finished() {
                    warn!("a previous reload thread is still in flight; not spawning a second one");
                    idle.reload_handle = Some(handle);
                    return;
                }
                // Straggler — unreachable in the normal flow (tick_idle
                // reaps finished handles first); join to avoid detaching.
                let _ = handle.join();
            }
            // Clear `engine_ready` synchronously on this (worker) thread
            // before launching, so no tick observes `Active + ready`
            // during the resume window (the input was just set to Null
            // on EnterIdle).
            idle.engine_ready.store(false, Ordering::Release);
            let input = Arc::clone(&idle.input);
            let engine_ready = Arc::clone(&idle.engine_ready);
            idle.reload_handle = Some(crate::idle::reload::spawn_reload_thread(
                input,
                engine_ready,
                || Ok(()),
            ));
        }
    }
}

/// Result of [`tick_idle`]: tells [`run_process_loop`] whether the
/// idle state-machine handled this iteration (`Continue`) or whether
/// the worker should fall through and pull a real frame
/// (`ProcessFrame`). The variant carries no payload — placeholder
/// pushes are issued inside the helper so the caller only sees a
/// two-arm match.
enum IdleTickAction {
    /// Idle helper either pushed a placeholder or parked on the
    /// slot; the caller should `continue` to the next loop iteration.
    Continue,
    /// No idle handling required (idle disabled OR Active level with
    /// engine ready); the caller should pull a frame and run the
    /// chain.
    ProcessFrame,
}

/// Run one tick of the idle state machine and dispatch the resulting
/// edge + level.  Pulled out of [`run_process_loop`] so the orchestrator
/// loop stays a thin three-stage skeleton (drain control commands → tick
/// idle → process frame). Mixing the state-machine tick, edge dispatch,
/// placeholder push and shutdown-aware park inside the loop body
/// stacked three abstraction levels on top of each other; this helper
/// owns one of them.
///
/// `state` is taken as `&WorkerState` (not mutable) — the only field
/// read is `state.working_cfg.idle`, which the idle state machine uses
/// as its timing source.  Mutability on the per-iteration counters is
/// reserved for [`process_one_frame`].
fn tick_idle(
    idle: Option<&mut IdleRuntime>,
    state: &WorkerState,
    slot: &LatestFrameSlot,
    output: &OutputPipeline,
    metrics: &RuntimeMetrics,
    input_failure: &InputFailureSignal,
) -> Result<IdleTickAction, FluxError> {
    let Some(idle_rt) = idle else {
        return Ok(IdleTickAction::ProcessFrame);
    };
    // Reap a finished reload thread BEFORE ticking the state machine so
    // its outcome is never dropped or double-spawned. A failed camera
    // reacquire returns a transient error here, which unwinds
    // `run_process_loop` → `run_chain` → `run_once`. Under
    // `device = "auto"` that re-enters `run_auto`, which re-enumerates
    // and re-selects the (re-plugged) device; for a fixed device path
    // the daemon exits and relies on the service supervisor to restart
    // it (documented in the changelog). Either way beats the old
    // behaviour of spinning in placeholder forever.
    if let Some(outcome) = reap_reload(&mut idle_rt.reload_handle) {
        use crate::idle::reload::ResumeOutcome;
        match outcome {
            // The reload thread already logs completion (at info, with
            // elapsed); nothing to do here but drop the reaped handle.
            ResumeOutcome::Completed => {}
            ResumeOutcome::EngineFailed(reason) => {
                // ORT/disk problem, not the camera — stay in placeholder;
                // a later ResumeActive edge re-spawns the reload attempt.
                warn!(
                    target: "fluxframe::idle",
                    error = %reason,
                    "reload: engine rebuild failed — staying in placeholder mode"
                );
            }
            ResumeOutcome::InputFailed(e) => {
                metrics.counters.inc_resume_failures();
                if input_failure.handles(fluxframe_gst::BusSource::Input) {
                    // Hand this to the supervised recovery path instead
                    // of unwinding the run. This is the single most
                    // common way the camera is lost in practice — the
                    // consumer reconnects, we resume, and the device is
                    // not back yet — and unwinding here tore the
                    // loopback down every time, which is precisely what
                    // made clients lose the virtual camera.
                    warn!(
                        target: "fluxframe::idle",
                        error = %e,
                        action = "reacquire",
                        "reload: camera reacquire failed — recovering without \
                         tearing the output down"
                    );
                    input_failure.raise();
                    return Ok(IdleTickAction::Continue);
                }
                warn!(
                    target: "fluxframe::idle",
                    error = %e,
                    "reload: camera reacquire failed — restarting chain to re-select the device"
                );
                return Err(e);
            }
        }
    }
    let status =
        crate::idle::ConsumerStatus::from_u8(idle_rt.consumer_status.load(Ordering::Acquire));
    let tick = idle_rt
        .state_machine
        .tick(status, Instant::now(), &state.working_cfg.idle);
    handle_idle_edge(idle_rt, tick.edge, slot, output, metrics);
    let active_allowed = matches!(tick.level, crate::idle::IdleLevel::Active)
        && idle_rt.engine_ready.load(Ordering::Acquire);
    if !active_allowed {
        if let Err(e) = push_placeholder(
            idle_rt.placeholder.as_ref(),
            idle_rt.output_w,
            idle_rt.output_h,
            idle_rt.output_format,
            output,
            metrics,
        ) {
            error!(error = %e, "placeholder push failed; stopping");
            return Err(e);
        }
        idle_rt.was_active = false;
        // Park up to `next_tick_in` on the slot (which doubles as a
        // shutdown signal) so we don't busy-loop. The worker still
        // gets woken if a real frame ever lands (resume happened and
        // the input came back).
        let _ = slot.recv_timeout(tick.next_tick_in);
        return Ok(IdleTickAction::Continue);
    }
    // Fix #7: log the one-shot resume edge — placeholder mode → active
    // processing. The state machine's own `EnterIdle`/`ResumeActive`
    // edges fire on the consumer-status transition; this complementary
    // line surfaces when the worker actually goes back to running the
    // full chain (which may lag the `ResumeActive` edge by the reload
    // thread's `input.start()` window).
    if !idle_rt.was_active {
        info!(target: "fluxframe::idle", "resumed active processing");
        idle_rt.was_active = true;
    }
    Ok(IdleTickAction::ProcessFrame)
}

/// Stable-during-run dependencies handed to [`run_process_loop`].
///
/// The supervisor's worker loop reaches for ~10 values — most of
/// them never change across iterations (the merged `FluxConfig`, the
/// config-file path used by `reload`, the preset name we started on,
/// the processing context, the metrics bundle). Bundling them in one
/// struct keeps the loop's signature small enough to drop the
/// `#[allow(clippy::too_many_arguments)]` and makes the genuinely
/// per-iteration handles (`running`, `slot`, `chain`, `output`,
/// `control_rx`, `idle`) stand out at the call site.
///
/// All fields are borrowed references, so the struct is `Copy` and
/// can be passed by value to the loop without imposing a borrow
/// lifetime past the call.
#[derive(Clone, Copy)]
struct WorkerDeps<'a> {
    cfg: &'a FluxConfig,
    config_path: Option<&'a std::path::Path>,
    initial_preset_name: &'a str,
    processing_ctx: &'a ProcessingContext,
    metrics: &'a RuntimeMetrics,
    /// Control-socket receiver, or `None` when the socket is disabled.
    control_rx: Option<&'a crossbeam_channel::Receiver<ControlEnvelope>>,
}

/// The input-failure signal paired with the generation observed when
/// the worker loop was entered.
///
/// Carrying the two together makes the only meaningful question —
/// "has the input failed *since I started*?" — a method rather than a
/// comparison the caller could get subtly wrong.
#[derive(Clone, Copy)]
struct InputFailureWatch<'a> {
    signal: &'a InputFailureSignal,
    entry_generation: u64,
}

/// The input side of a supervised run: the pipeline to recover and the
/// signal that says when it needs recovering. Paired because neither is
/// useful to the supervisor without the other.
#[derive(Clone, Copy)]
struct SupervisedInput<'a> {
    pipeline: &'a Arc<InputPipeline>,
    failure: &'a InputFailureSignal,
}

impl InputFailureWatch<'_> {
    /// Has a new input failure been raised since this watch was taken?
    fn tripped(&self) -> bool {
        self.signal.generation() != self.entry_generation
    }
}

fn run_process_loop(
    deps: WorkerDeps<'_>,
    running: &AtomicBool,
    slot: &LatestFrameSlot,
    chain: &mut EffectChain,
    output: &OutputPipeline,
    mut idle: Option<&mut IdleRuntime>,
    input_failure: InputFailureWatch<'_>,
) -> Result<(), FluxError> {
    let mut state = WorkerState::new(deps.cfg, deps.initial_preset_name, deps.metrics);
    // Leaving on a *new* input failure hands control to the supervisor,
    // which recovers the camera and calls back in. Without this the loop
    // would spin against a dead input: `tick_idle` still reports Active
    // while a consumer is attached, so every iteration would fall
    // through to a `recv_timeout` that can never succeed.
    while running.load(Ordering::Acquire) && !input_failure.tripped() {
        drain_control_commands(
            &mut state,
            deps.control_rx,
            deps.config_path,
            deps.processing_ctx,
            chain,
        );

        // Stage 15 idle integration: tick the state machine, dispatch
        // any side-effect edge, choose between Active level (pull a
        // real frame + run chain) and Placeholder level (push the
        // cached fill).
        match tick_idle(
            idle.as_deref_mut(),
            &state,
            slot,
            output,
            deps.metrics,
            input_failure.signal,
        )? {
            IdleTickAction::Continue => continue,
            IdleTickAction::ProcessFrame => {}
        }

        let Some(frame) = slot.recv_timeout(WORKER_POLL_TIMEOUT) else {
            // Either timeout (no frame within the poll window) or slot
            // closed by shutdown.  Re-check the flag and continue.
            continue;
        };
        process_one_frame(
            &mut state,
            frame,
            chain,
            output,
            deps.metrics,
            slot,
            no_consumer_attached(idle.as_deref()),
        )?;
    }
    // Reap any in-flight reload thread so it doesn't outlive the
    // supervisor. Surface the wall-clock cost so a reload that was
    // racing shutdown shows up in the teardown log line instead of
    // disappearing into a dropped `JoinHandle`.
    if let Some(idle_rt) = idle.as_mut() {
        if let Some(handle) = idle_rt.reload_handle.take() {
            match handle.join() {
                Ok(outcome) => {
                    tracing::debug!(
                        target: "fluxframe::idle",
                        ?outcome,
                        "reload thread reaped at shutdown",
                    );
                }
                Err(_) => {
                    warn!(target: "fluxframe::idle", "reload thread panicked during shutdown");
                }
            }
        }
    }
    Ok(())
}

fn teardown(input: &InputPipeline, output: &OutputPipeline, chain: &mut EffectChain) {
    if let Err(e) = input.stop() {
        warn!(error = %e, "input.stop failed");
    }
    if let Err(e) = output.stop() {
        warn!(error = %e, "output.stop failed");
    }
    if let Err(e) = chain.shutdown_all() {
        warn!(error = %e, "effect chain shutdown reported an error");
    }
}

fn resolve_output_sink(cfg: &FluxConfig) -> Result<OutputSink, FluxError> {
    // Routing rules (see Stage 2 plan, §"Output side"):
    //   * "auto"             -> autovideosink (manual glance verification).
    //   * "fakesink"         -> fakesink (CI / dev smoke without a loopback).
    //   * /dev/...           -> v4l2sink to a loopback device.
    //   * "pipewire[:name]"  -> pipewiresink publishing a PW node.
    //   * everything else    -> structured §27 Config error.
    //
    // Triage lives in [`classify_output`]; this function only maps the
    // typed `OutputSpec` onto the GStreamer-facing `OutputSink`.
    match classify_output(cfg) {
        OutputSpec::Auto => Ok(OutputSink::Auto),
        OutputSpec::Fake => Ok(OutputSink::Fake),
        OutputSpec::V4l2(device) => Ok(OutputSink::V4l2Loopback { device }),
        OutputSpec::Pipewire(node_name) => Ok(OutputSink::Pipewire { node_name }),
        OutputSpec::Unsupported(d) => Err(FluxError::Config {
            reason: format!("output '{d}' is not supported"),
            hint: Some("supported outputs: auto, fakesink, /dev/video<N>, pipewire[:name]".into()),
        }),
    }
}

fn output_sink_label(sink: &OutputSink) -> &'static str {
    // `OutputSink` is not `#[non_exhaustive]` inside the workspace, so
    // this match is genuinely exhaustive: adding a variant will produce
    // a compile-time prompt here.  Borrowed because `V4l2Loopback` owns
    // a `PathBuf` and is therefore no longer `Copy`.
    match sink {
        OutputSink::Fake => "fakesink",
        OutputSink::Auto => "autovideosink",
        OutputSink::V4l2Loopback { .. } => "v4l2sink",
        OutputSink::Pipewire { .. } => "pipewiresink",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cfg() -> FluxConfig {
        // Default Test configuration: start from `FluxConfig::default()`.
        // The fields tweaked here are the ones exercised by each test.
        FluxConfig::default()
    }

    // --- bus-event routing -------------------------------------------

    fn fatal(source: BusSource) -> BusEvent {
        BusEvent::FatalError {
            source,
            element: "input_src".into(),
            message: "Device '/dev/video0' is busy".into(),
            debug: None,
        }
    }

    /// Run one bus event against fresh state and report what it did.
    fn route(event: &BusEvent, supervised: bool) -> (bool, bool, u64, bool) {
        let running = AtomicBool::new(true);
        let bus_error = Mutex::new(None);
        let slot = LatestFrameSlot::new();
        let signal = InputFailureSignal::new(supervised);
        on_bus_event(event, &running, &bus_error, &slot, &signal);
        (
            running.load(Ordering::Acquire),
            slot.is_closed(),
            signal.generation(),
            bus_error.lock().expect("poisoned").is_some(),
        )
    }

    #[test]
    fn supervised_input_fatal_is_recovered_not_fatal() {
        let (running, closed, generation, latched) = route(&fatal(BusSource::Input), true);
        assert!(running, "an input failure must not stop the run");
        // The load-bearing assertion: `LatestFrameSlot` has no reopen, so
        // closing it here would silently discard every frame produced
        // after a successful recovery.
        assert!(!closed, "the frame slot must stay open across recovery");
        assert_eq!(generation, 1, "the supervisor must be told to recover");
        assert!(
            !latched,
            "a recovered failure must not be returned as the run's error"
        );
    }

    #[test]
    fn output_fatal_is_always_fatal() {
        // Nothing about reacquiring the camera can fix the sink.
        let (running, closed, generation, latched) = route(&fatal(BusSource::Output), true);
        assert!(!running);
        assert!(closed);
        assert_eq!(generation, 0);
        assert!(latched);
    }

    #[test]
    fn unsupervised_input_fatal_stays_fatal() {
        // Without a placeholder the writer thread would keep re-pushing
        // the last live frame, so the consumer would see a frozen image
        // instead of any indication that the camera is gone.
        let (running, closed, generation, latched) = route(&fatal(BusSource::Input), false);
        assert!(!running);
        assert!(closed);
        assert_eq!(generation, 0);
        assert!(latched);
    }

    #[test]
    fn supervised_input_eos_is_recovered() {
        // `v4l2src` reports a hot-unplug as EOS at least as often as it
        // reports an error.
        let (running, closed, generation, _) = route(
            &BusEvent::Eos {
                source: BusSource::Input,
            },
            true,
        );
        assert!(running);
        assert!(!closed);
        assert_eq!(generation, 1);
    }

    #[test]
    fn output_eos_still_ends_the_run() {
        let (running, closed, generation, _) = route(
            &BusEvent::Eos {
                source: BusSource::Output,
            },
            true,
        );
        assert!(!running);
        assert!(closed);
        assert_eq!(generation, 0);
    }

    #[test]
    fn warnings_have_no_side_effects() {
        let (running, closed, generation, latched) = route(
            &BusEvent::Warning {
                source: BusSource::Input,
                element: "input_src".into(),
                message: "something odd".into(),
                debug: None,
            },
            true,
        );
        assert!(running);
        assert!(!closed);
        assert_eq!(generation, 0);
        assert!(!latched);
    }

    #[test]
    fn a_burst_of_failures_raises_distinct_generations() {
        // The bus is drained on a timer, so losing a camera delivers
        // several messages. A boolean flag could not distinguish "the
        // failure I am already recovering from" from "another one just
        // happened", which is why this is a counter.
        let signal = InputFailureSignal::new(true);
        assert_eq!(signal.generation(), 0);
        signal.raise();
        signal.raise();
        assert_eq!(signal.generation(), 2);
    }

    // --- supervisor decision matrix ----------------------------------

    #[test]
    fn next_action_recovers_only_a_live_run_with_a_failed_input() {
        assert_eq!(next_action(true, true, false), LoopAction::ReacquireInput);
        // Clean exit of the worker loop.
        assert_eq!(next_action(true, false, false), LoopAction::Exit);
        // Ctrl-C.
        assert_eq!(next_action(false, false, false), LoopAction::Exit);
        // Shutdown wins over a concurrent input failure.
        assert_eq!(next_action(false, true, false), LoopAction::Exit);
        // A chain/output error is not recoverable by reacquiring.
        assert_eq!(next_action(true, true, true), LoopAction::TeardownAll);
        assert_eq!(next_action(true, false, true), LoopAction::TeardownAll);
    }

    // --- reap_reload -------------------------------------------------

    fn spawn_finished(
        outcome: crate::idle::reload::ResumeOutcome,
    ) -> std::thread::JoinHandle<crate::idle::reload::ResumeOutcome> {
        let handle = std::thread::spawn(move || outcome);
        // Wait so `is_finished()` is observably true before the reap.
        while !handle.is_finished() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        handle
    }

    #[test]
    fn reap_reload_none_when_empty() {
        let mut handle = None;
        assert!(reap_reload(&mut handle).is_none());
    }

    #[test]
    fn reap_reload_leaves_in_flight_handle() {
        use std::sync::mpsc;
        let (tx, rx) = mpsc::channel::<()>();
        let mut handle = Some(std::thread::spawn(move || {
            let _ = rx.recv(); // block until released
            crate::idle::reload::ResumeOutcome::Completed
        }));
        // Still running → None, and the handle must be put back.
        assert!(reap_reload(&mut handle).is_none());
        assert!(handle.is_some(), "in-flight handle must be preserved");
        tx.send(()).unwrap();
        handle.take().unwrap().join().unwrap();
    }

    #[test]
    fn reap_reload_returns_input_failed_outcome() {
        use crate::idle::reload::ResumeOutcome;
        use fluxframe_core::error::PipelineError;
        let mut handle = Some(spawn_finished(ResumeOutcome::InputFailed(FluxError::from(
            PipelineError::StateChangeFailed {
                reason: "synthetic reacquire failure".into(),
            },
        ))));
        let outcome = reap_reload(&mut handle);
        assert!(
            matches!(outcome, Some(ResumeOutcome::InputFailed(ref e)) if e.is_transient()),
            "InputFailed must be surfaced and be transient so run_auto retries, got {outcome:?}"
        );
        assert!(handle.is_none(), "reaped handle must be consumed");
    }

    #[test]
    fn reap_reload_returns_completed_outcome() {
        use crate::idle::reload::ResumeOutcome;
        let mut handle = Some(spawn_finished(ResumeOutcome::Completed));
        assert!(matches!(
            reap_reload(&mut handle),
            Some(ResumeOutcome::Completed)
        ));
        assert!(handle.is_none());
    }

    #[test]
    fn classify_input_recognises_testsrc() {
        let mut cfg = base_cfg();
        cfg.input.device = InputDevice::Testsrc;
        assert_eq!(classify_input(&cfg), InputSpec::Testsrc);
    }

    #[test]
    fn is_input_contention_retries_only_device_unavailable() {
        // Camera busy/absent → retry (the only contention class).
        let busy = FluxError::Pipeline(fluxframe_core::PipelineError::InputDeviceUnavailable {
            device: "/dev/video0".into(),
            reason: "device is busy".into(),
            hint: "another app holds it".into(),
        });
        assert!(is_input_contention(&busy));

        // A missing element or a config error is permanent → fail fast,
        // never spin forever on a real misconfiguration.
        let missing = FluxError::Pipeline(fluxframe_core::PipelineError::MissingElement {
            element: "v4l2src".into(),
            hint: "install gst-plugins-good".into(),
        });
        assert!(!is_input_contention(&missing));
        let cfg_err = FluxError::Config {
            reason: "bad".into(),
            hint: None,
        };
        assert!(!is_input_contention(&cfg_err));
    }

    #[test]
    fn supervised_acquire_requires_idle_loopback_and_v4l2_input() {
        // idle off → not supervised regardless of devices.
        let mut cfg = base_cfg();
        cfg.idle.enabled = false;
        cfg.input.device = InputDevice::Path("/dev/video0".into());
        cfg.output.device = "/dev/video10".into();
        assert!(!supervised_acquire(&cfg));

        // idle on + v4l2 loopback sink + v4l2 camera input → supervised.
        cfg.idle.enabled = true;
        assert!(supervised_acquire(&cfg));

        // testsrc input never contends → not supervised.
        cfg.input.device = InputDevice::Testsrc;
        assert!(!supervised_acquire(&cfg));
    }

    #[test]
    fn classify_input_classifies_real_device_as_v4l2() {
        let mut cfg = base_cfg();
        cfg.input.device = InputDevice::Path(PathBuf::from("/dev/video0"));
        assert_eq!(
            classify_input(&cfg),
            InputSpec::V4l2(PathBuf::from("/dev/video0"))
        );
    }

    #[test]
    fn classify_input_marks_auto_as_unsupported_escape_hatch() {
        // `Auto` must be intercepted by `commands::run` before reaching
        // `classify_input`; returning `Unsupported` is the deliberate
        // safe escape so a dispatch bug surfaces as a §27 error rather
        // than a panic.
        let mut cfg = base_cfg();
        cfg.input.device = InputDevice::Auto;
        match classify_input(&cfg) {
            InputSpec::Unsupported(s) => {
                assert!(s.contains("auto"), "diagnostic should mention auto: {s}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn resolve_output_sink_maps_auto() {
        let mut cfg = base_cfg();
        cfg.output.device = "auto".into();
        assert_eq!(
            resolve_output_sink(&cfg).expect("auto resolves"),
            OutputSink::Auto
        );
    }

    #[test]
    fn resolve_output_sink_maps_dev_path_to_v4l2_loopback() {
        let mut cfg = base_cfg();
        cfg.output.device = "/dev/video10".into();
        assert_eq!(
            resolve_output_sink(&cfg).expect("dev path resolves"),
            OutputSink::V4l2Loopback {
                device: PathBuf::from("/dev/video10"),
            }
        );
    }

    #[test]
    fn resolve_output_sink_maps_fakesink() {
        let mut cfg = base_cfg();
        cfg.output.device = "fakesink".into();
        assert_eq!(
            resolve_output_sink(&cfg).expect("fakesink resolves"),
            OutputSink::Fake
        );
    }

    #[test]
    fn resolve_output_sink_rejects_unsupported() {
        let mut cfg = base_cfg();
        cfg.output.device = "http://example.com/stream".into();
        let err = resolve_output_sink(&cfg).expect_err("unsupported sink must fail");
        let msg = format!("{err}");
        assert!(msg.contains("not supported"), "got: {msg}");
    }

    #[test]
    fn classify_output_marks_unsupported() {
        let mut cfg = base_cfg();
        cfg.output.device = "http://example.com/stream".into();
        assert_eq!(
            classify_output(&cfg),
            OutputSpec::Unsupported("http://example.com/stream".to_string())
        );
    }

    #[test]
    fn output_sink_label_covers_each_variant() {
        assert_eq!(output_sink_label(&OutputSink::Fake), "fakesink");
        assert_eq!(output_sink_label(&OutputSink::Auto), "autovideosink");
        assert_eq!(
            output_sink_label(&OutputSink::V4l2Loopback {
                device: PathBuf::from("/dev/video10"),
            }),
            "v4l2sink"
        );
    }

    #[test]
    fn classify_input_path_returns_v4l2_with_path() {
        let mut cfg = base_cfg();
        cfg.input.device = InputDevice::Path(PathBuf::from("/dev/video2"));
        assert_eq!(
            classify_input(&cfg),
            InputSpec::V4l2(PathBuf::from("/dev/video2"))
        );
    }

    #[test]
    fn classify_output_maps_auto() {
        let mut cfg = base_cfg();
        cfg.output.device = "auto".into();
        assert_eq!(classify_output(&cfg), OutputSpec::Auto);
    }

    #[test]
    fn classify_output_maps_fakesink() {
        let mut cfg = base_cfg();
        cfg.output.device = "fakesink".into();
        assert_eq!(classify_output(&cfg), OutputSpec::Fake);
    }

    #[test]
    fn classify_output_maps_dev_path() {
        let mut cfg = base_cfg();
        cfg.output.device = "/dev/video10".into();
        assert_eq!(
            classify_output(&cfg),
            OutputSpec::V4l2(PathBuf::from("/dev/video10"))
        );
    }

    #[test]
    fn classify_output_maps_pipewire_bare() {
        let mut cfg = base_cfg();
        cfg.output.device = "pipewire".into();
        assert_eq!(classify_output(&cfg), OutputSpec::Pipewire(None));
    }

    #[test]
    fn classify_output_maps_pipewire_with_name() {
        let mut cfg = base_cfg();
        cfg.output.device = "pipewire:my-cam".into();
        assert_eq!(
            classify_output(&cfg),
            OutputSpec::Pipewire(Some("my-cam".to_string()))
        );
    }

    #[test]
    fn classify_output_treats_pipewire_typos_as_unsupported() {
        // `pipewireFOO` is not a `pipewire:NAME` form — surface as
        // unsupported rather than silently treating as PipeWire with
        // garbage suffix.
        let mut cfg = base_cfg();
        cfg.output.device = "pipewirefoo".into();
        match classify_output(&cfg) {
            OutputSpec::Unsupported(s) => assert_eq!(s, "pipewirefoo"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn classify_output_pipewire_colon_with_empty_name_is_anonymous() {
        let mut cfg = base_cfg();
        cfg.output.device = "pipewire:".into();
        assert_eq!(classify_output(&cfg), OutputSpec::Pipewire(None));
    }

    #[test]
    fn resolve_output_sink_maps_pipewire() {
        let mut cfg = base_cfg();
        cfg.output.device = "pipewire:flux".into();
        assert_eq!(
            resolve_output_sink(&cfg).expect("pipewire resolves"),
            OutputSink::Pipewire {
                node_name: Some("flux".into())
            }
        );
    }

    // ----------------------------------------------------------------
    // Control-socket dispatcher tests (Stage 13)
    //
    // These exercise `apply_control_command` / `apply_set_command` /
    // `apply_get_config_command` / `apply_reload_command` directly
    // with a stub `EffectChain` (a single `PassthroughEffect`). The
    // dispatch logic does not touch GStreamer, so we can synthesise
    // the inputs without the rest of the supervisor.
    // ----------------------------------------------------------------

    use fluxframe_core::PipelineSection;
    use fluxframe_core::frame::PixelFormat;
    use fluxframe_effects::EffectChain as StubEffectChain;
    use fluxframe_effects::PassthroughEffect;

    /// Build a chain containing one `PassthroughEffect`. Sufficient
    /// for tests that exercise the dispatcher control flow without
    /// caring about per-frame processing.
    fn stub_chain() -> StubEffectChain {
        StubEffectChain::new(vec![Box::new(PassthroughEffect::new())])
    }

    fn stub_ctx() -> ProcessingContext {
        ProcessingContext {
            width: 4,
            height: 4,
            format: PixelFormat::Rgb,
            fps: 30,
            counters: None,
        }
    }

    fn ok_payload(r: &ControlResponse) -> &serde_json::Value {
        match r {
            ControlResponse::Ok { data } => data,
            ControlResponse::Err { .. } => panic!("expected Ok, got {r:?}"),
            _ => panic!("non-exhaustive Response variant: {r:?}"),
        }
    }

    fn err_reason(r: &ControlResponse) -> &str {
        match r {
            ControlResponse::Err { error, .. } => error.as_str(),
            ControlResponse::Ok { .. } => panic!("expected Err, got {r:?}"),
            _ => panic!("non-exhaustive Response variant: {r:?}"),
        }
    }

    #[test]
    fn apply_control_command_list_presets_returns_names() {
        let mut cfg = base_cfg();
        cfg.presets.insert("a".into(), Preset::default());
        cfg.presets.insert("b".into(), Preset::default());
        let mut chain = stub_chain();
        let mut name = "a".to_string();
        let mut preset = Preset::default();
        let ctx = stub_ctx();
        let resp = apply_control_command(
            &mut cfg,
            None,
            &ctx,
            &mut chain,
            &mut name,
            &mut preset,
            ControlCommand::ListPresets,
        );
        let data = ok_payload(&resp);
        let arr = data.as_array().expect("ListPresets returns array");
        let names: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"a"), "names: {names:?}");
        assert!(names.contains(&"b"), "names: {names:?}");
    }

    #[test]
    fn apply_list_effects_returns_inventory_for_all_sections() {
        use fluxframe_effects::{mask_effects, plane_effects};

        let resp = apply_list_effects_command();
        let data = ok_payload(&resp);
        let obj = data.as_object().expect("ListEffects returns object");
        // All four section keys present, even under slim build.
        for section in ["mask", "background", "foreground", "post"] {
            let arr = obj
                .get(section)
                .and_then(serde_json::Value::as_array)
                .unwrap_or_else(|| panic!("section '{section}' missing or not array"));
            for row in arr {
                let row_obj = row.as_object().expect("metadata is object");
                assert!(row_obj.contains_key("name"));
                assert!(row_obj.contains_key("params"));
            }
        }
        // mask/background/foreground always populated; `post` only
        // under the `ml` feature.
        for section in ["mask", "background", "foreground"] {
            assert!(
                !obj[section].as_array().unwrap().is_empty(),
                "section '{section}' must list at least one effect"
            );
        }
        #[cfg(feature = "ml")]
        assert!(
            !obj["post"].as_array().unwrap().is_empty(),
            "post must populate under ml feature"
        );
        // Spot-check a known effect.
        let mask_spot: Vec<_> = obj["mask"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v["name"].as_str())
            .collect();
        assert!(mask_spot.contains(&"threshold"), "got: {mask_spot:?}");

        // Cross-check each section against the registry's own
        // `names()` so a future drift between the registry and the
        // response (duplicate registrations, missing entries, order
        // changes) trips the test rather than silently shipping.
        let mask_names: Vec<&str> = obj["mask"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v["name"].as_str())
            .collect();
        assert_eq!(
            mask_names,
            mask_effects::default_registry().names(),
            "ListEffects mask list must equal default_registry().names()",
        );
        let bg_names: Vec<&str> = obj["background"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v["name"].as_str())
            .collect();
        assert_eq!(
            bg_names,
            plane_effects::default_registry().names(),
            "ListEffects background list must equal plane registry names()",
        );
        let fg_names: Vec<&str> = obj["foreground"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v["name"].as_str())
            .collect();
        assert_eq!(
            fg_names,
            plane_effects::default_registry().names(),
            "ListEffects foreground list must equal plane registry names()",
        );
        #[cfg(feature = "ml")]
        {
            use fluxframe_effects::post_effects;
            let post_names: Vec<&str> = obj["post"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|v| v["name"].as_str())
                .collect();
            assert_eq!(
                post_names,
                post_effects::default_registry().names(),
                "ListEffects post list must equal post registry names() under ml",
            );
        }
        // Contract: `build_features` is always present so the GUI
        // can distinguish slim-by-build from missing-by-config.
        assert!(
            obj.contains_key("build_features"),
            "build_features must be present in ListEffects response",
        );
        assert!(
            obj["build_features"].is_array(),
            "build_features must be an array",
        );
    }

    #[test]
    fn apply_list_effects_foreground_equals_background_today() {
        // foreground and background share the same plane registry, so
        // their inventories MUST match element-by-element. If the
        // codebase ever introduces foreground-only or background-only
        // effects, this test fails — at which point the GUI contract
        // (and the comment near the `background.clone()` call) need
        // updating.
        let resp = apply_list_effects_command();
        let data = ok_payload(&resp);
        let obj = data.as_object().expect("ListEffects returns object");
        let bg = obj["background"].as_array().expect("background is array");
        let fg = obj["foreground"].as_array().expect("foreground is array");
        assert_eq!(
            bg.len(),
            fg.len(),
            "foreground and background must have the same length today",
        );
        for (i, (b, f)) in bg.iter().zip(fg.iter()).enumerate() {
            assert_eq!(
                b, f,
                "foreground[{i}] must equal background[{i}] (shared plane registry)",
            );
        }
    }

    #[test]
    fn apply_control_command_set_preset_unknown_name_lists_available() {
        let mut cfg = base_cfg();
        cfg.presets.insert("alpha".into(), Preset::default());
        let mut chain = stub_chain();
        let mut name = "alpha".to_string();
        let mut preset = Preset::default();
        let ctx = stub_ctx();
        let resp = apply_control_command(
            &mut cfg,
            None,
            &ctx,
            &mut chain,
            &mut name,
            &mut preset,
            ControlCommand::SetPreset {
                name: "missing".into(),
            },
        );
        match resp {
            ControlResponse::Err { error, hint } => {
                assert!(error.contains("missing"), "error: {error}");
                let hint = hint.expect("hint must list available presets");
                assert!(hint.contains("alpha"), "hint: {hint}");
            }
            ControlResponse::Ok { .. } => panic!("expected Err"),
            _ => panic!("non-exhaustive Response variant"),
        }
    }

    #[test]
    fn apply_set_command_revert_on_invalid_value() {
        // Stub chain has no composite, so `reconfigure_named_effect`
        // returns Err for any path. The in-memory preset must NOT
        // grow the new key — the revert path puts it back the way it
        // was.
        let mut chain = stub_chain();
        let mut preset = Preset {
            background: Some(PipelineSection::default()),
            ..Preset::default()
        };
        let before = preset.background.as_ref().unwrap().per_effect.clone();
        let resp = apply_set_command(
            &mut chain,
            &mut preset,
            "background.blur.radius",
            serde_json::json!(42),
        );
        assert!(matches!(resp, ControlResponse::Err { .. }));
        let after = preset.background.as_ref().unwrap().per_effect.clone();
        assert_eq!(
            before, after,
            "in-memory preset must be unchanged after revert"
        );
    }

    #[test]
    fn apply_get_config_command_walks_nested_path() {
        // Build a preset with a known nested shape so we can walk
        // into it via the dot-path.
        let section = PipelineSection {
            chain: vec!["blur".into()],
            ..PipelineSection::default()
        };
        let preset = Preset {
            background: Some(section),
            ..Preset::default()
        };
        let resp = apply_get_config_command(&preset, Some("background.chain"));
        let data = ok_payload(&resp);
        let arr = data.as_array().expect("chain returns array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0].as_str(), Some("blur"));
    }

    #[test]
    fn apply_get_config_command_rejects_empty_component() {
        let preset = Preset::default();
        let resp = apply_get_config_command(&preset, Some("background..chain"));
        let reason = err_reason(&resp);
        assert!(reason.contains("empty"), "reason: {reason}");
    }

    #[test]
    fn ensure_plane_or_post_section_mut_materialises_absent_post() {
        let mut p = fluxframe_core::Preset::default();
        assert!(p.post.is_none(), "fixture preset has no [post]");
        let section = ensure_plane_or_post_section_mut(&mut p, SubchainKind::Post);
        assert!(
            section.chain.is_empty(),
            "newly-materialised chain is empty"
        );
        assert!(section.per_effect.is_empty(), "no per-effect overrides");
        assert!(p.post.is_some(), "section persisted into preset");
    }

    #[test]
    fn ensure_plane_or_post_section_mut_returns_existing_background() {
        let mut p = fluxframe_core::Preset {
            background: Some(fluxframe_core::PipelineSection {
                chain: vec!["blur".to_string()],
                ..Default::default()
            }),
            ..Default::default()
        };
        let section = ensure_plane_or_post_section_mut(&mut p, SubchainKind::Background);
        assert_eq!(
            section.chain,
            vec!["blur".to_string()],
            "existing chain preserved"
        );
    }

    #[test]
    fn mask_section_mut_does_not_materialise_absent() {
        let mut p = fluxframe_core::Preset::default();
        assert!(mask_section_mut(&mut p).is_none(), "absent mask stays None");
        assert!(p.mask.is_none(), "preset.mask is not mutated");
    }

    #[test]
    fn mask_section_mut_returns_existing() {
        let mut p = fluxframe_core::Preset {
            mask: Some(fluxframe_core::PipelineSection {
                chain: vec!["threshold".to_string()],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(mask_section_mut(&mut p).is_some());
    }

    #[test]
    fn apply_reload_command_without_path_errors() {
        let mut cfg = base_cfg();
        let mut chain = stub_chain();
        let mut name = "default".to_string();
        let mut preset = Preset::default();
        let ctx = stub_ctx();
        let resp = apply_reload_command(&mut cfg, None, &ctx, &mut chain, &mut name, &mut preset);
        let reason = err_reason(&resp);
        assert!(reason.contains("config path"), "reason: {reason}");
    }
}

#[cfg(all(test, feature = "ml", feature = "image-fill"))]
mod default_synthesis_tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn synthesises_color_fill_defaults_when_per_effect_missing() {
        let per_effect = BTreeMap::new();
        let result = build_and_configure_subchain(
            SubchainKind::Background,
            &["color_fill".to_string()],
            &per_effect,
        );
        assert!(
            result.is_ok(),
            "color_fill should succeed with metadata-synthesised defaults; got {:?}",
            result.as_ref().err().map(|e| &e.reason),
        );
    }

    #[test]
    fn rejects_image_fill_with_helpful_message() {
        let per_effect = BTreeMap::new();
        // `SubChainPayload` does not implement `Debug`, so we cannot
        // use `expect_err`; use `let...else` instead.
        let Err(err) = build_and_configure_subchain(
            SubchainKind::Background,
            &["image_fill".to_string()],
            &per_effect,
        ) else {
            panic!("image_fill should fail without an explicit path")
        };
        assert!(
            err.reason.contains("missing required field"),
            "expected missing-field error, got: {}",
            err.reason,
        );
        assert!(
            err.reason.contains("path"),
            "expected `path` mentioned in error, got: {}",
            err.reason,
        );
        assert!(
            err.hint.is_some(),
            "expected hint with editor suggestion, got reason={} hint={:?}",
            err.reason,
            err.hint,
        );
    }
}
