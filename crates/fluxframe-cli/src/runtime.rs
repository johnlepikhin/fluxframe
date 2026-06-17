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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::PipelineError;
use fluxframe_core::{FluxConfig, FluxError, InputDevice, Preset, SubchainKind};
use fluxframe_effects::EffectChain;
use fluxframe_gst::input::{InputParams, InputPipeline};
use fluxframe_gst::output::{OutputParams, OutputPipeline, OutputSink};
use fluxframe_gst::{BusEvent, BusListener, BusSource, LatestFrameSlot, WatchedPipeline};
use tracing::{error, info, warn};

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
        // `ControlCommand` is `#[non_exhaustive]`; a future variant
        // that this build does not yet handle gets a structured
        // error rather than panicking the worker thread.
        _ => ControlResponse::err(
            "unknown command — daemon was built without support for this wire variant",
            Some("update the daemon, or check the client is not ahead of the daemon".into()),
        ),
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

    fn configure_each<E>(
        chain: &mut [Box<E>],
        names: &[String],
        per_effect: &std::collections::BTreeMap<String, toml::Value>,
    ) -> Result<(), fluxframe_core::EffectError>
    where
        E: ?Sized + Configure,
    {
        for (effect, name) in chain.iter_mut().zip(names.iter()) {
            let params = per_effect
                .get(name)
                .cloned()
                .unwrap_or_else(|| toml::Value::Table(toml::Table::new()));
            effect.configure_mut(params)?;
        }
        Ok(())
    }
    match section {
        SubchainKind::Mask => {
            let mut built = mask_effects::default_registry()
                .build_chain(names)
                .map_err(|e| SubChainError {
                    reason: format!("mask chain build failed: {e}"),
                    hint: None,
                })?;
            configure_each::<dyn fluxframe_core::MaskEffect>(&mut built, names, per_effect)
                .map_err(|e| SubChainError {
                    reason: format!("configure failed: {e}"),
                    hint: None,
                })?;
            Ok(SubChainPayload::Mask(built))
        }
        SubchainKind::Background => {
            let mut built = plane_effects::default_registry()
                .build_chain(names)
                .map_err(|e| SubChainError {
                    reason: format!("background chain build failed: {e}"),
                    hint: None,
                })?;
            configure_each::<dyn fluxframe_core::PlaneEffect>(&mut built, names, per_effect)
                .map_err(|e| SubChainError {
                    reason: format!("configure failed: {e}"),
                    hint: None,
                })?;
            Ok(SubChainPayload::Background(built))
        }
        SubchainKind::Foreground => {
            let mut built = plane_effects::default_registry()
                .build_chain(names)
                .map_err(|e| SubChainError {
                    reason: format!("foreground chain build failed: {e}"),
                    hint: None,
                })?;
            configure_each::<dyn fluxframe_core::PlaneEffect>(&mut built, names, per_effect)
                .map_err(|e| SubChainError {
                    reason: format!("configure failed: {e}"),
                    hint: None,
                })?;
            Ok(SubChainPayload::Foreground(built))
        }
        SubchainKind::Post => {
            let mut built = post_effects::default_registry()
                .build_chain(names)
                .map_err(|e| SubChainError {
                    reason: format!("post chain build failed: {e}"),
                    hint: None,
                })?;
            configure_each::<dyn fluxframe_core::PostEffect>(&mut built, names, per_effect)
                .map_err(|e| SubChainError {
                    reason: format!("configure failed: {e}"),
                    hint: None,
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
    // The `v4l::Device` opened inside `detect_native_mode` is dropped
    // at end-of-call (before the closure below opens the device via
    // `v4l2src`).  Do not lift this call into the closure or a struct
    // field — that would race v4l2src on VIDIOC_REQBUFS.
    let detected = fluxframe_gst::v4l2_caps::detect_native_mode(&device_path, cfg.input.fps)?;
    info!(
        device = %device_path.display(),
        cfg_w = cfg.input.width,
        cfg_h = cfg.input.height,
        cfg_fps = cfg.input.fps,
        ?cfg.input.format,
        detected_w = detected.width,
        detected_h = detected.height,
        detected_fps = detected.fps,
        detected_format = ?detected.format,
        "auto-detected v4l2 camera native mode"
    );

    // Only fps is taken from the camera; width/height stay at the
    // operator-configured values.
    let mut resolved = cfg.clone();
    resolved.input.fps = detected.fps;

    let device_path_for_builder = device_path.clone();
    run_chain(
        &resolved,
        preset_name,
        config_path,
        chain,
        "v4l2src",
        move |params| InputPipeline::build_v4l2(&device_path_for_builder, params),
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
    F: FnOnce(InputParams) -> Result<InputPipeline, PipelineError>,
{
    fluxframe_gst::init()?;

    // Build metrics BEFORE `prepare_all` so the runtime can hand the
    // shared `Arc<Counters>` to effects through `ProcessingContext`.
    // This lets, e.g., the Stage 7 sticky-fallback decorator inside
    // `BackgroundBlurEffect`'s blur backend publish transition events
    // through the same counter bundle the supervisor reads later.
    let metrics = RuntimeMetrics::new();

    let (input, output, mut processing_ctx, sink_label) = build_pipelines(cfg, input_builder)?;
    // Stage 15 needs to share `input` with the reload thread, so the
    // supervisor wraps it in an Arc up-front. `&*input` keeps the
    // legacy `&InputPipeline` ergonomics for the rest of this fn.
    let input: Arc<InputPipeline> = Arc::new(input);
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
    input.start()?;

    let slot = input.slot();
    // The token guard MUST live until the end of this function: when it
    // drops it removes the run's weak entry from the global registry.
    let (_token_guard, running) = register_token(slot.clone());

    // Bus listener relays GStreamer fatal errors and EOS into the shared
    // shutdown flag, and surfaces any captured error back to the caller.
    let bus_error: Arc<Mutex<Option<FluxError>>> = Arc::new(Mutex::new(None));
    let _bus_listener = build_bus_listener(
        &input,
        &output,
        Arc::clone(&running),
        Arc::clone(&bus_error),
        slot.clone(),
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
    let idle_runtime = build_idle_runtime(cfg, &input, &running);

    let process_result = run_process_loop(
        WorkerDeps {
            cfg,
            config_path,
            initial_preset_name: preset_name,
            processing_ctx: &processing_ctx,
            metrics: &metrics,
        },
        &running,
        &slot,
        &mut chain,
        &output,
        control_rx.as_ref(),
        idle_runtime,
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
    // Emit per-field structured values so log-ingestors (ELK, Vector,
    // tracing-subscriber JSON layer) can query each metric without
    // string-parsing.  Mirrors the periodic reporter's per-tick line
    // (`metrics_reporter::emit`) but covers absolute totals at
    // teardown, not the rolling-window deltas the reporter shows.
    info!(
        frames_in = snap.counters.frames_in,
        frames_out = snap.counters.frames_out,
        frames_dropped = snap.counters.frames_dropped,
        fallback = snap.counters.fallback_count,
        effect_err = snap.counters.effect_error_count,
        processing_p50_us = snap.processing.percentile_us(0.5),
        processing_p95_us = snap.processing.percentile_us(0.95),
        output_p50_us = snap.output.percentile_us(0.5),
        output_p95_us = snap.output.percentile_us(0.95),
        end_to_end_p50_us = snap.end_to_end.percentile_us(0.5),
        end_to_end_p95_us = snap.end_to_end.percentile_us(0.95),
        "run metrics"
    );

    // Surface bus-reported errors when the processing loop itself was
    // clean, so the operator sees the real cause of shutdown.
    let bus_err = bus_error.lock().expect("bus_error mutex poisoned").take();
    match (process_result, bus_err) {
        (Ok(()), Some(e)) | (Err(e), _) => Err(e),
        (Ok(()), None) => Ok(()),
    }
}

fn build_pipelines<F>(
    cfg: &FluxConfig,
    input_builder: F,
) -> Result<
    (
        InputPipeline,
        OutputPipeline,
        ProcessingContext,
        &'static str,
    ),
    FluxError,
>
where
    F: FnOnce(InputParams) -> Result<InputPipeline, PipelineError>,
{
    let input_params = InputParams::new(
        cfg.input.width,
        cfg.input.height,
        cfg.input.fps,
        cfg.input.format,
    );
    let sink = resolve_output_sink(cfg)?;
    let sink_label = output_sink_label(&sink);
    // Sink dimensions = input × `output.scale`, rounded to even
    // pixels.  For v4l2 cameras the supervisor has already overridden
    // `cfg.input.width/height` with the auto-detected native mode, so
    // this multiplication produces the right output size for the
    // operator's chosen scale.  fps is inherited verbatim — operator
    // cannot misconfigure the output to a different rate (which would
    // silently insert videorate, duplicating or dropping frames).
    let (sink_w, sink_h) = cfg
        .output
        .effective_dimensions(cfg.input.width, cfg.input.height);
    let output_params = OutputParams::new(sink_w, sink_h, cfg.input.fps, cfg.input.format, sink)
        .with_sink_format(cfg.output.format);

    let input = input_builder(input_params)?;
    let output = OutputPipeline::build(output_params)?;

    let processing_ctx = ProcessingContext {
        width: cfg.input.width,
        height: cfg.input.height,
        format: cfg.input.format,
        fps: cfg.input.fps,
        // The caller (`run_chain`) fills in the supervisor's
        // `Arc<Counters>` after `build_pipelines` returns.  `None`
        // here is the correct default for any standalone caller that
        // doesn't run a full supervisor.
        counters: None,
    };

    Ok((input, output, processing_ctx, sink_label))
}

/// Build the [`BusListener`] watching both pipelines.  The listener is
/// returned so the caller can keep it alive (its `Drop` joins the thread).
fn build_bus_listener(
    input: &InputPipeline,
    output: &OutputPipeline,
    running: Arc<AtomicBool>,
    bus_error: Arc<Mutex<Option<FluxError>>>,
    slot: LatestFrameSlot,
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
        on_bus_event(&event, &running, &bus_error, &slot);
    })
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
) {
    match event {
        BusEvent::FatalError {
            element,
            message,
            debug: debug_payload,
            source,
        } => {
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

    if state.frames_seen % DROPPED_SYNC_INTERVAL == 0 {
        metrics.sync_dropped(slot.dropped_count());
    }
    Ok(())
}

/// Push a pre-rendered placeholder frame to the output, bypassing
/// the effect chain. Used by the worker loop while in Idle or
/// DeepIdle to keep v4l2loopback's ring buffer fresh without paying
/// the cost of the full pipeline.
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
    /// `chain.process` — true in Active, cleared on EnterDeepIdle,
    /// re-armed by the reload thread.
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
    running: &Arc<AtomicBool>,
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

        let detector_running = Arc::clone(running);
        let detector = crate::idle::ConsumerDetector::spawn(
            device_path,
            my_pid,
            std::time::Duration::from_millis(u64::from(cfg.idle.poll_interval_ms)),
            detector_running,
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
        let _ = (input, running);
        warn!(
            target: "fluxframe::idle",
            "idle mode requested but host is not Linux — idle disabled (the /proc/*/fd consumer detector is Linux-only)"
        );
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
        IdleEdge::EnterDeepIdle => {
            metrics.counters.inc_deep_idle_entered();
            info!(target: "fluxframe::idle", "entering deep idle");
            // Stage 15 Step 4 ships the visible idle behaviour; the
            // ONNX-engine unload that brings DeepIdle's RAM
            // reclamation to life is parked here until the
            // `EffectChain` ↔ `ManagedComposite` wiring lands.
            // engine_ready stays `true` so the worker keeps running
            // the full chain — DeepIdle is currently observationally
            // identical to Idle.
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
            // no-op here because Step 4 does not yet unload the ONNX
            // session; the input.start() inside `spawn_reload_thread`
            // is the meaningful work.
            // Fix #5: double-spawn guard. If a previous reload thread
            // is still running we keep the existing handle and return
            // early — kicking off a second reload while the first is
            // mid-`input.start()` would race the GStreamer state
            // transitions and likely produce a wedged input. Tested
            // shape: this path is currently observed only via logs;
            // a synthetic unit test would need a stalled reload
            // closure and a full `IdleRuntime` (Arc<dyn Placeholder>,
            // OutputPipeline, detector handle) which is integration
            // territory.
            // TODO(stage-15-followup): add a focused unit test by
            // extracting a `double_spawn_guard(handle) -> Option<handle>`
            // helper that does not depend on the full IdleRuntime.
            let prior = idle.reload_handle.take();
            if let Some(handle) = prior {
                if !handle.is_finished() {
                    warn!("a previous reload thread is still in flight; not spawning a second one");
                    idle.reload_handle = Some(handle);
                    return;
                }
                // Reap the outcome so the JoinHandle does not leak.
                let _ = handle.join();
            }
            // Fix #4: race between the `is_finished` check above and
            // this `store(false)` is narrow but real — a prior reload
            // that finished between the check and the store loses its
            // `engine_ready = true` write, so the worker spends one
            // extra placeholder cycle before the new reload thread
            // re-flips it. The clean fix is to push the `store(false)`
            // into `spawn_reload_thread`'s body so the supervisor never
            // touches `engine_ready` directly, but `idle/reload.rs` is
            // owned by a sibling work item.
            // TODO(stage-15-followup): move `engine_ready.store(false)`
            // into `spawn_reload_thread`'s spawned closure (first
            // action, before `input.start()`) and drop this line. The
            // observable cost today is at most one extra placeholder
            // push per ResumeActive that races a just-finished reload
            // — counted in `idle_frames_pushed`, harmless.
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
) -> Result<IdleTickAction, FluxError> {
    let Some(idle_rt) = idle else {
        return Ok(IdleTickAction::ProcessFrame);
    };
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
}

fn run_process_loop(
    deps: WorkerDeps<'_>,
    running: &AtomicBool,
    slot: &LatestFrameSlot,
    chain: &mut EffectChain,
    output: &OutputPipeline,
    control_rx: Option<&crossbeam_channel::Receiver<ControlEnvelope>>,
    mut idle: Option<IdleRuntime>,
) -> Result<(), FluxError> {
    let mut state = WorkerState::new(deps.cfg, deps.initial_preset_name, deps.metrics);
    while running.load(Ordering::Acquire) {
        drain_control_commands(
            &mut state,
            control_rx,
            deps.config_path,
            deps.processing_ctx,
            chain,
        );

        // Stage 15 idle integration: tick the state machine, dispatch
        // any side-effect edge, choose between Active level (pull a
        // real frame + run chain) and Placeholder level (push the
        // cached fill).
        match tick_idle(idle.as_mut(), &state, slot, output, deps.metrics)? {
            IdleTickAction::Continue => continue,
            IdleTickAction::ProcessFrame => {}
        }

        let Some(frame) = slot.recv_timeout(WORKER_POLL_TIMEOUT) else {
            // Either timeout (no frame within the poll window) or slot
            // closed by shutdown.  Re-check the flag and continue.
            continue;
        };
        process_one_frame(&mut state, frame, chain, output, deps.metrics, slot)?;
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
                        elapsed_ms = outcome.elapsed.as_millis() as u64,
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

    #[test]
    fn classify_input_recognises_testsrc() {
        let mut cfg = base_cfg();
        cfg.input.device = InputDevice::Testsrc;
        assert_eq!(classify_input(&cfg), InputSpec::Testsrc);
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
