//! Application state owned by the root AppModel and read by the
//! header bar and the chain editor.
//!
//! Kept as a plain struct (no `Rc<RefCell>`); relm4's `update()`
//! mutates it directly on the UI thread. All config-tree transitions
//! (active vs. baseline) live here so they are testable without GTK.

use std::collections::BTreeMap;
use std::path::PathBuf;

use fluxframe_core::{EffectSchema, OutputInfo, SetPath, SubchainKind};
use serde_json::{Map, Value};

/// Inventory of available effects per sub-chain, as reported by the
/// daemon's `list_effects` command.
#[derive(Debug, Default, Clone)]
pub(crate) struct EffectInventory {
    /// One entry per [`SubchainKind`].
    pub(crate) sections: BTreeMap<SubchainKind, Vec<EffectSchema>>,
    /// Cargo features the daemon was built with (`"ml"`,
    /// `"image-fill"`). Shown in the About dialog's debug info so a
    /// slim daemon can be told apart from a misconfigured one.
    pub(crate) build_features: Vec<String>,
}

/// Connection state for the IPC link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConnectionStatus {
    /// Before the first connect attempt completes (or after Retry).
    Connecting,
    /// Handshake succeeded; the cached `EffectInventory`, preset
    /// list, and active preset name are now authoritative.
    Connected,
    /// IPC failed; the reason is shown on the status page.
    Disconnected,
}

/// What selecting a preset should do, given the current state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PresetSwitch {
    /// The preset is already active — nothing to send, nothing to ask.
    AlreadyActive,
    /// Unsaved edits would be lost — ask the operator first.
    NeedsConfirmation,
    /// Switch right away.
    Proceed,
}

/// A change to one sub-chain's effect list. Indices are 0-based
/// positions in the chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ChainEdit {
    /// Append the named effect at the tail.
    Append(String),
    /// Remove the effect at the index.
    Remove(usize),
    /// Swap the effect at the index with its predecessor.
    MoveUp(usize),
    /// Swap the effect at the index with its successor.
    MoveDown(usize),
}

impl ChainEdit {
    /// Apply the edit to `chain`. Returns `false` (leaving `chain`
    /// untouched) when the edit is a no-op, e.g. an index out of range
    /// or moving the head up.
    pub(crate) fn apply(self, chain: &mut Vec<String>) -> bool {
        match self {
            Self::Append(effect) => chain.push(effect),
            Self::Remove(index) if index < chain.len() => {
                chain.remove(index);
            }
            Self::MoveUp(index) if index > 0 && index < chain.len() => {
                chain.swap(index - 1, index);
            }
            Self::MoveDown(index) if index + 1 < chain.len() => {
                chain.swap(index, index + 1);
            }
            Self::Remove(_) | Self::MoveUp(_) | Self::MoveDown(_) => return false,
        }
        true
    }
}

/// How a fresh `get_config` snapshot relates to the dirty baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfigSync {
    /// The snapshot reflects in-memory daemon state that may carry
    /// unsaved edits (e.g. the refetch after `set_chain`): replace
    /// `active_config` only, so the dirty marker survives.
    KeepBaseline,
    /// The snapshot is a clean starting point (handshake, preset
    /// switch, reload, revert): replace both trees, clearing dirty.
    ResyncBaseline,
}

/// What the GUI showed right before the daemon link dropped; compared
/// against the next handshake by [`AppState::apply_handshake`].
#[derive(Debug, Clone)]
pub(crate) struct LostSession {
    active_preset: Option<String>,
    active_config: Value,
    baseline_config: Value,
}

/// Application state, owned by the root `AppModel`.
#[derive(Debug)]
pub(crate) struct AppState {
    /// Path to the daemon socket (resolved from CLI/env at startup).
    pub(crate) socket_path: PathBuf,
    /// Current connection status.
    pub(crate) status: ConnectionStatus,
    /// Preset names returned by `list_presets`. Empty until the
    /// handshake completes.
    pub(crate) presets: Vec<String>,
    /// Currently-active preset name, per `current_preset`.
    pub(crate) active_preset: Option<String>,
    /// Effect inventory, per `list_effects`.
    pub(crate) inventory: EffectInventory,
    /// Active preset's full configuration, as returned by
    /// `get_config { path: None }`. Stored as the raw JSON tree the
    /// chain editor walks (`background.chain`,
    /// `background.<effect>.<field>` — the daemon's `per_effect` map
    /// is flattened into the section).
    pub(crate) active_config: Value,
    /// Last snapshot of `active_config` known to match what the
    /// daemon has persisted. [`is_dirty`](Self::is_dirty) compares
    /// `active_config` against this snapshot to drive the Save /
    /// Revert button sensitivity and the title-bar dirty marker.
    pub(crate) baseline_config: Value,
    /// Writable TOML path reported by the daemon's `ConfigPath`
    /// command, or `None` when the daemon was started with no
    /// resolvable config (e.g. `--no-default-config` and no
    /// `--config`). The Save button is greyed out in that case.
    pub(crate) config_path: Option<PathBuf>,
    /// Where the daemon publishes video, per `daemon_info`; `None` for a
    /// daemon too old to report it.
    pub(crate) output: Option<OutputInfo>,
}

impl AppState {
    /// Build a fresh `AppState` for the given socket path. All other
    /// fields start empty; the worker fills them on `Connected`.
    #[must_use]
    pub(crate) fn new(socket_path: PathBuf) -> Self {
        Self {
            socket_path,
            status: ConnectionStatus::Connecting,
            presets: Vec::new(),
            active_preset: None,
            inventory: EffectInventory::default(),
            active_config: Value::Null,
            baseline_config: Value::Null,
            config_path: None,
            output: None,
        }
    }

    /// `true` when [`active_config`](Self::active_config) has diverged
    /// from [`baseline_config`](Self::baseline_config) — i.e. the
    /// operator changed something via the GUI that has not been
    /// persisted (or reverted) yet.
    #[must_use]
    pub(crate) fn is_dirty(&self) -> bool {
        self.active_config != self.baseline_config
    }

    /// Whether the handshake has completed and the link is up.
    #[must_use]
    pub(crate) fn is_connected(&self) -> bool {
        self.status == ConnectionStatus::Connected
    }

    /// Decide how to handle a request to activate preset `name`.
    #[must_use]
    pub(crate) fn preset_switch(&self, name: &str) -> PresetSwitch {
        if self.active_preset.as_deref() == Some(name) {
            PresetSwitch::AlreadyActive
        } else if self.is_dirty() {
            PresetSwitch::NeedsConfirmation
        } else {
            PresetSwitch::Proceed
        }
    }

    /// Install a `get_config` snapshot according to `sync`.
    pub(crate) fn apply_config_snapshot(&mut self, data: Value, sync: ConfigSync) {
        if sync == ConfigSync::ResyncBaseline {
            self.baseline_config = data.clone();
        }
        self.active_config = data;
    }

    /// Snapshot of the session to compare the next handshake against
    /// after the link drops; see [`Self::apply_handshake`].
    pub(crate) fn lost_session(&self) -> LostSession {
        LostSession {
            active_preset: self.active_preset.clone(),
            active_config: self.active_config.clone(),
            baseline_config: self.baseline_config.clone(),
        }
    }

    /// Install the active preset and config reported by a handshake.
    ///
    /// On a first connect (`lost` is `None`) the snapshot is the clean
    /// baseline. After a reconnect, unsaved edits the daemon still holds
    /// — the same preset with the same config as before the drop — stay
    /// dirty against the old baseline. Returns `true` when unsaved edits
    /// the operator saw before the drop are gone.
    pub(crate) fn apply_handshake(
        &mut self,
        active_preset: String,
        config: Value,
        lost: Option<&LostSession>,
    ) -> bool {
        let dirty_before = lost.filter(|l| l.active_config != l.baseline_config);
        let survived = dirty_before.filter(|l| {
            l.active_preset.as_deref() == Some(active_preset.as_str()) && l.active_config == config
        });
        if let Some(l) = survived {
            self.baseline_config = l.baseline_config.clone();
            self.active_config = config;
        } else {
            self.apply_config_snapshot(config, ConfigSync::ResyncBaseline);
        }
        self.active_preset = Some(active_preset);
        dirty_before.is_some() && survived.is_none()
    }

    /// The daemon persisted the active preset: the in-memory state is
    /// the new baseline.
    pub(crate) fn mark_saved(&mut self) {
        self.baseline_config = self.active_config.clone();
    }

    /// Effects the daemon offers for `section` (empty before the
    /// handshake).
    pub(crate) fn effects_for(&self, section: SubchainKind) -> &[EffectSchema] {
        self.inventory
            .sections
            .get(&section)
            .map_or(&[][..], Vec::as_slice)
    }

    /// Look up a single per-effect parameter in the active config.
    /// Returns the raw JSON `Value` or `Null` if absent.
    ///
    /// Walks `<section>.<effect>.<field>`.
    pub(crate) fn config_field(&self, path: &SetPath) -> &Value {
        const NULL: Value = Value::Null;
        self.active_config
            .get(path.section.as_str())
            .and_then(|s| s.get(&path.effect))
            .and_then(|e| e.get(&path.field))
            .unwrap_or(&NULL)
    }

    /// Write `value` at `<section>.<effect>.<field>` in the active
    /// config, creating (or replacing non-object) intermediate nodes.
    /// Mirrors a `set` the daemon accepted so rebuilds render the new
    /// value without another `get_config` round-trip.
    pub(crate) fn set_config_field(&mut self, path: &SetPath, value: Value) {
        let keys = [path.section.as_str(), path.effect.as_str()];
        let mut node = &mut self.active_config;
        for key in keys {
            node = object_mut(node)
                .entry(key)
                .or_insert_with(|| Value::Object(Map::new()));
        }
        object_mut(node).insert(path.field.clone(), value);
    }

    /// Whether `effect` in `section` runs: its table's `enabled` key,
    /// `true` when absent (the daemon only stores `enabled = false`).
    pub(crate) fn effect_enabled(&self, section: SubchainKind, effect: &str) -> bool {
        self.active_config
            .get(section.as_str())
            .and_then(|s| s.get(effect))
            .and_then(|e| e.get(fluxframe_core::EFFECT_ENABLED_KEY))
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }

    /// Mirror a `set_enabled` the daemon accepted, in the daemon's
    /// canonical form: `enabled = false` in the effect table when
    /// disabled; no key when enabled, dropping a table the key leaves
    /// empty. Matching that form keeps the dirty marker exact without a
    /// page rebuild.
    pub(crate) fn set_effect_enabled(
        &mut self,
        section: SubchainKind,
        effect: &str,
        enabled: bool,
    ) {
        let key = fluxframe_core::EFFECT_ENABLED_KEY;
        if enabled {
            let Some(section_obj) = self
                .active_config
                .get_mut(section.as_str())
                .and_then(Value::as_object_mut)
            else {
                return;
            };
            let emptied = section_obj
                .get_mut(effect)
                .and_then(Value::as_object_mut)
                .is_some_and(|table| {
                    table.remove(key);
                    table.is_empty()
                });
            if emptied {
                section_obj.remove(effect);
            }
        } else {
            let mut node = &mut self.active_config;
            for segment in [section.as_str(), effect] {
                node = object_mut(node)
                    .entry(segment)
                    .or_insert_with(|| Value::Object(Map::new()));
            }
            object_mut(node).insert(key.to_string(), Value::Bool(false));
        }
    }

    /// Chain of effect names for `section`. Empty if the section is
    /// absent or has no `chain` array.
    pub(crate) fn chain_for(&self, section: SubchainKind) -> Vec<String> {
        self.active_config
            .get(section.as_str())
            .and_then(|s| s.get("chain"))
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Coerce `node` into a JSON object (discarding a non-object value)
/// and return its map.
fn object_mut(node: &mut Value) -> &mut Map<String, Value> {
    if !node.is_object() {
        *node = Value::Object(Map::new());
    }
    match node {
        Value::Object(map) => map,
        _ => unreachable!("node coerced to object just above"),
    }
}

/// Test fixture: the `get_config` payload the daemon produces for a
/// preset written as `preset_toml` (sections at the top level, as under
/// `[presets.NAME]`). Goes through `fluxframe_core::Preset` exactly like
/// the daemon's `serde_json::to_value(active_preset)`, so GUI tests
/// cannot drift from the real wire shape.
#[cfg(test)]
pub(crate) fn daemon_config(preset_toml: &str) -> Value {
    let preset: fluxframe_core::Preset = toml::from_str(preset_toml).expect("valid preset TOML");
    serde_json::to_value(preset).expect("preset serialises")
}

#[cfg(test)]
mod tests {
    use fluxframe_core::parse_set_path;

    use super::*;

    fn path(s: &str) -> SetPath {
        parse_set_path(s).expect("valid test path")
    }

    fn fixture() -> AppState {
        let mut s = AppState::new(PathBuf::from("/tmp/test.sock"));
        s.active_config = daemon_config(
            r#"
[background]
chain = ["blur", "vignette"]

[background.blur]
radius = 30
passes = 2

[background.vignette]
strength = 0.4

[foreground]
chain = []
"#,
        );
        s.baseline_config = s.active_config.clone();
        s
    }

    /// Regression: `get_config` flattens `per_effect` into the section
    /// (`background.blur.radius`); walking a `per_effect` level made
    /// every widget fall back to its metadata default.
    #[test]
    fn config_field_reads_daemon_get_config_shape() {
        let s = fixture();
        assert!(s.active_config["background"].get("per_effect").is_none());
        assert_eq!(s.config_field(&path("background.blur.passes")), 2);
        assert_eq!(s.config_field(&path("background.vignette.strength")), 0.4);
    }

    #[test]
    fn config_field_returns_present_value() {
        let s = fixture();
        let v = s.config_field(&path("background.blur.radius"));
        assert_eq!(v.as_i64(), Some(30));
    }

    #[test]
    fn config_field_returns_null_for_missing_section() {
        let s = fixture();
        assert!(s.config_field(&path("post.auto_frame.threshold")).is_null());
    }

    #[test]
    fn config_field_returns_null_for_missing_effect() {
        let s = fixture();
        assert!(s.config_field(&path("background.unknown.field")).is_null());
    }

    #[test]
    fn config_field_returns_null_for_missing_field() {
        let s = fixture();
        assert!(
            s.config_field(&path("background.blur.nonexistent"))
                .is_null()
        );
    }

    #[test]
    fn chain_for_returns_names_in_order() {
        let s = fixture();
        assert_eq!(
            s.chain_for(SubchainKind::Background),
            vec!["blur".to_string(), "vignette".to_string()]
        );
    }

    #[test]
    fn chain_for_empty_section_returns_empty_vec() {
        let s = fixture();
        assert!(s.chain_for(SubchainKind::Foreground).is_empty());
    }

    #[test]
    fn chain_for_missing_section_returns_empty_vec() {
        let s = fixture();
        assert!(s.chain_for(SubchainKind::Post).is_empty());
    }

    #[test]
    fn chain_for_handles_non_array_chain_gracefully() {
        let mut s = AppState::new(PathBuf::from("/tmp/test.sock"));
        s.active_config = serde_json::json!({
            "mask": { "chain": "not-an-array" }
        });
        assert!(s.chain_for(SubchainKind::Mask).is_empty());
    }

    #[test]
    fn set_config_field_updates_existing_value_and_marks_dirty() {
        let mut s = fixture();
        s.set_config_field(&path("background.blur.radius"), serde_json::json!(12));
        assert_eq!(s.config_field(&path("background.blur.radius")), 12);
        assert!(s.is_dirty());
    }

    #[test]
    fn set_config_field_creates_missing_and_replaces_non_object_nodes() {
        let mut s = AppState::new(PathBuf::from("/tmp/test.sock"));
        s.active_config = serde_json::json!({ "post": "garbage" });
        s.set_config_field(&path("post.mirror.flip"), Value::Bool(true));
        assert_eq!(
            s.active_config,
            serde_json::json!({ "post": { "mirror": { "flip": true } } })
        );
    }

    /// Regression: the refetch after `set_chain` returns the daemon's
    /// in-memory preset, which already carries unsaved `set` edits.
    /// It must not become the baseline, or the dirty marker (and the
    /// discard-changes confirmation) silently disappears.
    #[test]
    fn keep_baseline_snapshot_preserves_unsaved_edits() {
        let mut s = fixture();
        s.set_config_field(&path("background.blur.radius"), serde_json::json!(12));
        let daemon_view = s.active_config.clone();
        s.apply_config_snapshot(daemon_view, ConfigSync::KeepBaseline);
        assert!(s.is_dirty(), "unsaved edit must stay dirty after refetch");
    }

    #[test]
    fn resync_baseline_snapshot_clears_dirty() {
        let mut s = fixture();
        s.set_config_field(&path("background.blur.radius"), serde_json::json!(12));
        s.apply_config_snapshot(
            serde_json::json!({ "mask": {} }),
            ConfigSync::ResyncBaseline,
        );
        assert!(!s.is_dirty());
        assert_eq!(s.active_config, serde_json::json!({ "mask": {} }));
    }

    /// Regression: re-selecting the active preset (e.g. the drop-down
    /// echo after a programmatic selection) must neither prompt nor
    /// send `set_preset`, even with unsaved edits.
    #[test]
    fn preset_switch_to_active_preset_is_a_no_op_even_when_dirty() {
        let mut s = fixture();
        s.active_preset = Some("a".into());
        s.set_config_field(&path("background.blur.radius"), serde_json::json!(12));
        assert_eq!(s.preset_switch("a"), PresetSwitch::AlreadyActive);
        assert_eq!(s.preset_switch("b"), PresetSwitch::NeedsConfirmation);
        s.mark_saved();
        assert_eq!(s.preset_switch("b"), PresetSwitch::Proceed);
    }

    fn chain(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| (*n).to_string()).collect()
    }

    #[test]
    fn chain_edit_applies_in_range_edits() {
        let mut c = chain(&["a", "b", "c"]);
        assert!(ChainEdit::MoveUp(2).apply(&mut c));
        assert_eq!(c, chain(&["a", "c", "b"]));
        assert!(ChainEdit::MoveDown(0).apply(&mut c));
        assert_eq!(c, chain(&["c", "a", "b"]));
        assert!(ChainEdit::Remove(1).apply(&mut c));
        assert_eq!(c, chain(&["c", "b"]));
        assert!(ChainEdit::Append("d".into()).apply(&mut c));
        assert_eq!(c, chain(&["c", "b", "d"]));
    }

    #[test]
    fn chain_edit_rejects_boundary_and_out_of_range_edits() {
        let mut c = chain(&["a", "b"]);
        assert!(!ChainEdit::MoveUp(0).apply(&mut c));
        assert!(!ChainEdit::MoveUp(5).apply(&mut c));
        assert!(!ChainEdit::MoveDown(1).apply(&mut c));
        assert!(!ChainEdit::Remove(2).apply(&mut c));
        assert_eq!(c, chain(&["a", "b"]));
    }

    #[test]
    fn effect_enabled_defaults_to_true_and_reads_the_flag() {
        let mut s = fixture();
        s.active_config = daemon_config(
            r#"
[background]
chain = ["blur", "vignette"]

[background.blur]
enabled = false
radius = 30
"#,
        );
        assert!(!s.effect_enabled(SubchainKind::Background, "blur"));
        assert!(s.effect_enabled(SubchainKind::Background, "vignette"));
        assert!(s.effect_enabled(SubchainKind::Post, "mirror"));
    }

    /// The local mirror must produce exactly what the daemon's
    /// `get_config` returns after the same toggles, or the dirty marker
    /// drifts until the next refetch.
    #[test]
    fn set_effect_enabled_matches_the_daemon_canonical_form() {
        let mut s = fixture();
        let bg = SubchainKind::Background;
        let original = s.active_config.clone();

        // Effect without a table: off creates `{ enabled = false }`.
        s.set_effect_enabled(bg, "color_fill", false);
        assert_eq!(
            s.active_config["background"],
            daemon_config(
                r#"
[background]
chain = ["blur", "vignette"]
[background.blur]
radius = 30
passes = 2
[background.vignette]
strength = 0.4
[background.color_fill]
enabled = false
"#
            )["background"]
        );
        assert!(s.is_dirty());

        // Effect with params: the table stays, only the key toggles.
        s.set_effect_enabled(bg, "blur", false);
        assert!(!s.effect_enabled(bg, "blur"));
        s.set_effect_enabled(bg, "blur", true);
        assert_eq!(s.config_field(&path("background.blur.radius")), 30);
        assert!(
            s.active_config["background"]["blur"]
                .get("enabled")
                .is_none()
        );

        // Back on: the flag-only table disappears and dirty clears.
        s.set_effect_enabled(bg, "color_fill", true);
        assert_eq!(s.active_config, original);
        assert!(!s.is_dirty());
    }

    #[test]
    fn first_handshake_is_a_clean_baseline() {
        let mut s = AppState::new(PathBuf::from("/tmp/test.sock"));
        let edits_lost = s.apply_handshake("a".into(), serde_json::json!({"mask": {}}), None);
        assert!(!edits_lost);
        assert!(!s.is_dirty());
        assert_eq!(s.active_preset.as_deref(), Some("a"));
    }

    /// The link dropped but the daemon kept running: its in-memory preset
    /// still carries the edits, so they must stay marked unsaved.
    #[test]
    fn reconnect_keeps_edits_the_daemon_still_holds() {
        let mut s = fixture();
        s.active_preset = Some("a".into());
        s.set_config_field(&path("background.blur.radius"), serde_json::json!(12));
        let lost = s.lost_session();
        let daemon_view = s.active_config.clone();
        let edits_lost = s.apply_handshake("a".into(), daemon_view, Some(&lost));
        assert!(!edits_lost);
        assert!(s.is_dirty(), "edits the daemon kept stay unsaved");
    }

    /// The daemon restarted from its file: the edits are gone, and the
    /// operator must be told.
    #[test]
    fn reconnect_reports_edits_lost_to_a_daemon_restart() {
        let mut s = fixture();
        s.active_preset = Some("a".into());
        let saved = s.baseline_config.clone();
        s.set_config_field(&path("background.blur.radius"), serde_json::json!(12));
        let lost = s.lost_session();
        let edits_lost = s.apply_handshake("a".into(), saved.clone(), Some(&lost));
        assert!(edits_lost);
        assert!(!s.is_dirty());
        assert_eq!(s.active_config, saved);
    }

    #[test]
    fn reconnect_without_unsaved_edits_reports_nothing() {
        let mut s = fixture();
        s.active_preset = Some("a".into());
        let lost = s.lost_session();
        let edits_lost =
            s.apply_handshake("b".into(), serde_json::json!({"mask": {}}), Some(&lost));
        assert!(!edits_lost);
        assert!(!s.is_dirty());
        assert_eq!(s.active_preset.as_deref(), Some("b"));
    }

    #[test]
    fn mark_saved_clears_dirty() {
        let mut s = fixture();
        s.set_config_field(&path("background.blur.radius"), serde_json::json!(12));
        s.mark_saved();
        assert!(!s.is_dirty());
    }
}
