//! Application state shared between the IPC worker, the header bar,
//! and (eventually) the chain editor.
//!
//! Kept as a plain struct (no `Rc<RefCell>`); relm4's `update()`
//! mutates it directly on the UI thread.

use std::collections::BTreeMap;
use std::path::PathBuf;

use fluxframe_core::EffectMetadata;

/// Inventory of available effects per sub-chain, as reported by the
/// daemon's `list_effects` command.
#[derive(Debug, Default, Clone)]
pub(crate) struct EffectInventory {
    /// One key per [`fluxframe_core::SubchainKind`]
    /// (`"mask"`, `"background"`, `"foreground"`, `"post"`).
    pub(crate) sections: BTreeMap<String, Vec<EffectMetadata>>,
    /// Cargo features the daemon was built with (`"ml"`,
    /// `"image-fill"`). Lets the GUI distinguish a slim daemon from a
    /// misconfigured one.
    #[allow(
        dead_code,
        reason = "surfaced by Stage 14 Step 7's 'About daemon' dialog"
    )]
    pub(crate) build_features: Vec<String>,
}

/// Connection state for the IPC link.
#[allow(
    dead_code,
    reason = "variants inspected by Step 7 'connection status' indicator"
)]
#[derive(Debug, Clone)]
pub(crate) enum ConnectionStatus {
    /// Initial state before the first connect attempt completes.
    Connecting,
    /// Handshake succeeded; the cached `EffectInventory`, preset
    /// list, and active preset name are now authoritative.
    Connected,
    /// IPC failed; the inner string is a one-line human-readable
    /// reason suitable for an [`adw::StatusPage`].
    Disconnected(String),
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
    /// `background.per_effect.<name>.<field>`).
    pub(crate) active_config: serde_json::Value,
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
            active_config: serde_json::Value::Null,
        }
    }

    /// Look up a single per-effect parameter in the active config.
    /// Returns the raw JSON `Value` or `Null` if absent.
    ///
    /// Walks `<section>.per_effect.<effect>.<field>`. Used by the
    /// chain editor to seed widget initial values without re-querying
    /// the daemon.
    pub(crate) fn config_field(
        &self,
        section: &str,
        effect: &str,
        field: &str,
    ) -> &serde_json::Value {
        const NULL: serde_json::Value = serde_json::Value::Null;
        self.active_config
            .get(section)
            .and_then(|s| s.get("per_effect"))
            .and_then(|p| p.get(effect))
            .and_then(|e| e.get(field))
            .unwrap_or(&NULL)
    }

    /// Chain of effect names for the given section (`"mask"`,
    /// `"background"`, `"foreground"`, `"post"`). Empty if the
    /// section is absent or has no `chain` array.
    pub(crate) fn chain_for(&self, section: &str) -> Vec<String> {
        self.active_config
            .get(section)
            .and_then(|s| s.get("chain"))
            .and_then(|c| c.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> AppState {
        let mut s = AppState::new(PathBuf::from("/tmp/test.sock"));
        s.active_config = serde_json::json!({
            "background": {
                "chain": ["blur", "vignette"],
                "per_effect": {
                    "blur": { "radius": 30, "passes": 2 },
                    "vignette": { "strength": 0.4 }
                }
            },
            "foreground": {
                "chain": []
            }
        });
        s
    }

    #[test]
    fn config_field_returns_present_value() {
        let s = fixture();
        let v = s.config_field("background", "blur", "radius");
        assert_eq!(v.as_i64(), Some(30));
    }

    #[test]
    fn config_field_returns_null_for_missing_section() {
        let s = fixture();
        let v = s.config_field("post", "auto_frame", "threshold");
        assert!(v.is_null());
    }

    #[test]
    fn config_field_returns_null_for_missing_per_effect() {
        let s = fixture();
        let v = s.config_field("background", "unknown_effect", "field");
        assert!(v.is_null());
    }

    #[test]
    fn config_field_returns_null_for_missing_field() {
        let s = fixture();
        let v = s.config_field("background", "blur", "nonexistent");
        assert!(v.is_null());
    }

    #[test]
    fn chain_for_returns_names_in_order() {
        let s = fixture();
        assert_eq!(
            s.chain_for("background"),
            vec!["blur".to_string(), "vignette".to_string()]
        );
    }

    #[test]
    fn chain_for_empty_section_returns_empty_vec() {
        let s = fixture();
        assert!(s.chain_for("foreground").is_empty());
    }

    #[test]
    fn chain_for_missing_section_returns_empty_vec() {
        let s = fixture();
        assert!(s.chain_for("nonexistent").is_empty());
    }

    #[test]
    fn chain_for_handles_non_array_chain_gracefully() {
        let mut s = AppState::new(PathBuf::from("/tmp/test.sock"));
        s.active_config = serde_json::json!({
            "mask": { "chain": "not-an-array" }
        });
        assert!(s.chain_for("mask").is_empty());
    }
}
