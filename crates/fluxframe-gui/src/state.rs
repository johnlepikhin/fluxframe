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
#[allow(dead_code, reason = "fields consumed by Stage 14 Step 5 chain editor")]
#[derive(Debug, Default, Clone)]
pub(crate) struct EffectInventory {
    /// One key per [`fluxframe_core::SubchainKind`]
    /// (`"mask"`, `"background"`, `"foreground"`, `"post"`).
    pub(crate) sections: BTreeMap<String, Vec<EffectMetadata>>,
    /// Cargo features the daemon was built with (`"ml"`,
    /// `"image-fill"`). Lets the GUI distinguish a slim daemon from a
    /// misconfigured one.
    pub(crate) build_features: Vec<String>,
}

/// Connection state for the IPC link.
#[allow(
    dead_code,
    reason = "variants inspected by Stage 14 Step 5 chain editor"
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
        }
    }
}
