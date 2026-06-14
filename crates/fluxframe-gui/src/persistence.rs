//! Lightweight window-geometry persistence.
//!
//! We deliberately avoid `gio::Settings` here because that would
//! require an installed GSchema XML file at
//! `~/.local/share/glib-2.0/schemas/io.fluxframe.gui.gschema.xml`
//! plus a `glib-compile-schemas` step. A plain JSON file in
//! `$XDG_CONFIG_HOME/fluxframe/gui.json` is good enough for a single
//! window's width/height/maximised state and works out of the box
//! without any installer scaffolding.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Minimum window dimensions enforced on load. Below this gtk would
/// refuse to map or display a 1×1 dot.
const MIN_WIDTH: i32 = 200;
const MIN_HEIGHT: i32 = 150;
/// Upper guard against a hand-edited gui.json with absurd values.
/// 8192 covers any reasonable desktop resolution while leaving plenty
/// of headroom.
const MAX_WIDTH: i32 = 8192;
const MAX_HEIGHT: i32 = 8192;

/// Persisted window geometry — what we record between runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WindowState {
    /// Width in CSS pixels (gtk's logical unit).
    pub width: i32,
    /// Height in CSS pixels.
    pub height: i32,
    /// Whether the window was maximised when it last closed.
    pub maximized: bool,
}

impl Default for WindowState {
    fn default() -> Self {
        Self {
            width: 720,
            height: 540,
            maximized: false,
        }
    }
}

/// Clamp width/height into the supported bounds. Self-healing for
/// pathological values coming from a corrupt or hand-edited gui.json
/// (negative, zero, or absurdly large).
fn sanitize(mut state: WindowState) -> WindowState {
    state.width = state.width.clamp(MIN_WIDTH, MAX_WIDTH);
    state.height = state.height.clamp(MIN_HEIGHT, MAX_HEIGHT);
    state
}

// Linux-only by construction: XDG_CONFIG_HOME / $HOME/.config.
// The project targets Linux only (see CLAUDE.md); a future Windows
// port should consult dirs::config_dir() instead.
/// Resolve the on-disk path for the persisted state.
///
/// Honours `$XDG_CONFIG_HOME` when set; falls back to
/// `$HOME/.config`. The `fluxframe` subdirectory is created on first
/// save.
fn config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(config_path_in(&base))
}

/// Build the on-disk path inside an explicit XDG config root.
///
/// Split out from [`config_path`] so unit tests can target a
/// scratch dir without mutating the process-wide `XDG_CONFIG_HOME`
/// env var — the workspace forbids `unsafe_code`, which rules out the
/// Rust 2024 `std::env::set_var` (now `unsafe`).
fn config_path_in(base: &Path) -> PathBuf {
    base.join("fluxframe").join("gui.json")
}

/// Read the persisted window state from disk.
///
/// Returns the default on any failure (missing file, unreadable,
/// malformed JSON). All failures are logged at debug level so
/// first-run noise stays quiet. Successfully loaded state is clamped
/// to `[MIN_*, MAX_*]` bounds so hand-edited absurdities never reach
/// gtk.
pub(crate) fn load() -> WindowState {
    let Some(path) = config_path() else {
        tracing::debug!("no XDG_CONFIG_HOME or HOME — skipping load");
        return WindowState::default();
    };
    load_from(&path)
}

/// Read the persisted window state from an explicit path.
///
/// Test seam — see [`config_path_in`] for why this split exists.
fn load_from(path: &Path) -> WindowState {
    match std::fs::read_to_string(path) {
        Ok(s) => match serde_json::from_str::<WindowState>(&s) {
            Ok(state) => {
                tracing::debug!(?path, ?state, "loaded window state");
                sanitize(state)
            }
            Err(e) => {
                tracing::debug!(error = %e, ?path, "window state parse failed, using default");
                WindowState::default()
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => WindowState::default(),
        Err(e) => {
            tracing::debug!(error = %e, ?path, "window state read failed, using default");
            WindowState::default()
        }
    }
}

/// Persist the window state to disk.
///
/// Creates the parent directory if it does not exist. Writes go
/// through a sibling `*.tmp` file followed by `rename` so a crash or
/// power loss mid-write cannot corrupt the live `gui.json`; the
/// `sanitize` step in `load` heals any leftover damage from a
/// pre-atomic-write file. Failures are logged at warn level
/// (persistence is best-effort, not critical).
pub(crate) fn save(state: &WindowState) {
    let Some(path) = config_path() else {
        return;
    };
    save_to(&path, state);
}

/// Persist the window state to an explicit path.
///
/// Test seam — see [`config_path_in`] for why this split exists.
fn save_to(path: &Path, state: &WindowState) {
    if let Some(parent) = path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        tracing::warn!(error = %e, ?parent, "failed to create config dir");
        return;
    }
    let payload = match serde_json::to_string_pretty(state) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "window state serialise failed");
            return;
        }
    };
    let tmp_path = path.with_extension("json.tmp");
    if let Err(e) = std::fs::write(&tmp_path, payload) {
        tracing::warn!(error = %e, ?tmp_path, "window state write failed");
        return;
    }
    if let Err(e) = std::fs::rename(&tmp_path, path) {
        tracing::warn!(error = %e, ?path, "window state rename failed");
        // Best-effort cleanup; ignore failure.
        let _ = std::fs::remove_file(&tmp_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests deliberately do NOT touch process env vars. The
    // workspace pins `unsafe_code = "forbid"`, which makes the
    // Rust 2024 `std::env::set_var` (now `unsafe`) unusable here.
    // Instead each test uses the `config_path_in` / `load_from` /
    // `save_to` seam to drive an isolated tempdir directly. That
    // also sidesteps the cargo-parallel-tests race that XDG env
    // mutation would otherwise introduce.

    #[test]
    fn load_missing_returns_default() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let path = config_path_in(tmp.path());
        let state = load_from(&path);
        let def = WindowState::default();
        assert_eq!(state.width, def.width);
        assert_eq!(state.height, def.height);
        assert_eq!(state.maximized, def.maximized);
    }

    #[test]
    fn save_then_load_roundtrip() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let path = config_path_in(tmp.path());
        let original = WindowState {
            width: 1024,
            height: 768,
            maximized: true,
        };
        save_to(&path, &original);
        let loaded = load_from(&path);
        assert_eq!(loaded.width, original.width);
        assert_eq!(loaded.height, original.height);
        assert_eq!(loaded.maximized, original.maximized);
    }

    #[test]
    fn load_corrupt_json_returns_default() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let path = config_path_in(tmp.path());
        let dir = path.parent().expect("path has parent");
        std::fs::create_dir_all(dir).expect("create fluxframe dir");
        std::fs::write(&path, "not valid json").expect("write corrupt json");
        let state = load_from(&path);
        let def = WindowState::default();
        assert_eq!(state.width, def.width);
        assert_eq!(state.height, def.height);
        assert_eq!(state.maximized, def.maximized);
    }

    #[test]
    fn load_clamps_negative_dimensions() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let path = config_path_in(tmp.path());
        let dir = path.parent().expect("path has parent");
        std::fs::create_dir_all(dir).expect("create fluxframe dir");
        std::fs::write(&path, r#"{"width": -5, "height": 0, "maximized": false}"#)
            .expect("write json");
        let state = load_from(&path);
        assert_eq!(state.width, MIN_WIDTH);
        assert_eq!(state.height, MIN_HEIGHT);
        assert!(!state.maximized);
    }

    #[test]
    fn load_clamps_huge_dimensions() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let path = config_path_in(tmp.path());
        let dir = path.parent().expect("path has parent");
        std::fs::create_dir_all(dir).expect("create fluxframe dir");
        std::fs::write(
            &path,
            r#"{"width": 999999, "height": 888888, "maximized": true}"#,
        )
        .expect("write json");
        let state = load_from(&path);
        assert_eq!(state.width, MAX_WIDTH);
        assert_eq!(state.height, MAX_HEIGHT);
        assert!(state.maximized);
    }

    #[test]
    fn save_atomic_write_does_not_leave_tmp_on_success() {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let path = config_path_in(tmp.path());
        let state = WindowState {
            width: 800,
            height: 600,
            maximized: false,
        };
        save_to(&path, &state);
        let dir = path.parent().expect("path has parent");
        let entries: Vec<_> = std::fs::read_dir(dir)
            .expect("read fluxframe dir")
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            entries.iter().any(|n| n == "gui.json"),
            "gui.json should exist after save, got: {entries:?}",
        );
        assert!(
            !entries.iter().any(|n| std::path::Path::new(n)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("tmp"))),
            "no .tmp file should remain after successful save, got: {entries:?}",
        );
    }
}
