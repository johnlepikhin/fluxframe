//! Wire format for the control socket: typed [`Command`] and
//! [`Response`] enums, line-delimited JSON.
//!
//! Each socket connection follows a strict request/response cadence:
//! the client writes one JSON object per line, the daemon writes one
//! JSON object per line back, repeat until either side closes the
//! socket.
//!
//! These types live in `fluxframe-core` so that both the daemon
//! (`fluxframe-cli`) and a GUI client (`fluxframe-gui`) can share the
//! exact same vocabulary without depending on the binary crate.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::SubchainKind;

/// All commands accepted by the control socket.
///
/// Discriminated on the `"cmd"` field with `snake_case` rename. The
/// `deny_unknown_fields` attribute keeps a payload typo (e.g.
/// `{"cmd":"set","pat":"..."}`) from being silently accepted as a
/// missing-field-defaults command.
///
/// ```json
/// {"cmd":"set_preset","name":"blur"}
/// {"cmd":"set","path":"background.blur.radius","value":40}
/// {"cmd":"set_chain","section":"background","chain":["blur","vignette"]}
/// ```
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[non_exhaustive]
#[serde(tag = "cmd", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    /// List the names of every preset defined in the loaded config.
    ListPresets,
    /// Return the name of the currently-active preset.
    CurrentPreset,
    /// List every effect available in each sub-chain registry, with
    /// its [`crate::metadata::EffectMetadata`] descriptor. The GUI
    /// uses this on startup to build param widgets without hard-
    /// coding the inventory.
    ListEffects,
    /// Dump the active preset (or a sub-path) as a JSON value.
    /// `path` follows the dot-syntax used by `Set` (e.g.
    /// `"background.blur"`); `null` means the whole preset.
    GetConfig {
        /// Dot-path into the active preset; `None` returns the whole
        /// preset.
        #[serde(default)]
        path: Option<String>,
    },
    /// Swap the active preset for the named one. Triggers a full
    /// composite rebuild (may take 100–500 ms when the model
    /// changes).
    SetPreset {
        /// Name of the preset to activate.
        name: String,
    },
    /// Update a single field. `path` is the dot-syntax
    /// `<section>.<effect>.<field>` (e.g.
    /// `"background.blur.radius"`); `<section>` ∈
    /// `mask|background|foreground|post`. `value` is any JSON value
    /// the effect's serde schema accepts.
    Set {
        /// Dot-path into the active preset.
        path: String,
        /// New value (typed as `serde_json::Value`, converted to
        /// `toml::Value` at apply time).
        value: serde_json::Value,
    },
    /// Replace the entire chain composition of a sub-section. Effects
    /// listed get built from the corresponding registry, configured
    /// from the preset's existing `per_effect` payload (or defaults
    /// when absent), prepared against the active resolution, and
    /// swapped in.
    SetChain {
        /// One of `mask|background|foreground|post` (parsed via
        /// [`SubchainKind`] at the wire boundary).
        section: SubchainKind,
        /// New chain (effect names in order).
        chain: Vec<String>,
    },
    /// Re-read the TOML config from disk and reapply the current
    /// preset against the freshly-parsed config. Equivalent to
    /// "edit the file then `set_preset <currently active>`".
    Reload,
}

/// Response body returned for each command.
///
/// Encoded as `{"ok":"true","data":...}` or
/// `{"ok":"false","error":"...","hint":"..."}`. The string
/// discriminator on `ok` keeps the wire format scriptable from shell
/// without parsing nested error variants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
#[serde(tag = "ok")]
pub enum Response {
    /// Successful response. `data` may be `Null`/an object/an array
    /// depending on the command.
    #[serde(rename = "true")]
    Ok {
        /// Optional payload (e.g. list of preset names for
        /// [`Command::ListPresets`]).
        #[serde(default = "null_value")]
        data: serde_json::Value,
    },
    /// Failure. `error` is a one-line `reason`; `hint` mirrors
    /// `FluxError`'s `Diagnostic::hint`.
    #[serde(rename = "false")]
    Err {
        /// One-line reason.
        error: String,
        /// Optional hint at how to fix it.
        #[serde(skip_serializing_if = "Option::is_none")]
        hint: Option<String>,
    },
}

fn null_value() -> serde_json::Value {
    serde_json::Value::Null
}

impl Response {
    /// Convenience: empty-OK response (no payload).
    #[must_use]
    pub fn ok() -> Self {
        Self::Ok {
            data: serde_json::Value::Null,
        }
    }

    /// Convenience: OK with a JSON payload.
    #[must_use]
    pub fn ok_with(data: serde_json::Value) -> Self {
        Self::Ok { data }
    }

    /// Convenience: error response with reason + optional hint.
    #[must_use]
    pub fn err(error: impl Into<String>, hint: Option<String>) -> Self {
        Self::Err {
            error: error.into(),
            hint,
        }
    }
}

/// A parsed `<section>.<effect>.<field>` path.
#[derive(Debug, Clone, PartialEq)]
pub struct SetPath {
    /// Sub-chain identifier parsed from the first dot-component.
    pub section: SubchainKind,
    /// Effect name inside the sub-chain (snake_case).
    pub effect: String,
    /// Field name on the effect's config struct.
    pub field: String,
}

/// Parse the dot-syntax expected by the `set` command.
///
/// Accepts exactly three components — `<section>.<effect>.<field>`.
/// Errors out on the wrong number of dots or an empty component.
///
/// # Errors
///
/// Returns a one-line reason string suitable for [`Response::err`].
pub fn parse_set_path(path: &str) -> Result<SetPath, String> {
    let parts: Vec<&str> = path.split('.').collect();
    if parts.len() != 3 {
        return Err(format!(
            "expected `<section>.<effect>.<field>`, got '{path}' ({} components)",
            parts.len()
        ));
    }
    if parts.iter().any(|s| s.is_empty()) {
        return Err(format!("empty component in path '{path}'"));
    }
    let section: SubchainKind = parts[0]
        .parse()
        .map_err(|e: String| format!("{e} in path '{path}'"))?;
    Ok(SetPath {
        section,
        effect: parts[1].to_string(),
        field: parts[2].to_string(),
    })
}

/// Resolve a default socket path from an explicit `$XDG_RUNTIME_DIR`
/// value. Splitting this pure helper from the env-reading wrapper
/// [`default_socket_path`] keeps the workspace's `forbid(unsafe_code)`
/// invariant intact — env mutation in tests would require unsafe.
#[must_use]
pub fn resolve_socket_path(xdg: Option<&str>) -> PathBuf {
    match xdg {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir).join("fluxframe.sock"),
        _ => PathBuf::from("/tmp/fluxframe.sock"),
    }
}

/// Resolve the default socket path from `$XDG_RUNTIME_DIR`. Used by
/// both the daemon (to decide where to bind) and the GUI client (to
/// decide where to connect) when no explicit path is configured.
#[must_use]
pub fn default_socket_path() -> PathBuf {
    resolve_socket_path(std::env::var("XDG_RUNTIME_DIR").ok().as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_presets_parses() {
        let cmd: Command = serde_json::from_str(r#"{"cmd":"list_presets"}"#).expect("parses");
        assert_eq!(cmd, Command::ListPresets);
    }

    #[test]
    fn list_effects_parses() {
        let cmd: Command = serde_json::from_str(r#"{"cmd":"list_effects"}"#).expect("parses");
        assert_eq!(cmd, Command::ListEffects);
    }

    #[test]
    fn set_preset_parses() {
        let cmd: Command =
            serde_json::from_str(r#"{"cmd":"set_preset","name":"blur"}"#).expect("parses");
        assert_eq!(
            cmd,
            Command::SetPreset {
                name: "blur".into(),
            }
        );
    }

    #[test]
    fn set_parses_with_number() {
        let cmd: Command =
            serde_json::from_str(r#"{"cmd":"set","path":"background.blur.radius","value":40}"#)
                .expect("parses");
        match cmd {
            Command::Set { path, value } => {
                assert_eq!(path, "background.blur.radius");
                assert_eq!(value, serde_json::json!(40));
            }
            other => panic!("expected Set, got {other:?}"),
        }
    }

    #[test]
    fn set_chain_parses_array() {
        let cmd: Command = serde_json::from_str(
            r#"{"cmd":"set_chain","section":"background","chain":["blur","vignette"]}"#,
        )
        .expect("parses");
        match cmd {
            Command::SetChain { section, chain } => {
                assert_eq!(section, SubchainKind::Background);
                assert_eq!(chain, vec!["blur", "vignette"]);
            }
            other => panic!("expected SetChain, got {other:?}"),
        }
    }

    #[test]
    fn set_chain_rejects_unknown_section() {
        let res: Result<Command, _> =
            serde_json::from_str(r#"{"cmd":"set_chain","section":"bogus","chain":["blur"]}"#);
        assert!(res.is_err(), "unknown section must be rejected");
    }

    #[test]
    fn command_rejects_unknown_field() {
        // `deny_unknown_fields` on the enum catches typos in payload
        // keys (e.g. `pat` instead of `path`).
        let res: Result<Command, _> =
            serde_json::from_str(r#"{"cmd":"set","pat":"background.blur.radius","value":40}"#);
        assert!(res.is_err(), "typo'd field must be rejected");
    }

    #[test]
    fn get_config_parses_with_optional_path() {
        let cmd: Command = serde_json::from_str(r#"{"cmd":"get_config"}"#).expect("parses");
        assert_eq!(cmd, Command::GetConfig { path: None });

        let cmd: Command = serde_json::from_str(r#"{"cmd":"get_config","path":"background.blur"}"#)
            .expect("parses");
        assert_eq!(
            cmd,
            Command::GetConfig {
                path: Some("background.blur".into())
            }
        );
    }

    #[test]
    fn unknown_command_rejected() {
        let res: Result<Command, _> = serde_json::from_str(r#"{"cmd":"nope"}"#);
        assert!(res.is_err());
    }

    #[test]
    fn response_ok_serialises_as_true() {
        let r = Response::ok();
        let s = serde_json::to_string(&r).expect("serialise");
        assert!(s.contains("\"ok\":\"true\""), "got: {s}");
    }

    #[test]
    fn response_err_includes_hint_when_present() {
        let r = Response::err("nope", Some("fix it".into()));
        let s = serde_json::to_string(&r).expect("serialise");
        assert!(s.contains("\"ok\":\"false\""));
        assert!(s.contains("\"hint\":\"fix it\""));
    }

    #[test]
    fn parse_set_path_happy() {
        let p = parse_set_path("background.blur.radius").expect("ok");
        assert_eq!(p.section, SubchainKind::Background);
        assert_eq!(p.effect, "blur");
        assert_eq!(p.field, "radius");
    }

    #[test]
    fn parse_set_path_rejects_wrong_component_count() {
        assert!(parse_set_path("background.blur").is_err());
        assert!(parse_set_path("background.blur.radius.extra").is_err());
        assert!(parse_set_path("").is_err());
    }

    #[test]
    fn parse_set_path_rejects_unknown_section() {
        let err = parse_set_path("nope.blur.radius").unwrap_err();
        assert!(err.contains("nope"), "got: {err}");
    }

    #[test]
    fn parse_set_path_rejects_empty_component() {
        assert!(parse_set_path("background..radius").is_err());
        assert!(parse_set_path(".blur.radius").is_err());
        assert!(parse_set_path("background.blur.").is_err());
    }

    #[test]
    fn resolve_socket_path_uses_xdg_when_present() {
        let p = resolve_socket_path(Some("/run/user/1000"));
        assert_eq!(p, PathBuf::from("/run/user/1000/fluxframe.sock"));
    }

    #[test]
    fn resolve_socket_path_falls_back_to_tmp_for_empty_xdg() {
        let p = resolve_socket_path(Some(""));
        assert_eq!(p, PathBuf::from("/tmp/fluxframe.sock"));
        let p = resolve_socket_path(None);
        assert_eq!(p, PathBuf::from("/tmp/fluxframe.sock"));
    }
}
