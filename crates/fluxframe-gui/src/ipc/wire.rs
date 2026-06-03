//! On-wire decoders for the daemon's `list_effects` / `list_presets`
//! handshake payloads.
//!
//! The daemon serialises [`EffectMetadata`] using `&'static str` fields
//! that live in `const` storage. On the client side those strings
//! arrive as owned `String`s, so we re-allocate and then leak them
//! into `'static` to reconstruct the same shape.
//!
//! Leaked memory is bounded per [`crate::ipc::worker::WorkerOutput::Connected`]
//! event: one inventory worth of strings per successful handshake.
//! **Every Retry click in the AppModel re-runs the handshake and
//! therefore re-leaks the full inventory.** That is an acceptable
//! trade-off for a session-bound utility (the daemon's metadata is on
//! the order of a few KB), but an operator debugging permission
//! errors across many retries should be aware of it.

use fluxframe_core::EffectMetadata;
use serde::Deserialize;

use crate::state::EffectInventory;

/// Decode the `list_effects` payload into [`EffectInventory`].
///
/// Walks the `{mask, background, foreground, post, build_features}`
/// object and deserialises every `EffectMetadata` entry on the wire.
/// Returns a descriptive error when any field is missing or
/// mistyped — those are protocol violations that the GUI cannot
/// recover from.
pub fn parse_inventory(data: &serde_json::Value) -> Result<EffectInventory, String> {
    let obj = data
        .as_object()
        .ok_or_else(|| "list_effects payload is not an object".to_string())?;
    let mut sections = std::collections::BTreeMap::new();
    for key in ["mask", "background", "foreground", "post"] {
        let arr = obj
            .get(key)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("list_effects missing section '{key}'"))?;
        let mut effects = Vec::with_capacity(arr.len());
        for item in arr {
            let meta = WireMetadata::deserialize(item)
                .map_err(|e| format!("section '{key}': metadata decode failed: {e}"))?;
            effects.push(
                meta.into_owned()
                    .map_err(|e| format!("section '{key}': {e}"))?,
            );
        }
        sections.insert(key.to_string(), effects);
    }
    let build_features = obj
        .get("build_features")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    Ok(EffectInventory {
        sections,
        build_features,
    })
}

/// Decode the `list_presets` payload into a list of preset names.
pub fn parse_presets(data: &serde_json::Value) -> Result<Vec<String>, String> {
    let arr = data
        .as_array()
        .ok_or_else(|| "list_presets payload is not an array".to_string())?;
    Ok(arr
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect())
}

/// On-wire form of [`EffectMetadata`] using `String`/`Vec` so it can
/// be `Deserialize`d. The daemon authors metadata as `&'static str`
/// constants; here we re-allocate into owned strings on the client
/// side and stuff them back into the static-shaped struct via
/// leaking (see the module-level comment for the lifecycle).
#[derive(Debug, serde::Deserialize)]
struct WireMetadata {
    name: String,
    help: String,
    params: Vec<WireParam>,
}

#[derive(Debug, serde::Deserialize)]
struct WireParam {
    name: String,
    kind: serde_json::Value,
    help: String,
    commit: serde_json::Value,
}

impl WireMetadata {
    /// Convert into [`EffectMetadata`] by leaking the strings/slices
    /// into `'static`. Returns `Err` when any nested param fails to
    /// decode; the error mentions the offending param's name so the
    /// caller can surface a meaningful diagnostic.
    fn into_owned(self) -> Result<EffectMetadata, String> {
        let mut params_owned = Vec::with_capacity(self.params.len());
        for param in self.params {
            params_owned.push(param.into_owned()?);
        }
        let params = params_owned.leak();
        Ok(EffectMetadata {
            name: leak_str(self.name),
            help: leak_str(self.help),
            params,
        })
    }
}

impl WireParam {
    fn into_owned(self) -> Result<fluxframe_core::ParamDescriptor, String> {
        let name = self.name;
        let kind = WireKind::deserialize(&self.kind)
            .map_err(|e| format!("param '{name}': kind decode failed: {e}"))?
            .into_owned();
        let commit = WireCommit::deserialize(&self.commit)
            .map_err(|e| format!("param '{name}': commit decode failed: {e}"))?
            .into_owned();
        Ok(fluxframe_core::ParamDescriptor {
            name: leak_str(name),
            kind,
            help: leak_str(self.help),
            commit,
        })
    }
}

/// On-wire `ParamKind`, mirroring the `#[serde(tag = "type", rename_all = "snake_case")]`
/// attribute on the daemon-side enum.
#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireKind {
    Float {
        default: f32,
        min: f32,
        max: f32,
        step: f32,
        scale: WireScale,
    },
    Integer {
        default: i64,
        min: i64,
        max: i64,
        step: i64,
        scale: WireScale,
    },
    Bool {
        default: bool,
    },
    Color {
        default: [u8; 3],
    },
    Path {
        #[serde(default)]
        default: Option<String>,
        #[serde(default)]
        extensions: Vec<String>,
        required: bool,
    },
    Enum {
        default: String,
        variants: Vec<String>,
    },
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireScale {
    Linear,
    Logarithmic,
}

impl WireScale {
    fn into_owned(self) -> fluxframe_core::Scale {
        match self {
            Self::Linear => fluxframe_core::Scale::Linear,
            Self::Logarithmic => fluxframe_core::Scale::Logarithmic,
        }
    }
}

impl WireKind {
    fn into_owned(self) -> fluxframe_core::ParamKind {
        match self {
            Self::Float {
                default,
                min,
                max,
                step,
                scale,
            } => fluxframe_core::ParamKind::Float {
                default,
                min,
                max,
                step,
                scale: scale.into_owned(),
            },
            Self::Integer {
                default,
                min,
                max,
                step,
                scale,
            } => fluxframe_core::ParamKind::Integer {
                default,
                min,
                max,
                step,
                scale: scale.into_owned(),
            },
            Self::Bool { default } => fluxframe_core::ParamKind::Bool { default },
            Self::Color { default } => fluxframe_core::ParamKind::Color { default },
            Self::Path {
                default,
                extensions,
                required,
            } => fluxframe_core::ParamKind::Path {
                default: default.map(leak_str),
                extensions: leak_vec_of_str(extensions),
                required,
            },
            Self::Enum { default, variants } => fluxframe_core::ParamKind::Enum {
                default: leak_str(default),
                variants: leak_vec_of_str(variants),
            },
        }
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum WireCommit {
    Live { debounce_ms: u32 },
    OnCommit,
    Instant,
}

impl WireCommit {
    fn into_owned(self) -> fluxframe_core::CommitStrategy {
        match self {
            Self::Live { debounce_ms } => fluxframe_core::CommitStrategy::Live { debounce_ms },
            Self::OnCommit => fluxframe_core::CommitStrategy::OnCommit,
            Self::Instant => fluxframe_core::CommitStrategy::Instant,
        }
    }
}

fn leak_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn leak_vec_of_str(v: Vec<String>) -> &'static [&'static str] {
    let leaked: Vec<&'static str> = v.into_iter().map(leak_str).collect();
    Box::leak(leaked.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_inventory_round_trips_minimal_payload() {
        let payload = serde_json::json!({
            "mask": [{
                "name": "threshold",
                "help": "Binarise",
                "params": [{
                    "name": "level",
                    "kind": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "scale": "linear"},
                    "help": "cutoff",
                    "commit": {"mode": "live", "debounce_ms": 50}
                }]
            }],
            "background": [],
            "foreground": [],
            "post": [],
            "build_features": ["ml"]
        });
        let inv = parse_inventory(&payload).expect("inventory parses");
        assert_eq!(inv.sections["mask"].len(), 1);
        assert_eq!(inv.sections["mask"][0].name, "threshold");
        assert_eq!(inv.build_features, vec!["ml".to_string()]);
    }

    #[test]
    fn parse_inventory_rejects_missing_section() {
        let payload = serde_json::json!({
            "mask": [],
            "background": [],
            "foreground": []
            // 'post' missing — protocol violation
        });
        let err = parse_inventory(&payload).expect_err("missing section is an error");
        assert!(err.contains("'post'"), "got: {err}");
    }

    #[test]
    fn parse_inventory_rejects_unknown_param_type() {
        let payload = serde_json::json!({
            "mask": [{
                "name": "weird",
                "help": "",
                "params": [{
                    "name": "x",
                    "kind": {"type": "bogus"},
                    "help": "",
                    "commit": {"mode": "instant"}
                }]
            }],
            "background": [],
            "foreground": [],
            "post": []
        });
        let err =
            parse_inventory(&payload).expect_err("unknown kind type must surface as an error");
        assert!(
            err.contains("'x'"),
            "error should mention the offending param name, got: {err}"
        );
        assert!(
            err.contains("kind decode failed"),
            "error should describe the failure, got: {err}"
        );
    }

    #[test]
    fn parse_presets_extracts_names() {
        let payload = serde_json::json!(["alpha", "beta", "gamma"]);
        let names = parse_presets(&payload).expect("parses");
        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    }
}
