//! On-wire decoders for the daemon's `list_effects` / `list_presets`
//! handshake payloads.
//!
//! Effect metadata decodes straight into the owned
//! [`fluxframe_core::EffectSchema`] mirror that core keeps in lockstep
//! with the daemon-side `EffectMetadata`.

use fluxframe_core::{EffectSchema, SubchainKind};
use serde::Deserialize as _;

use crate::state::EffectInventory;

/// Decode the `list_effects` payload into [`EffectInventory`].
///
/// Walks one array per [`SubchainKind`] plus `build_features`.
/// Returns a descriptive error when any field is missing or
/// mistyped — those are protocol violations that the GUI cannot
/// recover from.
pub fn parse_inventory(data: &serde_json::Value) -> Result<EffectInventory, String> {
    let obj = data
        .as_object()
        .ok_or_else(|| "list_effects payload is not an object".to_string())?;
    let mut sections = std::collections::BTreeMap::new();
    for section in SubchainKind::ALL {
        let rows = obj
            .get(section.as_str())
            .ok_or_else(|| format!("list_effects missing section '{section}'"))?;
        let effects = Vec::<EffectSchema>::deserialize(rows)
            .map_err(|e| format!("section '{section}': metadata decode failed: {e}"))?;
        sections.insert(section, effects);
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
        assert_eq!(inv.sections[&SubchainKind::Mask].len(), 1);
        assert_eq!(inv.sections[&SubchainKind::Mask][0].name, "threshold");
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
            err.contains("section 'mask'") && err.contains("metadata decode failed"),
            "error should name the section and describe the failure, got: {err}"
        );
    }

    #[test]
    fn parse_presets_extracts_names() {
        let payload = serde_json::json!(["alpha", "beta", "gamma"]);
        let names = parse_presets(&payload).expect("parses");
        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    }
}
