//! Thin compatibility shim. The wire-format types (`Command`,
//! `Response`, `SetPath`, `parse_set_path`) now live in
//! [`fluxframe_core::protocol`] so that the GUI client crate can
//! depend on them without pulling the daemon binary. This module
//! re-exports the shared types and keeps the daemon-only
//! `json_to_toml` helper here.

#[allow(
    unused_imports,
    reason = "wire-format type re-exported as part of the control surface"
)]
pub use fluxframe_core::protocol::{Command, Response, SetPath, parse_set_path};

/// Convert a `serde_json::Value` into a `toml::Value`. Used to turn
/// the wire-format payload (JSON) into the configuration vocabulary
/// (TOML) the effect's `configure()` expects.
///
/// Lives in the cli crate because it is an apply-time helper for the
/// daemon — the GUI never round-trips JSON through TOML.
///
/// # Errors
///
/// Returns a reason string for unrepresentable inputs (`null`,
/// out-of-range numbers).
pub fn json_to_toml(v: serde_json::Value) -> Result<toml::Value, String> {
    match v {
        serde_json::Value::Null => Err("`null` is not representable in TOML".into()),
        serde_json::Value::Bool(b) => Ok(toml::Value::Boolean(b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(toml::Value::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Ok(toml::Value::Float(f))
            } else {
                Err(format!("number out of range: {n}"))
            }
        }
        serde_json::Value::String(s) => Ok(toml::Value::String(s)),
        serde_json::Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                out.push(json_to_toml(item)?);
            }
            Ok(toml::Value::Array(out))
        }
        serde_json::Value::Object(obj) => {
            let mut table = toml::Table::new();
            for (k, v) in obj {
                table.insert(k, json_to_toml(v)?);
            }
            Ok(toml::Value::Table(table))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_to_toml_scalars_round_trip() {
        let toml_int = json_to_toml(serde_json::json!(42)).unwrap();
        assert_eq!(toml_int, toml::Value::Integer(42));
        let toml_float = json_to_toml(serde_json::json!(0.5)).unwrap();
        assert!(matches!(toml_float, toml::Value::Float(f) if (f - 0.5).abs() < 1e-9));
        let toml_str = json_to_toml(serde_json::json!("hello")).unwrap();
        assert_eq!(toml_str, toml::Value::String("hello".into()));
        let toml_bool = json_to_toml(serde_json::json!(true)).unwrap();
        assert_eq!(toml_bool, toml::Value::Boolean(true));
    }

    #[test]
    fn json_to_toml_rejects_null() {
        assert!(json_to_toml(serde_json::Value::Null).is_err());
    }

    #[test]
    fn json_to_toml_array_of_ints() {
        let v = json_to_toml(serde_json::json!([1, 2, 3])).unwrap();
        match v {
            toml::Value::Array(a) => {
                assert_eq!(a.len(), 3);
                assert_eq!(a[0], toml::Value::Integer(1));
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }
}
