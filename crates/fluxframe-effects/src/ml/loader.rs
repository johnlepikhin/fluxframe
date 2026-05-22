//! Helpers for loading a `ModelConfig` sidecar file or producing a
//! conservative placeholder.

use std::path::Path;

use tracing::warn;

use super::ModelConfig;
use fluxframe_core::error::InferenceError;

/// Load `<model>.toml` next to `model_path` if it exists, otherwise
/// produce a minimal placeholder config and emit a `warn!`.
///
/// The placeholder is intentionally invalid for any real inference
/// effect (1x1 input) so callers are forced to ship a proper sidecar
/// before Stage 4 effects accept the model.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidModelConfig`] when the sidecar
/// exists but cannot be parsed or fails validation.
pub fn load_sidecar_or_placeholder(model_path: &Path) -> Result<ModelConfig, InferenceError> {
    let sidecar = model_path.with_extension("toml");
    if sidecar.exists() {
        ModelConfig::load(&sidecar)
    } else {
        warn!(
            sidecar = %sidecar.display(),
            "no model config sidecar found; using a 1x1 placeholder — Stage 4 effects will refuse this"
        );
        ModelConfig::from_toml_str(
            r#"
name = "<unknown>"
input_width = 1
input_height = 1
"#,
        )
    }
}
