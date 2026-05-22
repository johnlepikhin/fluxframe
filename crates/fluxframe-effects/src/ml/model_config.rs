//! Model configuration loaded from a TOML sidecar file.
//!
//! Mirrors the schema from §24.2 of the spec.  The config is consumed by
//! [`crate::ml::onnx::OnnxEngine`] (Stage 3, Wave 2) and by effects such
//! as `background_blur` (Stage 4) which need to know the model's tensor
//! IO layout to pre- and post-process frames correctly.

use std::path::Path;

use fluxframe_core::error::InferenceError;
use fluxframe_core::frame::PixelFormat;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// PixelFormat serde adapter (local — keeps `frame::PixelFormat` serde-free,
// mirrors the pattern used in `fluxframe_core::config::pixel_format_serde`).
// ---------------------------------------------------------------------------

/// Serde adapter for [`PixelFormat`] using the uppercase tag strings
/// documented in §13 (`"RGB"`, `"RGBA"`, `"BGR"`, `"YUY2"`, `"NV12"`,
/// `"GRAY8"`).
mod pixel_format_serde {
    use super::PixelFormat;
    use serde::{Deserialize, Deserializer, Serializer};

    #[allow(
        clippy::trivially_copy_pass_by_ref,
        reason = "serde calls expect &T signature"
    )]
    pub(super) fn serialize<S: Serializer>(value: &PixelFormat, ser: S) -> Result<S::Ok, S::Error> {
        let tag = match value {
            PixelFormat::Rgb => "RGB",
            PixelFormat::Rgba => "RGBA",
            PixelFormat::Bgr => "BGR",
            PixelFormat::Yuy2 => "YUY2",
            PixelFormat::Nv12 => "NV12",
            PixelFormat::Gray8 => "GRAY8",
        };
        ser.serialize_str(tag)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<PixelFormat, D::Error> {
        let raw = String::deserialize(de)?;
        match raw.to_ascii_uppercase().as_str() {
            "RGB" => Ok(PixelFormat::Rgb),
            "RGBA" => Ok(PixelFormat::Rgba),
            "BGR" => Ok(PixelFormat::Bgr),
            "YUY2" => Ok(PixelFormat::Yuy2),
            "NV12" => Ok(PixelFormat::Nv12),
            "GRAY8" => Ok(PixelFormat::Gray8),
            other => Err(serde::de::Error::custom(format!(
                "unknown pixel format `{other}`; expected one of RGB, RGBA, BGR, YUY2, NV12, GRAY8"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor layout / dtype enums
// ---------------------------------------------------------------------------

/// Tensor layout for inference inputs.
///
/// # Invariants
///
/// `InputLayout` does not include the `Hw` variant — input tensors
/// always carry batch and channel dimensions.  Use [`OutputLayout::Hw`]
/// for pre-squeezed segmentation masks.  Splitting `InputLayout` and
/// [`OutputLayout`] makes the "no flat 2D input" rule a compile-time
/// guarantee instead of a runtime check in [`ModelConfig::validate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum InputLayout {
    /// `[batch, height, width, channels]`.
    #[serde(rename = "NHWC")]
    Nhwc,
    /// `[batch, channels, height, width]`.
    #[serde(rename = "NCHW")]
    Nchw,
}

/// Tensor layout for inference outputs.
///
/// # Invariants
///
/// `OutputLayout::Hw` is valid for pre-squeezed segmentation masks; the
/// other variants mirror [`InputLayout`].  See the doc on [`InputLayout`]
/// for why these are distinct types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum OutputLayout {
    /// `[height, width]` — flat 2D, only for outputs.
    #[serde(rename = "HW")]
    Hw,
    /// `[batch, height, width, channels]`.
    #[serde(rename = "NHWC")]
    Nhwc,
    /// `[batch, channels, height, width]`.
    #[serde(rename = "NCHW")]
    Nchw,
}

/// Tensor element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum TensorDType {
    /// 32-bit IEEE 754 float.
    F32,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
}

/// Semantic interpretation of the output tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OutputType {
    /// Single-channel confidence mask (0..1).
    Mask,
    /// Per-class probabilities (softmaxed).
    Probabilities,
    /// Per-class logits (pre-softmax).
    Logits,
    /// Integer category mask; combine with `person_class_index`.
    CategoryMask,
}

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------

/// TOML-deserialised model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ModelConfig {
    /// Human-readable model identifier.
    pub name: String,

    /// Expected input tensor width in pixels.
    pub input_width: u32,
    /// Expected input tensor height in pixels.
    pub input_height: u32,
    /// Memory layout the model expects for its input tensor.
    #[serde(default = "default_input_layout")]
    pub input_layout: InputLayout,
    /// Element type.  Stage 3 normalises to `f32` on the host side;
    /// future stages may add `u8`/`i8` quantised paths.
    #[serde(default = "default_input_dtype")]
    pub input_dtype: TensorDType,
    /// Pixel format expected at input (before per-channel scaling).
    #[serde(
        default = "default_input_color",
        serialize_with = "pixel_format_serde::serialize",
        deserialize_with = "pixel_format_serde::deserialize"
    )]
    pub input_color: PixelFormat,
    /// Multiplicative factor applied to each pixel value before
    /// inference (e.g. `1.0 / 255.0` for `[0, 1]` normalisation).
    #[serde(default = "default_input_scale")]
    pub input_scale: f32,
    /// Subtracted from each pixel value before scaling.
    #[serde(default)]
    pub input_zero_point: f32,

    /// Which output the effect consumes (0-based).
    #[serde(default)]
    pub output_index: usize,
    /// Layout of the selected output.
    #[serde(default = "default_output_layout")]
    pub output_layout: OutputLayout,
    /// Semantic kind of the output tensor.
    #[serde(default = "default_output_type")]
    pub output_type: OutputType,

    /// For [`OutputType::CategoryMask`] — class id treated as
    /// foreground (e.g. "person").  Ignored for other output types.
    #[serde(default)]
    pub person_class_index: Option<u32>,
    /// Threshold applied to the mask/probability output before
    /// post-processing.  None ⇒ effect-default.
    #[serde(default)]
    pub threshold: Option<f32>,
}

fn default_input_layout() -> InputLayout {
    InputLayout::Nhwc
}
fn default_input_dtype() -> TensorDType {
    TensorDType::F32
}
fn default_input_color() -> PixelFormat {
    PixelFormat::Rgb
}
fn default_input_scale() -> f32 {
    1.0 / 255.0
}
fn default_output_layout() -> OutputLayout {
    OutputLayout::Hw
}
fn default_output_type() -> OutputType {
    OutputType::Mask
}

impl ModelConfig {
    /// Construct a minimal [`ModelConfig`] with defaults for every
    /// optional field.
    ///
    /// Useful in tests and as a building block for synthetic
    /// placeholders.  Callers that need a non-default `input_scale`,
    /// `output_type` etc. should mutate the returned value before
    /// calling [`ModelConfig::validate`].
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        input_width: u32,
        input_height: u32,
    ) -> Self {
        Self {
            name: name.into(),
            input_width,
            input_height,
            input_layout: default_input_layout(),
            input_dtype: default_input_dtype(),
            input_color: default_input_color(),
            input_scale: default_input_scale(),
            input_zero_point: 0.0,
            output_index: 0,
            output_layout: default_output_layout(),
            output_type: default_output_type(),
            person_class_index: None,
            threshold: None,
        }
    }

    /// Parse a TOML document into a [`ModelConfig`] and validate.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidModelConfig`] on parse or
    /// validation failure.
    pub fn from_toml_str(text: &str) -> Result<Self, InferenceError> {
        let cfg: Self = toml::from_str(text).map_err(|e| InferenceError::InvalidModelConfig {
            reason: e.to_string(),
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Read and parse a model config from the given path.
    ///
    /// # Errors
    ///
    /// * [`InferenceError::ModelNotFound`] — the file does not exist.
    ///   The variant name covers both ONNX models and their sidecar
    ///   configs: "model OR config file not found".
    /// * [`InferenceError::InvalidModelConfig`] — the file exists but
    ///   cannot be read, parsed or fails validation.
    pub fn load(path: &Path) -> Result<Self, InferenceError> {
        let text = std::fs::read_to_string(path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => InferenceError::ModelNotFound {
                path: path.display().to_string(),
            },
            _ => InferenceError::InvalidModelConfig {
                reason: format!("cannot read model config {}: {e}", path.display()),
            },
        })?;
        Self::from_toml_str(&text)
    }

    /// Structural validation beyond what serde already enforces.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidModelConfig`] when invariants
    /// are violated.
    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.input_width == 0 || self.input_height == 0 {
            return Err(InferenceError::InvalidModelConfig {
                reason: "input_width and input_height must be > 0".into(),
            });
        }
        // Note: `input_layout = HW` is intentionally impossible by types
        // — see [`InputLayout`]'s invariants doc.
        if self.input_scale == 0.0 || !self.input_scale.is_finite() {
            return Err(InferenceError::InvalidModelConfig {
                reason: format!(
                    "input_scale must be a positive finite number, got {}",
                    self.input_scale
                ),
            });
        }
        if matches!(self.output_type, OutputType::CategoryMask) && self.person_class_index.is_none()
        {
            return Err(InferenceError::InvalidModelConfig {
                reason: "person_class_index is required when output_type = \"category_mask\""
                    .into(),
            });
        }
        if let Some(t) = self.threshold {
            if !(0.0..=1.0).contains(&t) {
                return Err(InferenceError::InvalidModelConfig {
                    reason: format!("threshold must be in [0,1], got {t}"),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_config() {
        let cfg = ModelConfig::from_toml_str(
            r#"
name = "person-seg"
input_width = 256
input_height = 144
input_layout = "NHWC"
input_dtype = "f32"
input_color = "RGB"
input_scale = 0.003921568627
input_zero_point = 0.0
output_index = 0
output_layout = "HW"
output_type = "mask"
threshold = 0.5
"#,
        )
        .expect("parse ok");
        assert_eq!(cfg.name, "person-seg");
        assert_eq!(cfg.input_width, 256);
        assert_eq!(cfg.input_layout, InputLayout::Nhwc);
        assert_eq!(cfg.output_type, OutputType::Mask);
        assert_eq!(cfg.threshold, Some(0.5));
    }

    #[test]
    fn applies_defaults_when_omitted() {
        let cfg = ModelConfig::from_toml_str(
            r#"
name = "minimal"
input_width = 64
input_height = 48
"#,
        )
        .expect("defaults ok");
        assert_eq!(cfg.input_layout, InputLayout::Nhwc);
        assert_eq!(cfg.input_dtype, TensorDType::F32);
        assert_eq!(cfg.input_color, PixelFormat::Rgb);
        assert!((cfg.input_scale - 1.0 / 255.0).abs() < 1e-6);
        assert!(cfg.input_zero_point.abs() < f32::EPSILON);
        assert_eq!(cfg.output_layout, OutputLayout::Hw);
        assert_eq!(cfg.output_type, OutputType::Mask);
        assert!(cfg.threshold.is_none());
    }

    #[test]
    fn rejects_zero_dimensions() {
        let err = ModelConfig::from_toml_str(
            r#"
name = "bad"
input_width = 0
input_height = 1
"#,
        )
        .expect_err("zero width must fail");
        let msg = format!("{err}");
        assert!(msg.contains("input_width"), "got: {msg}");
    }

    #[test]
    fn rejects_nonfinite_scale() {
        let mut cfg = ModelConfig::new("x", 1, 1);
        cfg.input_scale = f32::NAN;
        assert!(cfg.validate().is_err());
        cfg.input_scale = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_hw_for_input_layout_in_toml() {
        // `HW` is not a valid serde tag for `InputLayout` — the parse
        // must fail at deserialisation time, before `validate` runs.
        let err = ModelConfig::from_toml_str(
            r#"
name = "x"
input_width = 1
input_height = 1
input_layout = "HW"
"#,
        )
        .expect_err("HW input layout must fail to parse");
        let msg = format!("{err}");
        assert!(msg.to_ascii_lowercase().contains("hw") || msg.contains("input_layout"), "got: {msg}");
    }

    #[test]
    fn rejects_unknown_fields() {
        let err = ModelConfig::from_toml_str(
            r#"
name = "x"
input_width = 1
input_height = 1
mystery = "field"
"#,
        )
        .expect_err("unknown field must fail");
        let msg = format!("{err}");
        assert!(msg.contains("mystery"), "got: {msg}");
    }

    #[test]
    fn rejects_category_mask_without_class_index() {
        let err = ModelConfig::from_toml_str(
            r#"
name = "x"
input_width = 1
input_height = 1
output_type = "category_mask"
"#,
        )
        .expect_err("missing person_class_index must fail");
        let msg = format!("{err}");
        assert!(msg.contains("person_class_index"), "got: {msg}");
    }

    #[test]
    fn rejects_out_of_range_threshold() {
        let err = ModelConfig::from_toml_str(
            r#"
name = "x"
input_width = 1
input_height = 1
threshold = 1.5
"#,
        )
        .expect_err("threshold > 1 must fail");
        let msg = format!("{err}");
        assert!(msg.contains("threshold"), "got: {msg}");
    }

    #[test]
    fn accepts_threshold_in_range() {
        let cfg = ModelConfig::from_toml_str(
            r#"
name = "x"
input_width = 1
input_height = 1
threshold = 0.0
"#,
        )
        .expect("0.0 ok");
        assert_eq!(cfg.threshold, Some(0.0));
    }
}
