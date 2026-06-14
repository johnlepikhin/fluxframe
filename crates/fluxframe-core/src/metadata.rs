//! Self-describing parameter metadata for video effects.
//!
//! Each effect publishes a `pub const METADATA: EffectMetadata` and
//! registries hand `&'static EffectMetadata` references to clients
//! (the GUI's `list_effects` command, in-process introspection,
//! future linters). Keeping the metadata `&'static` means a client
//! never has to construct an effect instance just to inspect its
//! parameter schema.
//!
//! `Serialize` is derived so the daemon can dump the inventory
//! straight onto the control socket as JSON. `Deserialize` is **not**
//! derived — these descriptors are an output of the daemon, never an
//! input.

use serde::Serialize;

/// Default debounce window for light continuous parameters (sliders
/// whose `process()` cost is per-pixel constant — threshold, level,
/// strength). 50 ms keeps slider tracking responsive while limiting
/// the daemon to ~20 commands per second per parameter.
pub const DEBOUNCE_FAST_MS: u32 = 50;

/// Debounce window for mid-cost parameters (integer params whose
/// changes cost extra passes — dilate iterations, blur passes,
/// auto-frame smoothing). 100 ms halves the rate vs `DEBOUNCE_FAST_MS`.
pub const DEBOUNCE_STANDARD_MS: u32 = 100;

/// Debounce window for heavy parameters that trigger scratch-buffer
/// reallocation (blur.downscale changes scratch size; pixelate
/// block_size is similar). 200 ms gives the worker breathing room
/// between successive reconfigurations.
pub const DEBOUNCE_HEAVY_MS: u32 = 200;

/// Top-level metadata exported by one effect (mask, plane, or post).
///
/// Lives behind a `&'static` reference; field types are `&'static
/// str` and `&'static [_]` so the whole structure can be authored as
/// a `const`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct EffectMetadata {
    /// Registry name (matches the `NAME` const on each effect type
    /// and the snake_case key used in TOML configs).
    pub name: &'static str,
    /// One-line summary suitable for a GUI tooltip / `--help` row.
    pub help: &'static str,
    /// Parameter descriptors, in display order.
    pub params: &'static [ParamDescriptor],
}

/// One parameter on an effect.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ParamDescriptor {
    /// Field name on the effect's `Config` struct (snake_case).
    pub name: &'static str,
    /// Type-shape, defaults, bounds.
    pub kind: ParamKind,
    /// One-line help text (tooltip).
    pub help: &'static str,
    /// How the GUI should batch user input before sending it on the
    /// wire.
    pub commit: CommitStrategy,
}

/// Type-shape of a single parameter.
///
/// Variants intentionally mirror what the existing effect configs
/// accept via serde — `f32`, `u32`/`i64`, `bool`, `[u8; 3]`,
/// `PathBuf`, named enums. Any new variant has to plumb through the
/// GUI's `param_row.rs` as well.
#[derive(Debug, Clone, Copy, Serialize)]
#[non_exhaustive]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParamKind {
    /// A continuous floating-point parameter.
    Float {
        /// Default value used when the field is absent in TOML.
        default: f32,
        /// Inclusive minimum.
        min: f32,
        /// Inclusive maximum.
        max: f32,
        /// Recommended GUI step size.
        step: f32,
        /// Linear vs logarithmic mapping. Use [`Scale::Logarithmic`]
        /// for parameters whose perceptual effect grows multiplicatively
        /// (e.g. blur radius from 1 to 256).
        scale: Scale,
    },
    /// A discrete integer parameter.
    Integer {
        /// Default value used when the field is absent in TOML.
        default: i64,
        /// Inclusive minimum.
        min: i64,
        /// Inclusive maximum.
        max: i64,
        /// Recommended GUI step size.
        step: i64,
        /// Linear vs logarithmic mapping.
        scale: Scale,
    },
    /// On/off toggle.
    Bool {
        /// Default value.
        default: bool,
    },
    /// An RGB triple, typically rendered as a colour picker.
    Color {
        /// Default RGB value (each channel in `[0, 255]`).
        default: [u8; 3],
    },
    /// A filesystem path. `required = true` + `default = None`
    /// indicates "the user must pick something before this effect
    /// will function" (e.g. `image_fill.path`).
    Path {
        /// Default path, if any.
        default: Option<&'static str>,
        /// File-extension hints (lower-case, no dot) for file
        /// dialogues. Empty slice = no filter.
        extensions: &'static [&'static str],
        /// Whether the parameter must be set before the effect can
        /// be used.
        required: bool,
    },
    /// A closed set of named string variants. GUI renders a
    /// drop-down.
    Enum {
        /// Default variant name. Must be a member of `variants`.
        default: &'static str,
        /// Allowed variant names, in display order.
        variants: &'static [&'static str],
    },
}

/// Linear vs logarithmic mapping for [`ParamKind::Float`] and
/// [`ParamKind::Integer`].
#[derive(Debug, Clone, Copy, Serialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum Scale {
    /// Default: GUI slider tracks value linearly.
    Linear,
    /// GUI slider tracks `log(value)` — better for parameters with
    /// wide multiplicative ranges (e.g. blur radius 1..256).
    Logarithmic,
}

/// When the GUI should send a `set` command for a given parameter.
///
/// The control socket has a bounded channel (16 deep); sending one
/// command per mouse-move would saturate it within a second of
/// dragging a slider. `CommitStrategy` lets each parameter declare
/// its own batching cadence.
#[derive(Debug, Clone, Copy, Serialize)]
#[non_exhaustive]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum CommitStrategy {
    /// Live updates, throttled to one command per `debounce_ms`
    /// window. Suitable for continuous sliders.
    Live {
        /// Debounce window in milliseconds.
        debounce_ms: u32,
    },
    /// Commit only on a discrete "done" event (file picker accept,
    /// colour dialogue close). The GUI does not send a command on
    /// every intermediate notification.
    OnCommit,
    /// Send immediately on change — for low-frequency events like a
    /// toggle switch or enum selection.
    Instant,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_param_serialises_with_all_fields() {
        let p = ParamDescriptor {
            name: "radius",
            kind: ParamKind::Float {
                default: 20.0,
                min: 1.0,
                max: 256.0,
                step: 1.0,
                scale: Scale::Logarithmic,
            },
            help: "blur radius",
            commit: CommitStrategy::Live { debounce_ms: 50 },
        };
        let s = serde_json::to_string(&p).expect("serialise");
        assert!(s.contains("\"name\":\"radius\""), "got: {s}");
        assert!(s.contains("\"type\":\"float\""), "got: {s}");
        assert!(s.contains("\"scale\":\"logarithmic\""), "got: {s}");
        assert!(s.contains("\"mode\":\"live\""), "got: {s}");
        assert!(s.contains("\"debounce_ms\":50"), "got: {s}");
    }

    #[test]
    fn integer_path_color_bool_enum_serialise() {
        for kind in [
            ParamKind::Integer {
                default: 2,
                min: 1,
                max: 16,
                step: 1,
                scale: Scale::Linear,
            },
            ParamKind::Bool { default: false },
            ParamKind::Color {
                default: [200, 100, 50],
            },
            ParamKind::Path {
                default: None,
                extensions: &["png", "jpg"],
                required: true,
            },
            ParamKind::Enum {
                default: "cover",
                variants: &["cover", "contain", "stretch"],
            },
        ] {
            let s = serde_json::to_string(&kind).expect("serialise");
            assert!(s.contains("\"type\":"), "got: {s}");
        }
    }

    #[test]
    fn effect_metadata_serialises_as_object() {
        const META: EffectMetadata = EffectMetadata {
            name: "threshold",
            help: "Binarise the mask at a threshold",
            params: &[ParamDescriptor {
                name: "level",
                kind: ParamKind::Float {
                    default: 0.5,
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                    scale: Scale::Linear,
                },
                help: "pixels at or above this become 1.0",
                commit: CommitStrategy::Live { debounce_ms: 50 },
            }],
        };
        let s = serde_json::to_string(&META).expect("serialise");
        assert!(s.contains("\"name\":\"threshold\""), "got: {s}");
        assert!(s.contains("\"params\":["), "got: {s}");
    }
}
