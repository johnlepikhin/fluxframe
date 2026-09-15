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
//! straight onto the control socket as JSON. The `&'static` shapes
//! cannot be deserialised; clients decode the same JSON into the
//! owned [`EffectSchema`] mirror instead. [`EffectSchema::from`] is an
//! exhaustive conversion, so adding a [`ParamKind`] variant fails to
//! compile until the schema learns it too, and the tests below pin
//! both serialisations to the same wire form.
//!
//! The metadata is also the canonical source of default values:
//! [`EffectMetadata::default_config`] synthesises a fresh TOML
//! table when the operator adds an effect without an explicit
//! `per_effect` block.

use serde::{Deserialize, Serialize};

use crate::error::EffectError;
use crate::traits::RawEffectParams;

/// Reserved key inside an effect's parameter table
/// (`[presets.NAME.<section>.<effect>]`) that switches the effect on or
/// off without removing it from the chain. Never part of an effect's
/// own parameters: [`split_effect_table`] strips it before `configure`.
pub const EFFECT_ENABLED_KEY: &str = "enabled";

/// An effect's parameter table split into the enable flag and the
/// parameters handed to `configure`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedEffect {
    /// Whether the effect runs (`enabled` key; `true` when absent).
    pub enabled: bool,
    /// Parameters for the effect's `configure`, without `enabled`.
    pub params: RawEffectParams,
}

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum Scale {
    /// Default: GUI slider tracks value linearly.
    Linear,
    /// GUI slider tracks `log(value)` — better for parameters with
    /// wide multiplicative ranges (e.g. blur radius 1..256).
    Logarithmic,
}

impl EffectMetadata {
    /// Build a fresh-add default config TOML table from each param's
    /// declared default.
    ///
    /// Mirrors what `#[serde(default = "...")]` on the effect's
    /// `Config` struct would produce if every field had one, but
    /// using the metadata block as the canonical source. The daemon
    /// calls this when a chain entry is added without an explicit
    /// `per_effect` block (e.g. via the GUI's add-effect menu).
    ///
    /// Returns `toml::Table` rather than `serde_json::Value` because
    /// the operator-facing canonical form is `fluxframe.toml`; converting
    /// to JSON happens once at the IPC boundary.
    ///
    /// # Errors
    ///
    /// Returns the names of [`ParamKind::Path`] fields marked
    /// `required: true` with no `default` — those have no sensible
    /// default and the caller must reject the add until the operator
    /// supplies them (typically `image_fill.path`; composite is
    /// configured at pipeline level, not via the effect registry).
    ///
    /// Optional `Path` params without a default are silently skipped;
    /// the effect's `Config` struct must therefore accept an empty
    /// TOML table for those fields (typically via `#[serde(default)]`).
    /// The `Err` `Vec` is guaranteed non-empty by construction.
    pub fn default_config(&self) -> Result<toml::Table, Vec<String>> {
        let mut table = toml::Table::new();
        let mut missing: Vec<String> = Vec::new();
        for param in self.params {
            let value = match param.kind {
                ParamKind::Float { default, .. } => toml::Value::Float(f64::from(default)),
                ParamKind::Integer { default, .. } => toml::Value::Integer(default),
                ParamKind::Bool { default } => toml::Value::Boolean(default),
                ParamKind::Color { default } => toml::Value::Array(
                    default
                        .iter()
                        .map(|&c| toml::Value::Integer(i64::from(c)))
                        .collect(),
                ),
                ParamKind::Enum { default, .. } => toml::Value::String(default.to_string()),
                ParamKind::Path {
                    default, required, ..
                } => match (default, required) {
                    (Some(d), _) => toml::Value::String(d.to_string()),
                    (None, true) => {
                        missing.push(param.name.to_string());
                        continue;
                    }
                    (None, false) => continue,
                },
            };
            table.insert(param.name.to_string(), value);
        }
        if missing.is_empty() {
            Ok(table)
        } else {
            Err(missing)
        }
    }
}

/// Split the effect table `table` (`None` when the preset has none) of
/// effect `name` into its [`EFFECT_ENABLED_KEY`] flag and the remaining
/// parameters.
///
/// # Errors
///
/// [`EffectError::InvalidConfig`] when `table` is not a TOML table or
/// `enabled` is not a boolean.
pub fn split_effect_table(
    name: &str,
    table: Option<&toml::Value>,
) -> Result<(bool, toml::Table), EffectError> {
    let mut params = match table {
        None => toml::Table::new(),
        Some(toml::Value::Table(t)) => t.clone(),
        Some(other) => {
            return Err(EffectError::InvalidConfig {
                name: name.to_string(),
                reason: format!("parameters must be a table, got {}", other.type_str()),
                hint: Some(format!("write the parameters under [<section>.{name}]")),
            });
        }
    };
    let enabled = match params.remove(EFFECT_ENABLED_KEY) {
        None => true,
        Some(toml::Value::Boolean(b)) => b,
        Some(other) => {
            return Err(EffectError::InvalidConfig {
                name: name.to_string(),
                reason: format!(
                    "`{EFFECT_ENABLED_KEY}` must be a boolean, got {}",
                    other.type_str()
                ),
                hint: Some(format!("e.g. {EFFECT_ENABLED_KEY} = false")),
            });
        }
    };
    Ok((enabled, params))
}

/// Resolve what effect `name` is configured with: its enable flag plus
/// the parameters for `configure`.
///
/// An effect with no parameters of its own (no table, or a table that
/// only carries `enabled`) gets the defaults declared in `metadata` —
/// the same answer whether the chain came from `fluxframe.toml` or was
/// edited live. An explicit table is passed through unchanged, so the
/// effect's serde defaults fill any field it omits.
///
/// # Errors
///
/// [`EffectError::InvalidConfig`] when [`split_effect_table`] rejects
/// the table, when defaults are needed but `metadata` is `None`, or
/// when the metadata declares required fields without a default.
pub fn resolve_effect_params(
    name: &str,
    table: Option<&toml::Value>,
    metadata: Option<&EffectMetadata>,
) -> Result<ResolvedEffect, EffectError> {
    let (enabled, params) = split_effect_table(name, table)?;
    if !params.is_empty() {
        return Ok(ResolvedEffect {
            enabled,
            params: toml::Value::Table(params),
        });
    }
    let Some(metadata) = metadata else {
        return Err(EffectError::InvalidConfig {
            name: name.to_string(),
            reason: "effect is registered without metadata; cannot synthesise defaults".into(),
            hint: Some(
                "this is a programming error — every effect must declare \
                 `pub const METADATA: EffectMetadata`. Please file a bug."
                    .into(),
            ),
        });
    };
    match metadata.default_config() {
        Ok(defaults) => Ok(ResolvedEffect {
            enabled,
            params: toml::Value::Table(defaults),
        }),
        Err(missing) => {
            let fields = missing
                .iter()
                .map(|f| format!("`{f}`"))
                .collect::<Vec<_>>()
                .join(", ");
            Err(EffectError::InvalidConfig {
                name: name.to_string(),
                reason: format!(
                    "missing required field(s) {fields}; set them via the parameter editor \
                     (or in fluxframe.toml) before adding this effect to a chain"
                ),
                hint: Some(format!(
                    "set the following field(s) on `{name}` explicitly, \
                     e.g. via the parameter editor: {fields}"
                )),
            })
        }
    }
}

/// When the GUI should send a `set` command for a given parameter.
///
/// The control socket has a bounded channel (16 deep); sending one
/// command per mouse-move would saturate it within a second of
/// dragging a slider. `CommitStrategy` lets each parameter declare
/// its own batching cadence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

/// Owned, deserialisable form of [`EffectMetadata`] — what a client
/// decodes from the `list_effects` payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectSchema {
    /// See [`EffectMetadata::name`].
    pub name: String,
    /// See [`EffectMetadata::help`].
    pub help: String,
    /// See [`EffectMetadata::params`].
    pub params: Vec<ParamSchema>,
}

/// Owned form of [`ParamDescriptor`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamSchema {
    /// See [`ParamDescriptor::name`].
    pub name: String,
    /// See [`ParamDescriptor::kind`].
    pub kind: ParamKindSchema,
    /// See [`ParamDescriptor::help`].
    pub help: String,
    /// See [`ParamDescriptor::commit`].
    pub commit: CommitStrategy,
}

/// Owned form of [`ParamKind`]; variant and field docs live there.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(missing_docs, reason = "fields mirror ParamKind one-to-one")]
pub enum ParamKindSchema {
    Float {
        default: f32,
        min: f32,
        max: f32,
        step: f32,
        scale: Scale,
    },
    Integer {
        default: i64,
        min: i64,
        max: i64,
        step: i64,
        scale: Scale,
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

impl From<&EffectMetadata> for EffectSchema {
    fn from(meta: &EffectMetadata) -> Self {
        Self {
            name: meta.name.to_owned(),
            help: meta.help.to_owned(),
            params: meta.params.iter().map(ParamSchema::from).collect(),
        }
    }
}

impl From<&ParamDescriptor> for ParamSchema {
    fn from(param: &ParamDescriptor) -> Self {
        Self {
            name: param.name.to_owned(),
            kind: param.kind.into(),
            help: param.help.to_owned(),
            commit: param.commit,
        }
    }
}

impl From<ParamKind> for ParamKindSchema {
    fn from(kind: ParamKind) -> Self {
        // Exhaustive on purpose (in-crate `#[non_exhaustive]` does not
        // apply): a new `ParamKind` variant must be mirrored here.
        match kind {
            ParamKind::Float {
                default,
                min,
                max,
                step,
                scale,
            } => Self::Float {
                default,
                min,
                max,
                step,
                scale,
            },
            ParamKind::Integer {
                default,
                min,
                max,
                step,
                scale,
            } => Self::Integer {
                default,
                min,
                max,
                step,
                scale,
            },
            ParamKind::Bool { default } => Self::Bool { default },
            ParamKind::Color { default } => Self::Color { default },
            ParamKind::Path {
                default,
                extensions,
                required,
            } => Self::Path {
                default: default.map(str::to_owned),
                extensions: extensions.iter().map(|e| (*e).to_owned()).collect(),
                required,
            },
            ParamKind::Enum { default, variants } => Self::Enum {
                default: default.to_owned(),
                variants: variants.iter().map(|v| (*v).to_owned()).collect(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metadata with one defaulted field (`rgb`) — shaped like
    /// `color_fill`, whose config has no serde default for `rgb`.
    const FILL_META: EffectMetadata = EffectMetadata {
        name: "fill",
        help: "",
        params: &[ParamDescriptor {
            name: "rgb",
            kind: ParamKind::Color { default: [1, 2, 3] },
            help: "",
            commit: CommitStrategy::OnCommit,
        }],
    };

    fn table(text: &str) -> toml::Value {
        toml::Value::Table(toml::from_str(text).expect("valid TOML"))
    }

    fn reason(err: &EffectError) -> &str {
        match err {
            EffectError::InvalidConfig { reason, .. } => reason,
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn split_strips_enabled_and_defaults_to_true() {
        let (enabled, params) = split_effect_table("fill", Some(&table("rgb = [4, 5, 6]")))
            .expect("plain table splits");
        assert!(enabled);
        assert!(params.contains_key("rgb"));

        let (enabled, params) =
            split_effect_table("fill", Some(&table("enabled = false\nrgb = [4, 5, 6]")))
                .expect("disabled table splits");
        assert!(!enabled);
        assert!(!params.contains_key(EFFECT_ENABLED_KEY));

        let (enabled, params) = split_effect_table("fill", None).expect("absent table splits");
        assert!(enabled);
        assert!(params.is_empty());
    }

    #[test]
    fn split_rejects_non_bool_enabled_and_non_table() {
        let err = split_effect_table("fill", Some(&table(r#"enabled = "no""#)))
            .expect_err("string enabled");
        assert!(reason(&err).contains("must be a boolean"), "{err}");

        let err = split_effect_table("fill", Some(&toml::Value::Integer(3)))
            .expect_err("non-table params");
        assert!(reason(&err).contains("must be a table"), "{err}");
    }

    #[test]
    fn resolve_passes_explicit_params_through() {
        let resolved = resolve_effect_params("fill", Some(&table("rgb = [4, 5, 6]")), None)
            .expect("explicit params need no metadata");
        assert!(resolved.enabled);
        assert_eq!(resolved.params, table("rgb = [4, 5, 6]"));
    }

    /// Regression: a table holding only `enabled = false` used to count
    /// as explicit configuration, so `configure` got no `rgb` and the
    /// next `set_chain` failed for `color_fill`.
    #[test]
    fn resolve_synthesises_defaults_for_flag_only_table() {
        let resolved =
            resolve_effect_params("fill", Some(&table("enabled = false")), Some(&FILL_META))
                .expect("defaults synthesised");
        assert!(!resolved.enabled);
        assert_eq!(resolved.params, table("rgb = [1, 2, 3]"));

        let resolved = resolve_effect_params("fill", None, Some(&FILL_META)).expect("absent table");
        assert!(resolved.enabled);
        assert_eq!(resolved.params, table("rgb = [1, 2, 3]"));
    }

    #[test]
    fn resolve_rejects_missing_required_field_and_missing_metadata() {
        const PATH_META: EffectMetadata = EffectMetadata {
            name: "image",
            help: "",
            params: &[ParamDescriptor {
                name: "path",
                kind: ParamKind::Path {
                    default: None,
                    extensions: &[],
                    required: true,
                },
                help: "",
                commit: CommitStrategy::OnCommit,
            }],
        };
        let err = resolve_effect_params("image", None, Some(&PATH_META))
            .expect_err("required path without default");
        assert!(reason(&err).contains("`path`"), "{err}");

        let err = resolve_effect_params("image", None, None).expect_err("no metadata");
        assert!(reason(&err).contains("without metadata"), "{err}");
    }

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
    fn default_config_emits_one_entry_per_param_with_default() {
        const META: EffectMetadata = EffectMetadata {
            name: "demo",
            help: "",
            params: &[
                ParamDescriptor {
                    name: "radius",
                    kind: ParamKind::Float {
                        default: 20.0,
                        min: 1.0,
                        max: 256.0,
                        step: 1.0,
                        scale: Scale::Linear,
                    },
                    help: "",
                    commit: CommitStrategy::Live { debounce_ms: 50 },
                },
                ParamDescriptor {
                    name: "passes",
                    kind: ParamKind::Integer {
                        default: 2,
                        min: 1,
                        max: 16,
                        step: 1,
                        scale: Scale::Linear,
                    },
                    help: "",
                    commit: CommitStrategy::Live { debounce_ms: 50 },
                },
                ParamDescriptor {
                    name: "rgb",
                    kind: ParamKind::Color {
                        default: [128, 128, 128],
                    },
                    help: "",
                    commit: CommitStrategy::OnCommit,
                },
                ParamDescriptor {
                    name: "mode",
                    kind: ParamKind::Enum {
                        default: "cover",
                        variants: &["cover", "contain"],
                    },
                    help: "",
                    commit: CommitStrategy::Instant,
                },
            ],
        };
        let table = META
            .default_config()
            .expect("no required-without-default fields");
        assert_eq!(table.get("radius"), Some(&toml::Value::Float(20.0)));
        assert_eq!(table.get("passes"), Some(&toml::Value::Integer(2)));
        assert_eq!(
            table.get("rgb"),
            Some(&toml::Value::Array(vec![
                toml::Value::Integer(128),
                toml::Value::Integer(128),
                toml::Value::Integer(128),
            ]))
        );
        assert_eq!(
            table.get("mode"),
            Some(&toml::Value::String("cover".into()))
        );
    }

    #[test]
    fn default_config_returns_missing_for_required_path_without_default() {
        const META: EffectMetadata = EffectMetadata {
            name: "image_fill",
            help: "",
            params: &[ParamDescriptor {
                name: "path",
                kind: ParamKind::Path {
                    default: None,
                    extensions: &["png", "jpg"],
                    required: true,
                },
                help: "",
                commit: CommitStrategy::OnCommit,
            }],
        };
        let err = META.default_config().expect_err("required path => Err");
        assert_eq!(err, vec!["path".to_string()]);
    }

    #[test]
    fn default_config_emits_path_with_default() {
        const META: EffectMetadata = EffectMetadata {
            name: "demo",
            help: "",
            params: &[ParamDescriptor {
                name: "model",
                kind: ParamKind::Path {
                    default: Some("/var/lib/x.onnx"),
                    extensions: &["onnx"],
                    required: true,
                },
                help: "",
                commit: CommitStrategy::OnCommit,
            }],
        };
        let table = META.default_config().expect("Some(default) => Ok");
        assert_eq!(
            table.get("model"),
            Some(&toml::Value::String("/var/lib/x.onnx".into()))
        );
    }

    #[test]
    fn default_config_skips_optional_path_without_default() {
        const META: EffectMetadata = EffectMetadata {
            name: "demo",
            help: "",
            params: &[ParamDescriptor {
                name: "logo",
                kind: ParamKind::Path {
                    default: None,
                    extensions: &[],
                    required: false,
                },
                help: "",
                commit: CommitStrategy::OnCommit,
            }],
        };
        let table = META.default_config().expect("optional => Ok");
        assert!(
            table.is_empty(),
            "optional path with no default skips emission"
        );
    }

    /// Every `ParamKind` / `CommitStrategy` / `Scale` variant: the
    /// static metadata and its owned schema must share one wire form,
    /// and that form must decode back into the schema.
    #[test]
    fn effect_schema_matches_metadata_wire_form_for_every_variant() {
        const META: EffectMetadata = EffectMetadata {
            name: "all_kinds",
            help: "every variant",
            params: &[
                ParamDescriptor {
                    name: "f",
                    kind: ParamKind::Float {
                        default: 0.5,
                        min: 0.0,
                        max: 1.0,
                        step: 0.25,
                        scale: Scale::Logarithmic,
                    },
                    help: "float",
                    commit: CommitStrategy::Live { debounce_ms: 50 },
                },
                ParamDescriptor {
                    name: "i",
                    kind: ParamKind::Integer {
                        default: 2,
                        min: 1,
                        max: 16,
                        step: 1,
                        scale: Scale::Linear,
                    },
                    help: "int",
                    commit: CommitStrategy::OnCommit,
                },
                ParamDescriptor {
                    name: "b",
                    kind: ParamKind::Bool { default: true },
                    help: "",
                    commit: CommitStrategy::Instant,
                },
                ParamDescriptor {
                    name: "c",
                    kind: ParamKind::Color { default: [1, 2, 3] },
                    help: "",
                    commit: CommitStrategy::OnCommit,
                },
                ParamDescriptor {
                    name: "p",
                    kind: ParamKind::Path {
                        default: Some("/x.png"),
                        extensions: &["png"],
                        required: false,
                    },
                    help: "",
                    commit: CommitStrategy::OnCommit,
                },
                ParamDescriptor {
                    name: "p_none",
                    kind: ParamKind::Path {
                        default: None,
                        extensions: &[],
                        required: true,
                    },
                    help: "",
                    commit: CommitStrategy::OnCommit,
                },
                ParamDescriptor {
                    name: "e",
                    kind: ParamKind::Enum {
                        default: "a",
                        variants: &["a", "b"],
                    },
                    help: "",
                    commit: CommitStrategy::Instant,
                },
            ],
        };
        let schema = EffectSchema::from(&META);
        let meta_json = serde_json::to_value(META).expect("serialise metadata");
        let schema_json = serde_json::to_value(&schema).expect("serialise schema");
        assert_eq!(meta_json, schema_json);
        let decoded: EffectSchema = serde_json::from_value(meta_json).expect("decode");
        assert_eq!(decoded, schema);
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
