//! Generic name-keyed registry shared by the mask, plane, and post
//! effect registries.
//!
//! The three concrete registries
//! ([`crate::mask_effects::MaskEffectRegistry`],
//! [`crate::plane_effects::PlaneEffectRegistry`],
//! [`crate::post_effects::PostEffectRegistry`]) are thin newtypes
//! over `Registry<dyn TraitObject>` that exist only to self-document
//! which kind of effect they hold and to give the public API a
//! concrete type name.
//!
//! Each entry pairs a factory closure (`Fn() -> Box<E>`) with a
//! `&'static EffectMetadata` reference. Metadata lives in the registry
//! (not on the trait) so a caller introspecting the inventory never
//! has to build an instance.

use std::collections::BTreeMap;

use fluxframe_core::EffectMetadata;
use fluxframe_core::error::EffectError;

/// One row of the registry: factory closure + descriptor.
pub(crate) struct Entry<E: ?Sized> {
    pub(crate) factory: Box<dyn Fn() -> Box<E> + Send + Sync>,
    pub(crate) metadata: &'static EffectMetadata,
}

/// Name-keyed registry of effect factories + metadata. Holds entries
/// for an effect trait `E: ?Sized` chosen by the caller — typically a
/// `dyn MaskEffect` / `dyn PlaneEffect` / `dyn PostEffect`.
pub(crate) struct Registry<E: ?Sized> {
    entries: BTreeMap<&'static str, Entry<E>>,
}

impl<E: ?Sized> Default for Registry<E> {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }
}

impl<E: ?Sized + 'static> Registry<E> {
    pub(crate) fn register(
        &mut self,
        name: &'static str,
        factory: Box<dyn Fn() -> Box<E> + Send + Sync>,
        metadata: &'static EffectMetadata,
    ) {
        self.entries.insert(name, Entry { factory, metadata });
    }

    pub(crate) fn build(&self, name: &str) -> Option<Box<E>> {
        self.entries.get(name).map(|e| (e.factory)())
    }

    pub(crate) fn metadata(&self, name: &str) -> Option<&'static EffectMetadata> {
        self.entries.get(name).map(|e| e.metadata)
    }

    pub(crate) fn iter_metadata(&self) -> impl Iterator<Item = &'static EffectMetadata> + '_ {
        self.entries.values().map(|e| e.metadata)
    }

    pub(crate) fn names(&self) -> Vec<&'static str> {
        self.entries.keys().copied().collect()
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Build a chain from a list of effect names. The two
    /// caller-supplied strings parameterise the
    /// [`EffectError::InvalidConfig`] diagnostic — `unknown_kind`
    /// fills `"unknown {kind} effect: not registered"` and
    /// `unknown_hint` becomes the hint line.
    pub(crate) fn build_chain<S: AsRef<str>>(
        &self,
        names: &[S],
        unknown_kind: &str,
        unknown_hint: &str,
    ) -> Result<Vec<Box<E>>, EffectError> {
        names
            .iter()
            .map(|name| {
                let n = name.as_ref();
                self.build(n).ok_or_else(|| EffectError::InvalidConfig {
                    name: n.to_string(),
                    reason: format!("unknown {unknown_kind} effect: not registered"),
                    hint: Some(unknown_hint.to_string()),
                })
            })
            .collect()
    }
}

/// Shared test helper: walk every effect's metadata in the registry,
/// build a TOML table with the declared defaults, and assert
/// `configure()` accepts it. Used by the per-section
/// `metadata_defaults_round_trip_through_configure` tests so the body
/// of that test lives in one place.
///
/// The `configure` closure parameter is what differs between
/// mask/plane/post — each effect trait has its own `configure()`
/// signature, so the caller passes a tiny adapter that forwards to
/// the right method.
#[cfg(test)]
pub(crate) fn assert_metadata_defaults_round_trip<E>(
    reg: &Registry<E>,
    configure: impl Fn(&mut Box<E>, toml::Value) -> Result<(), fluxframe_core::EffectError>,
) where
    E: ?Sized + 'static,
{
    use fluxframe_core::metadata::ParamKind;
    'effects: for name in reg.names() {
        let meta = reg.metadata(name).expect("metadata");
        let mut effect = reg.build(name).expect("factory");
        let mut table = toml::map::Map::new();
        for p in meta.params {
            // The `Path` arms and the trailing wildcard all `continue`,
            // but the explicit `Path { default: None, required: false }`
            // documents *why* unset optional paths skip; collapsing it
            // into `_` would hide that intent.  The wildcard exists
            // solely for cross-crate `#[non_exhaustive]`.
            #[allow(clippy::match_same_arms)]
            let value = match p.kind {
                ParamKind::Float { default, .. } => toml::Value::Float(f64::from(default)),
                ParamKind::Integer { default, .. } => toml::Value::Integer(default),
                ParamKind::Bool { default } => toml::Value::Boolean(default),
                ParamKind::Color { default } => toml::Value::Array(vec![
                    toml::Value::Integer(default[0].into()),
                    toml::Value::Integer(default[1].into()),
                    toml::Value::Integer(default[2].into()),
                ]),
                ParamKind::Path {
                    default: Some(p), ..
                } => toml::Value::String(p.to_string()),
                ParamKind::Path {
                    default: None,
                    required: false,
                    ..
                } => continue,
                ParamKind::Path {
                    default: None,
                    required: true,
                    ..
                } => {
                    // image_fill-shaped param: no defaultable path.
                    // Skip the whole effect — it would also fail the
                    // trip with an empty table.
                    continue 'effects;
                }
                ParamKind::Enum { default, .. } => toml::Value::String(default.to_string()),
                // `ParamKind` is `#[non_exhaustive]` cross-crate.  Any
                // future variant has to teach this helper how to
                // synthesise a TOML default; until then, treat
                // unknown variants as "skip the param" rather than
                // silently materialising an incorrect value.
                _ => continue,
            };
            table.insert(p.name.to_string(), value);
        }
        let params = toml::Value::Table(table);
        configure(&mut effect, params)
            .unwrap_or_else(|e| panic!("effect '{name}' rejected its own METADATA defaults: {e}"));
    }
}
