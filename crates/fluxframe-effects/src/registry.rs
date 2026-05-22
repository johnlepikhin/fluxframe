//! Effect registry: maps an effect name (snake_case) to a factory that
//! constructs a fresh effect instance.
//!
//! Factories are registered at startup by the CLI; the registry itself
//! has no built-in knowledge of any effect, so adding a new effect is one
//! line in `fluxframe-cli` plus the effect implementation.

use std::collections::BTreeMap;

use fluxframe_core::error::EffectError;
use fluxframe_core::traits::VideoEffect;

/// Constructs a fresh effect instance.  Stateless factory — state lives
/// inside the effect.
pub trait EffectFactory: Send + Sync {
    /// Build a brand-new effect instance.  Called once per chain slot.
    fn build(&self) -> Box<dyn VideoEffect>;
}

impl<F> EffectFactory for F
where
    F: Fn() -> Box<dyn VideoEffect> + Send + Sync,
{
    fn build(&self) -> Box<dyn VideoEffect> {
        self()
    }
}

/// Name-keyed map of effect factories used to materialise an
/// [`crate::EffectChain`] from a list of effect names.
#[derive(Default)]
pub struct EffectRegistry {
    factories: BTreeMap<&'static str, Box<dyn EffectFactory>>,
}

impl EffectRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a factory under the canonical (snake_case) effect name.
    /// Re-registering the same name overwrites the previous factory.
    pub fn register(&mut self, name: &'static str, factory: Box<dyn EffectFactory>) {
        self.factories.insert(name, factory);
    }

    /// Look up a factory.  Returns `None` if the name was never registered.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn EffectFactory> {
        self.factories.get(name).map(std::convert::AsRef::as_ref)
    }

    /// Sorted list of registered effect names (`BTreeMap` ordering).
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.factories.keys().copied().collect()
    }

    /// Number of registered factories.
    #[must_use]
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// Returns `true` if no factory has been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// Returns `true` if a factory is registered under `name`.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }

    /// Build an [`EffectChain`] from a list of effect names.  Fails fast
    /// on the first unknown name — partial chains never leak out.
    ///
    /// Accepts any slice of string-like values (`&str`, `String`, `Arc<str>`)
    /// via [`AsRef<str>`], so callers do not have to allocate a `Vec<String>`
    /// when their inputs are already borrowed.
    ///
    /// [`EffectChain`]: crate::chain::EffectChain
    ///
    /// # Errors
    ///
    /// Returns [`EffectError::InvalidConfig`] if any requested name is not
    /// registered; the offending name appears in the error payload.
    pub fn build_chain<S: AsRef<str>>(
        &self,
        names: &[S],
    ) -> Result<crate::chain::EffectChain, EffectError> {
        let effects: Vec<Box<dyn VideoEffect>> =
            names
                .iter()
                .map(|name| {
                    let n = name.as_ref();
                    self.get(n).map(EffectFactory::build).ok_or_else(|| {
                        EffectError::InvalidConfig {
                        name: n.to_string(),
                        reason: "unknown effect: not registered".into(),
                        hint: Some(
                            "see `fluxframe check --effect <name>` for the list of built-in effects"
                                .into(),
                        ),
                    }
                    })
                })
                .collect::<Result<_, _>>()?;
        Ok(crate::chain::EffectChain::new(effects))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxframe_core::context::{FrameContext, ProcessingContext};
    use fluxframe_core::frame::VideoFrame;
    use fluxframe_core::traits::RawEffectParams;

    /// Minimal effect used only to verify factory wiring.
    struct DummyEffect(&'static str);

    impl VideoEffect for DummyEffect {
        fn name(&self) -> &'static str {
            self.0
        }

        fn configure(&mut self, _params: RawEffectParams) -> Result<(), EffectError> {
            Ok(())
        }

        fn prepare(&mut self, _context: &ProcessingContext) -> Result<(), EffectError> {
            Ok(())
        }

        fn process(
            &mut self,
            _frame: &mut VideoFrame,
            _context: &mut FrameContext,
        ) -> Result<(), EffectError> {
            Ok(())
        }
    }

    fn dummy(label: &'static str) -> Box<dyn EffectFactory> {
        Box::new(move || -> Box<dyn VideoEffect> { Box::new(DummyEffect(label)) })
    }

    #[test]
    fn build_chain_returns_effects_in_request_order() {
        let mut reg = EffectRegistry::new();
        reg.register("foo", dummy("foo"));
        reg.register("bar", dummy("bar"));

        let chain = reg
            .build_chain(&["bar".to_string(), "foo".to_string()])
            .expect("build ok");
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.names(), vec!["bar", "foo"]);
    }

    #[test]
    fn build_chain_accepts_str_slice() {
        let mut reg = EffectRegistry::new();
        reg.register("foo", dummy("foo"));
        reg.register("bar", dummy("bar"));

        let chain = reg.build_chain(&["foo", "bar"]).expect("build ok");
        assert_eq!(chain.names(), vec!["foo", "bar"]);
    }

    #[test]
    fn build_chain_fails_on_unknown_name() {
        let registry = EffectRegistry::new();
        let res = registry.build_chain(&["unknown".to_string()]);
        assert!(matches!(
            res,
            Err(EffectError::InvalidConfig { ref name, .. }) if name == "unknown"
        ));
    }

    #[test]
    fn build_chain_empty_input_returns_empty_chain() {
        let reg = EffectRegistry::new();
        let chain = reg.build_chain::<&str>(&[]).expect("empty build ok");
        assert!(chain.is_empty());
    }

    #[test]
    fn register_overrides_existing_factory() {
        let mut reg = EffectRegistry::new();
        reg.register("foo", dummy("first"));
        reg.register("foo", dummy("second"));

        let chain = reg.build_chain(&["foo".to_string()]).expect("build ok");
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.names(), vec!["second"]);
    }

    #[test]
    fn names_returns_registered_keys_sorted() {
        let mut reg = EffectRegistry::new();
        reg.register("b", dummy("b"));
        reg.register("a", dummy("a"));
        reg.register("c", dummy("c"));
        assert_eq!(reg.names(), vec!["a", "b", "c"]);
    }

    #[test]
    fn get_returns_none_for_unknown() {
        let reg = EffectRegistry::new();
        assert!(reg.get("missing").is_none());
    }

    #[test]
    fn len_is_empty_and_contains_track_registration() {
        let mut reg = EffectRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(!reg.contains("foo"));

        reg.register("foo", dummy("foo"));
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 1);
        assert!(reg.contains("foo"));
        assert!(!reg.contains("bar"));
    }
}
