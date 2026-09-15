//! Control state that outlives any single pipeline run.
//!
//! The daemon rebuilds its pipelines whenever the camera goes away —
//! in-place recovery, or a full restart under `device = "auto"` — but
//! what an operator sees through the control socket must not change
//! across those rebuilds: the active preset, its unsaved edits and the
//! loaded configuration belong to the process, not to a run.
//! [`ControlSession`] holds that state. It is created once in
//! `commands::run` and lent to every run, so control commands always
//! see — and every rebuilt chain starts from — the same preset.

use std::path::PathBuf;

use fluxframe_core::{FluxConfig, FluxError, Preset};
use fluxframe_effects::EffectChain;

use crate::preset;

/// Process-lifetime control state; see the module docs.
pub(crate) struct ControlSession {
    /// Configuration as loaded, CLI overrides applied. Replaced by
    /// `reload`.
    pub(crate) cfg: FluxConfig,
    /// TOML file `reload` and `save_preset` work on; `None` when the
    /// daemon runs without one.
    pub(crate) config_path: Option<PathBuf>,
    /// Name of the active preset.
    pub(crate) active_preset_name: String,
    /// Working copy of the active preset, unsaved edits included.
    pub(crate) active_preset: Preset,
}

impl ControlSession {
    /// Start a session on the preset `requested`, or on `default` when
    /// none is requested.
    ///
    /// # Errors
    ///
    /// [`FluxError::Config`] when the preset is not defined in `cfg`.
    pub(crate) fn new(
        cfg: FluxConfig,
        config_path: Option<PathBuf>,
        requested: Option<&str>,
    ) -> Result<Self, FluxError> {
        let (name, preset) = preset::resolve(&cfg, requested)?;
        let (active_preset_name, active_preset) = (name.to_string(), preset.clone());
        Ok(Self {
            cfg,
            config_path,
            active_preset_name,
            active_preset,
        })
    }

    /// Build a fresh effect chain for the active preset, unsaved edits
    /// included.
    ///
    /// # Errors
    ///
    /// Propagates [`preset::build_chain`] failures.
    pub(crate) fn build_chain(&self) -> Result<EffectChain, FluxError> {
        preset::build_chain(&self.active_preset_name, &self.active_preset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with(names: &[&str]) -> FluxConfig {
        let mut cfg = FluxConfig::default();
        for name in names {
            cfg.presets.insert((*name).to_string(), Preset::default());
        }
        cfg
    }

    #[test]
    fn new_starts_on_the_default_preset_when_none_is_requested() {
        let session = ControlSession::new(cfg_with(&["default", "blur"]), None, None)
            .expect("default preset exists");
        assert_eq!(session.active_preset_name, "default");
    }

    #[test]
    fn new_rejects_an_unknown_preset() {
        assert!(ControlSession::new(cfg_with(&["default"]), None, Some("missing")).is_err());
    }

    #[test]
    fn build_chain_uses_the_working_copy() {
        let mut session =
            ControlSession::new(cfg_with(&["default"]), None, None).expect("default preset exists");
        // A preset with a background but no mask is rejected by the
        // builder, so an edited working copy must be what gets built.
        session.active_preset.background = Some(fluxframe_core::PipelineSection::default());
        assert!(session.build_chain().is_err());
    }
}
