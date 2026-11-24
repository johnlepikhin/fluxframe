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

use fluxframe_core::context::ProcessingContext;
use fluxframe_core::paths::ConfigBase;
use fluxframe_core::{FluxConfig, FluxError, Preset};
use fluxframe_effects::EffectChain;
use tracing::warn;

use crate::preset::{self, PreparedPreset};

/// Process-lifetime control state; see the module docs.
pub(crate) struct ControlSession {
    /// Configuration as loaded, CLI overrides applied. Replaced by
    /// `reload`.
    pub(crate) cfg: FluxConfig,
    /// TOML file `reload` and `save_preset` work on; `None` when the
    /// daemon runs without one.
    pub(crate) config_path: Option<PathBuf>,
    /// Directory relative paths in the configuration resolve against:
    /// the directory of `config_path`.
    pub(crate) config_base: ConfigBase,
    /// Name of the active preset.
    pub(crate) active_preset_name: String,
    /// Working copy of the active preset, unsaved edits included.
    pub(crate) active_preset: Preset,
    /// The preset the operator selected, when a `fallback_preset` of it
    /// runs instead; `None` when the selected preset itself runs.
    pub(crate) fallen_back_from: Option<String>,
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
        let config_base = ConfigBase::for_config(config_path.as_deref());
        Ok(Self {
            cfg,
            config_path,
            config_base,
            active_preset_name,
            active_preset,
            fallen_back_from: None,
        })
    }

    /// Prepare `chain`, built from the active preset; when the preset
    /// cannot run for lack of an accelerator, switch to its fallback
    /// and return that chain instead.
    ///
    /// # Errors
    ///
    /// The chain fails to prepare for any other reason, the preset has
    /// no fallback, or the fallback fails too.
    pub(crate) fn prepare(
        &mut self,
        mut chain: EffectChain,
        ctx: &ProcessingContext,
    ) -> Result<EffectChain, FluxError> {
        match chain.prepare_all(ctx) {
            Ok(()) => Ok(chain),
            Err(e) if e.is_accelerator_unavailable() => {
                if self.active_preset.fallback_preset.is_some() {
                    warn!(
                        preset = %self.active_preset_name,
                        error = %e,
                        "preset cannot run on this machine; switching to its fallback preset",
                    );
                }
                self.fall_back(ctx)?.ok_or_else(|| e.into())
            }
            Err(e) => Err(e.into()),
        }
    }

    /// The preset the operator selected: the active one, unless a
    /// fallback of it runs.
    pub(crate) fn requested_preset_name(&self) -> &str {
        self.fallen_back_from
            .as_deref()
            .unwrap_or(&self.active_preset_name)
    }

    /// Make `prepared` the active preset, selected by the operator as
    /// `requested`; returns its chain.
    pub(crate) fn activate(&mut self, prepared: PreparedPreset, requested: &str) -> EffectChain {
        self.fallen_back_from = (prepared.name != requested).then(|| requested.to_string());
        self.active_preset_name = prepared.name;
        self.active_preset = prepared.preset;
        prepared.chain
    }

    /// Switch to the active preset's `fallback_preset` (following
    /// further fallbacks while those cannot run either) and return its
    /// prepared chain.  `Ok(None)` when the active preset has no
    /// fallback.  Unsaved edits of the active preset are dropped.
    ///
    /// # Errors
    ///
    /// The fallback preset cannot be built or prepared.
    pub(crate) fn fall_back(
        &mut self,
        ctx: &ProcessingContext,
    ) -> Result<Option<EffectChain>, FluxError> {
        let Some(fallback) = self.active_preset.fallback_preset.clone() else {
            return Ok(None);
        };
        let preset = self
            .cfg
            .presets
            .get(&fallback)
            .ok_or_else(|| FluxError::Config {
                reason: format!("fallback_preset '{fallback}' is not defined"),
                hint: None,
            })?;
        let prepared =
            preset::prepare_with_fallback(&self.cfg, &self.config_base, &fallback, preset, ctx)?;
        let requested = self.requested_preset_name().to_string();
        warn!(
            requested = %requested,
            active = %prepared.name,
            "running the fallback preset",
        );
        Ok(Some(self.activate(prepared, &requested)))
    }

    /// Build a fresh effect chain for the active preset, unsaved edits
    /// included.
    ///
    /// # Errors
    ///
    /// Propagates [`preset::build_chain`] failures.
    pub(crate) fn build_chain(&self) -> Result<EffectChain, FluxError> {
        preset::build_chain(
            &self.active_preset_name,
            &self.active_preset,
            &self.config_base,
        )
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

    #[test]
    fn requested_preset_is_the_active_one_unless_a_fallback_runs() {
        let mut session =
            ControlSession::new(cfg_with(&["default", "lite"]), None, None).expect("default");
        assert_eq!(session.requested_preset_name(), "default");
        let prepared = PreparedPreset {
            chain: EffectChain::new(Vec::new()),
            name: "lite".into(),
            preset: Preset::default(),
        };
        let _ = session.activate(prepared, "default");
        assert_eq!(session.active_preset_name, "lite");
        assert_eq!(session.requested_preset_name(), "default");

        let prepared = PreparedPreset {
            chain: EffectChain::new(Vec::new()),
            name: "default".into(),
            preset: Preset::default(),
        };
        let _ = session.activate(prepared, "default");
        assert_eq!(session.fallen_back_from, None);
    }

    #[test]
    fn fall_back_without_a_fallback_preset_is_none() {
        let mut session = ControlSession::new(cfg_with(&["default"]), None, None).expect("default");
        let ctx = ProcessingContext {
            width: 8,
            height: 8,
            format: fluxframe_core::frame::PixelFormat::Rgb,
            fps: 30,
            counters: None,
        };
        assert!(session.fall_back(&ctx).expect("no error").is_none());
        assert_eq!(session.active_preset_name, "default");
    }
}
