//! Request bookkeeping and reply handling for the daemon session.
//!
//! [`apply_reply`] is the pure core of the AppModel's reply path: it
//! folds one daemon reply into [`AppState`] and returns the UI work the
//! caller has to perform ([`ReplyEffects`]). Keeping widgets out of it
//! makes every reply transition unit-testable without a display.

use std::collections::HashMap;

use fluxframe_core::protocol::Response;
use fluxframe_core::{SetPath, SubchainKind};
use serde_json::Value;

use crate::state::{AppState, ConfigSync};

/// Classification of an in-flight request, used to interpret its reply
/// without inspecting `data` shape heuristics.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PendingKind {
    /// A `Command::Set` issued by a widget change. Rollback on `Err`
    /// works by leaving `active_config` untouched and resetting the
    /// one widget, so only the requested value is recorded.
    SetParam {
        /// Parameter the widget changed.
        path: SetPath,
        /// Value the daemon is being asked to apply.
        value: Value,
    },
    /// A `Command::GetConfig { path: None }` — the `Ok` payload is the
    /// full preset tree, installed according to the carried
    /// [`ConfigSync`].
    GetConfig(ConfigSync),
    /// A `Command::GetConfig { path: None }` keeping the baseline, sent
    /// after a toggle the GUI already mirrored locally. Its snapshot is
    /// installed without a page rebuild unless a chain changed shape.
    GetConfigQuiet,
    /// A `Command::SetEnabled` from an effect row's switch.
    SetEnabled {
        /// Sub-chain of the effect.
        section: SubchainKind,
        /// Effect name.
        effect: String,
        /// Requested flag.
        enabled: bool,
    },
    /// A `Command::SetPreset` switching to another preset. The clean
    /// baseline is refetched only once the daemon accepted the switch.
    SetPreset,
    /// A `Command::SetPreset` re-activating the active preset to drop
    /// unsaved edits: the daemon rebuilds it from its persisted copy,
    /// whereas `get_config` would echo the in-memory edits back.
    Revert,
    /// A `Command::Reload`. On `Ok` the preset list and the active
    /// preset are refetched, since the file on disk may have changed
    /// both.
    Reload,
    /// A `Command::ListPresets` — the `Ok` payload is the array of
    /// preset names.
    ListPresets,
    /// A `Command::CurrentPreset` — the `Ok` payload is a JSON string
    /// with the active preset name.
    CurrentPreset,
    /// A `Command::SavePreset` — on `Ok` the in-memory state becomes
    /// the baseline without another `GetConfig` round-trip.
    Save,
    /// A `Command::SavePresetAs` — on `Ok` the new name joins the preset
    /// list. The active preset does not change, so dirty is left as is.
    SaveAs {
        /// Name of the preset being created.
        new_name: String,
    },
    /// Anything else (SetChain, …). Tracked so their errors still
    /// reach the operator.
    Other,
}

/// In-flight requests keyed by their wire tag.
#[derive(Debug, Default)]
pub(crate) struct PendingRequests {
    next_tag: u64,
    kinds: HashMap<u64, PendingKind>,
}

impl PendingRequests {
    /// Allocate a fresh tag for a request of `kind`.
    pub(crate) fn register(&mut self, kind: PendingKind) -> u64 {
        let tag = self.next_tag;
        self.next_tag = self.next_tag.wrapping_add(1);
        self.kinds.insert(tag, kind);
        tag
    }

    /// Remove and return the kind registered for `tag`.
    pub(crate) fn take(&mut self, tag: u64) -> Option<PendingKind> {
        self.kinds.remove(&tag)
    }

    /// Forget every in-flight request (their replies will never arrive).
    /// Tags keep increasing, so a late reply cannot match a new request.
    pub(crate) fn clear(&mut self) {
        self.kinds.clear();
    }
}

/// Severity of a [`Notice`]; decides how long its toast stays up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NoticeKind {
    /// The daemon refused a request the operator cares about.
    Error,
    /// A request the operator triggered succeeded.
    Confirmation,
}

/// A message to surface to the operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Notice {
    /// Severity.
    pub(crate) kind: NoticeKind,
    /// One-line text.
    pub(crate) message: String,
}

/// UI work required after a reply was folded into the state. The
/// header is always refreshed by the caller, so it is not listed.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "one independent flag per follow-up the caller performs"
)]
pub(crate) struct ReplyEffects {
    /// Re-render the chain page from `active_config`.
    pub(crate) rebuild_chain: bool,
    /// Reset the widget of this parameter to its `active_config` value
    /// (the daemon refused the write).
    pub(crate) reset_param: Option<SetPath>,
    /// Show the enable switch of this effect as `active_config` says
    /// (after an accepted or refused `set_enabled`).
    pub(crate) sync_toggle: Option<(SubchainKind, String)>,
    /// Request the config without a page rebuild (baseline kept).
    pub(crate) refetch_config_quiet: bool,
    /// Re-populate the preset drop-down (list or active entry changed).
    pub(crate) presets_changed: bool,
    /// Request the active preset name and config as a clean baseline.
    pub(crate) refetch_active: bool,
    /// Request the preset list.
    pub(crate) refetch_presets: bool,
    /// Toast to show.
    pub(crate) notice: Option<Notice>,
}

/// Fold the reply to a request of `kind` (`None` for an unknown tag)
/// into `state` and report the UI work it requires.
pub(crate) fn apply_reply(
    state: &mut AppState,
    kind: Option<PendingKind>,
    response: Response,
) -> ReplyEffects {
    let mut effects = ReplyEffects::default();
    match response {
        Response::Ok { data } => match kind {
            Some(PendingKind::SetParam { path, value }) => {
                // Mirror the accepted write so rebuilds render it; it
                // is a delta against the persisted baseline.
                state.set_config_field(&path, value);
            }
            Some(PendingKind::CurrentPreset) => {
                if let Some(name) = data.as_str() {
                    state.active_preset = Some(name.to_string());
                    effects.presets_changed = true;
                }
            }
            Some(PendingKind::GetConfig(sync)) => {
                state.apply_config_snapshot(data, sync);
                effects.rebuild_chain = true;
            }
            Some(PendingKind::GetConfigQuiet) => {
                let shape = |state: &AppState| SubchainKind::ALL.map(|s| state.chain_for(s));
                let before = shape(state);
                state.apply_config_snapshot(data, ConfigSync::KeepBaseline);
                // Widgets already show this state; only a chain that
                // changed shape meanwhile needs new rows.
                effects.rebuild_chain = shape(state) != before;
            }
            Some(PendingKind::SetEnabled {
                section,
                effect,
                enabled,
            }) => {
                state.set_effect_enabled(section, &effect, enabled);
                effects.sync_toggle = Some((section, effect));
                // Confirm the mirror against the daemon's own view.
                effects.refetch_config_quiet = true;
            }
            Some(PendingKind::SetPreset) => effects.refetch_active = true,
            Some(PendingKind::Revert) => {
                effects.refetch_active = true;
                effects.notice = Some(confirmation("Unsaved changes discarded".into()));
            }
            Some(PendingKind::Reload) => {
                effects.refetch_active = true;
                effects.refetch_presets = true;
                effects.notice = Some(confirmation("Configuration reloaded".into()));
            }
            Some(PendingKind::ListPresets) => match crate::ipc::wire::parse_presets(&data) {
                Ok(names) => {
                    state.presets = names;
                    effects.presets_changed = true;
                }
                Err(e) => tracing::warn!(error = %e, "malformed list_presets reply"),
            },
            Some(PendingKind::Save) => {
                state.mark_saved();
                let message = match state.config_path.as_deref() {
                    Some(p) => format!("Saved to {}", p.display()),
                    None => "Saved".to_string(),
                };
                effects.notice = Some(confirmation(message));
            }
            Some(PendingKind::SaveAs { new_name }) => {
                effects.notice = Some(confirmation(format!("Saved as preset “{new_name}”")));
                if !state.presets.contains(&new_name) {
                    state.presets.push(new_name);
                    effects.presets_changed = true;
                }
            }
            // SetChain or an untracked tag — the follow-up GetConfig
            // refreshes the UI.
            Some(PendingKind::Other) | None => {}
        },
        Response::Err { error, hint } => {
            tracing::warn!(error = %error, hint = ?hint, "reply err");
            match kind {
                Some(PendingKind::SetParam { path, .. }) => {
                    // `active_config` still holds the previous value
                    // (we only commit on Ok), so resetting the widget
                    // from it snaps the control back.
                    effects.notice = Some(error_notice(&error, hint.as_deref()));
                    effects.reset_param = Some(path);
                }
                Some(PendingKind::SetEnabled {
                    section, effect, ..
                }) => {
                    // `active_config` is untouched, so syncing the switch
                    // from it flips it back.
                    effects.notice = Some(error_notice(&error, hint.as_deref()));
                    effects.sync_toggle = Some((section, effect));
                }
                Some(
                    PendingKind::Other
                    | PendingKind::Save
                    | PendingKind::SaveAs { .. }
                    | PendingKind::SetPreset
                    | PendingKind::Revert
                    | PendingKind::Reload,
                ) => {
                    // No rollback and no refetch: the daemon kept its
                    // state, so the dirty baseline must stay put. The
                    // operator still needs to know it refused.
                    effects.notice = Some(error_notice(&error, hint.as_deref()));
                }
                // Internal refetches and unknown tags — the warn above
                // is enough.
                Some(
                    PendingKind::CurrentPreset
                    | PendingKind::GetConfig(_)
                    | PendingKind::GetConfigQuiet
                    | PendingKind::ListPresets,
                )
                | None => {}
            }
        }
        // `Response` is `#[non_exhaustive]`; tolerate future variants.
        _ => tracing::warn!("unrecognised Response variant"),
    }
    effects
}

fn confirmation(message: String) -> Notice {
    Notice {
        kind: NoticeKind::Confirmation,
        message,
    }
}

/// One-line error text from the daemon's reason and optional hint.
/// Kept short — toasts wrap awkwardly past one line.
fn error_notice(error: &str, hint: Option<&str>) -> Notice {
    let message = match hint {
        Some(h) if !h.is_empty() => format!("{error} — {h}"),
        _ => error.to_string(),
    };
    Notice {
        kind: NoticeKind::Error,
        message,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use fluxframe_core::parse_set_path;
    use serde_json::json;

    use super::*;

    fn state() -> AppState {
        let mut s = AppState::new(PathBuf::from("/tmp/test.sock"));
        s.presets = vec!["a".into(), "b".into()];
        s.active_preset = Some("a".into());
        s.apply_config_snapshot(
            crate::state::daemon_config(
                "[background]\nchain = [\"blur\"]\n\n[background.blur]\nradius = 30\n",
            ),
            ConfigSync::ResyncBaseline,
        );
        s
    }

    fn set_radius(v: i64) -> PendingKind {
        PendingKind::SetParam {
            path: parse_set_path("background.blur.radius").expect("valid"),
            value: json!(v),
        }
    }

    fn ok(data: Value) -> Response {
        Response::Ok { data }
    }

    fn err(error: &str, hint: Option<&str>) -> Response {
        Response::Err {
            error: error.into(),
            hint: hint.map(str::to_string),
        }
    }

    #[test]
    fn pending_requests_hand_out_unique_tags_and_forget_on_clear() {
        let mut p = PendingRequests::default();
        let t1 = p.register(PendingKind::Save);
        let t2 = p.register(PendingKind::Other);
        assert_ne!(t1, t2);
        assert_eq!(p.take(t1), Some(PendingKind::Save));
        assert_eq!(p.take(t1), None);
        p.clear();
        assert_eq!(p.take(t2), None);
        assert!(p.register(PendingKind::Other) > t2);
    }

    #[test]
    fn accepted_set_marks_dirty_without_rebuild() {
        let mut s = state();
        let fx = apply_reply(&mut s, Some(set_radius(12)), ok(Value::Null));
        assert!(s.is_dirty());
        assert_eq!(fx, ReplyEffects::default());
    }

    #[test]
    fn rejected_set_keeps_config_rebuilds_and_reports() {
        let mut s = state();
        let fx = apply_reply(
            &mut s,
            Some(set_radius(9999)),
            err("out of range", Some("max 256")),
        );
        assert!(!s.is_dirty());
        // Regression: a refusal must reset only the one widget; a full
        // page rebuild destroys a slider mid-drag.
        assert!(!fx.rebuild_chain);
        assert_eq!(
            fx.reset_param,
            Some(parse_set_path("background.blur.radius").expect("valid"))
        );
        assert_eq!(
            fx.notice,
            Some(Notice {
                kind: NoticeKind::Error,
                message: "out of range — max 256".into()
            })
        );
    }

    #[test]
    fn chain_refetch_keeps_unsaved_edits_dirty() {
        let mut s = state();
        apply_reply(&mut s, Some(set_radius(12)), ok(Value::Null));
        let daemon_view = s.active_config.clone();
        let fx = apply_reply(
            &mut s,
            Some(PendingKind::GetConfig(ConfigSync::KeepBaseline)),
            ok(daemon_view),
        );
        assert!(fx.rebuild_chain);
        assert!(s.is_dirty());
    }

    /// Regression: Revert used to resync the baseline to `get_config`,
    /// which echoes the daemon's in-memory edits back — the dirty
    /// marker vanished while the edits stayed live. The baseline may
    /// only move once the daemon has re-activated the persisted preset.
    #[test]
    fn revert_waits_for_daemon_before_touching_baseline() {
        let mut s = state();
        apply_reply(&mut s, Some(set_radius(12)), ok(Value::Null));
        let fx = apply_reply(&mut s, Some(PendingKind::Revert), ok(Value::Null));
        assert!(fx.refetch_active);
        assert!(
            s.is_dirty(),
            "baseline moves only with the refetched snapshot"
        );
    }

    /// Regression: a refused Reload / SetPreset / Revert used to be
    /// followed by an unconditional clean-baseline refetch, silently
    /// clearing the dirty marker over edits the daemon still holds.
    #[test]
    fn refused_preset_commands_keep_dirty_and_skip_refetch() {
        for kind in [
            PendingKind::Reload,
            PendingKind::SetPreset,
            PendingKind::Revert,
        ] {
            let mut s = state();
            apply_reply(&mut s, Some(set_radius(12)), ok(Value::Null));
            let fx = apply_reply(&mut s, Some(kind.clone()), err("no config path", None));
            assert!(!fx.refetch_active, "{kind:?}");
            assert!(!fx.refetch_presets, "{kind:?}");
            assert!(s.is_dirty(), "{kind:?}");
            assert!(fx.notice.is_some(), "{kind:?}");
        }
    }

    #[test]
    fn accepted_reload_refetches_presets_and_active_and_confirms() {
        let mut s = state();
        let fx = apply_reply(&mut s, Some(PendingKind::Reload), ok(Value::Null));
        assert!(fx.refetch_active);
        assert!(fx.refetch_presets);
        assert_eq!(fx.notice.map(|n| n.kind), Some(NoticeKind::Confirmation));
    }

    fn set_enabled(effect: &str, enabled: bool) -> PendingKind {
        PendingKind::SetEnabled {
            section: SubchainKind::Background,
            effect: effect.into(),
            enabled,
        }
    }

    /// An accepted toggle updates state and the switch in place and
    /// asks for a quiet refetch — no page rebuild, so an in-flight
    /// slider drag elsewhere is not destroyed.
    #[test]
    fn accepted_set_enabled_mirrors_locally_without_rebuild() {
        let mut s = state();
        let fx = apply_reply(&mut s, Some(set_enabled("blur", false)), ok(Value::Null));
        assert!(!s.effect_enabled(SubchainKind::Background, "blur"));
        assert!(s.is_dirty());
        assert!(!fx.rebuild_chain);
        assert!(fx.refetch_config_quiet);
        assert_eq!(
            fx.sync_toggle,
            Some((SubchainKind::Background, "blur".to_string()))
        );
    }

    #[test]
    fn refused_set_enabled_resyncs_the_switch_and_reports() {
        let mut s = state();
        let fx = apply_reply(
            &mut s,
            Some(set_enabled("blur", false)),
            err("no effect named 'blur'", None),
        );
        assert!(s.effect_enabled(SubchainKind::Background, "blur"));
        assert!(!s.is_dirty());
        assert!(!fx.refetch_config_quiet);
        assert_eq!(
            fx.sync_toggle,
            Some((SubchainKind::Background, "blur".to_string()))
        );
        assert_eq!(fx.notice.map(|n| n.kind), Some(NoticeKind::Error));
    }

    #[test]
    fn quiet_refetch_rebuilds_only_when_a_chain_changed_shape() {
        let mut s = state();
        let same_shape = crate::state::daemon_config(
            "[background]\nchain = [\"blur\"]\n\n[background.blur]\nenabled = false\nradius = 30\n",
        );
        let fx = apply_reply(&mut s, Some(PendingKind::GetConfigQuiet), ok(same_shape));
        assert!(!fx.rebuild_chain);
        assert!(s.is_dirty(), "quiet refetch keeps the baseline");

        let new_shape =
            crate::state::daemon_config("[background]\nchain = [\"blur\", \"vignette\"]\n");
        let fx = apply_reply(&mut s, Some(PendingKind::GetConfigQuiet), ok(new_shape));
        assert!(fx.rebuild_chain);
    }

    #[test]
    fn list_presets_replaces_names() {
        let mut s = state();
        let fx = apply_reply(
            &mut s,
            Some(PendingKind::ListPresets),
            ok(json!(["x", "y"])),
        );
        assert!(fx.presets_changed);
        assert_eq!(s.presets, vec!["x", "y"]);
    }

    #[test]
    fn save_clears_dirty_and_confirms_with_target() {
        let mut s = state();
        s.config_path = Some(PathBuf::from("/etc/ff.toml"));
        apply_reply(&mut s, Some(set_radius(12)), ok(Value::Null));
        let fx = apply_reply(&mut s, Some(PendingKind::Save), ok(Value::Null));
        assert!(!s.is_dirty());
        assert_eq!(
            fx.notice,
            Some(Notice {
                kind: NoticeKind::Confirmation,
                message: "Saved to /etc/ff.toml".into()
            })
        );
    }

    #[test]
    fn save_as_adds_new_preset_once() {
        let mut s = state();
        let fx = apply_reply(
            &mut s,
            Some(PendingKind::SaveAs {
                new_name: "c".into(),
            }),
            ok(Value::Null),
        );
        assert!(fx.presets_changed);
        assert_eq!(s.presets, vec!["a", "b", "c"]);
        let fx = apply_reply(
            &mut s,
            Some(PendingKind::SaveAs {
                new_name: "c".into(),
            }),
            ok(Value::Null),
        );
        assert!(!fx.presets_changed);
        assert_eq!(s.presets.len(), 3);
    }

    #[test]
    fn current_preset_updates_active_name() {
        let mut s = state();
        let fx = apply_reply(&mut s, Some(PendingKind::CurrentPreset), ok(json!("b")));
        assert_eq!(s.active_preset.as_deref(), Some("b"));
        assert!(fx.presets_changed);
    }

    #[test]
    fn internal_refetch_errors_and_unknown_tags_are_silent() {
        let mut s = state();
        for kind in [
            Some(PendingKind::CurrentPreset),
            Some(PendingKind::GetConfig(ConfigSync::ResyncBaseline)),
            Some(PendingKind::ListPresets),
            None,
        ] {
            assert_eq!(
                apply_reply(&mut s, kind, err("boom", None)),
                ReplyEffects::default()
            );
        }
    }

    #[test]
    fn user_request_errors_are_reported() {
        let mut s = state();
        for kind in [PendingKind::Other, PendingKind::Save, PendingKind::Revert] {
            let fx = apply_reply(&mut s, Some(kind), err("nope", Some("")));
            assert_eq!(fx.notice.map(|n| n.message), Some("nope".into()));
            assert!(!fx.rebuild_chain);
        }
    }
}
