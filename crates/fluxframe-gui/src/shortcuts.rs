//! Window / application actions and their keyboard accelerators.
//!
//! Every user command reachable from the header bar or the keyboard is
//! a `GAction`, so buttons, menu items and shortcuts share one enabled
//! state:
//!
//! - `win.save` (`Ctrl+S`), `win.save-as` (`Ctrl+Shift+S`), `win.revert`
//! - `win.reload` (`Ctrl+R`, `F5`)
//! - `win.preset(u32)` (`Ctrl+1` .. `Ctrl+9`, 1-indexed slot)
//! - `win.shortcuts` (`Ctrl+?`), `win.about`
//! - `app.quit` (`Ctrl+Q`)

use std::num::NonZeroUsize;

use adw::prelude::*;
use gtk::{gio, glib};

use crate::components::Emit;

/// User intent reported by window actions. `Quit` and the shortcuts
/// overview are handled locally and never reach the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActionEvent {
    /// `win.save`.
    Save,
    /// `win.save-as`.
    SaveAs,
    /// `win.revert`.
    Revert,
    /// `win.reload`.
    Reload,
    /// `win.about`.
    About,
    /// `win.preset(slot)` — 1-indexed preset slot.
    PresetSlot(NonZeroUsize),
}

/// Detailed action names, shared with the header bar.
pub(crate) const ACTION_SAVE: &str = "win.save";
/// See [`ACTION_SAVE`].
pub(crate) const ACTION_SAVE_AS: &str = "win.save-as";
/// See [`ACTION_SAVE`].
pub(crate) const ACTION_REVERT: &str = "win.revert";
/// See [`ACTION_SAVE`].
pub(crate) const ACTION_RELOAD: &str = "win.reload";
/// See [`ACTION_SAVE`].
pub(crate) const ACTION_SHORTCUTS: &str = "win.shortcuts";
/// See [`ACTION_SAVE`].
pub(crate) const ACTION_ABOUT: &str = "win.about";
const ACTION_PRESET: &str = "win.preset";
const ACTION_QUIT: &str = "app.quit";

/// Highest preset slot reachable via `Ctrl+<digit>`.
const PRESET_SLOTS: u32 = 9;

/// Actions whose enabled state tracks the connection / dirty state.
pub(crate) struct WindowActions {
    save: gio::SimpleAction,
    save_as: gio::SimpleAction,
    revert: gio::SimpleAction,
    reload: gio::SimpleAction,
    preset: gio::SimpleAction,
}

/// Which state-dependent actions are currently usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "one independent enabled flag per GAction"
)]
pub(crate) struct Enablement {
    pub(crate) save: bool,
    pub(crate) save_as: bool,
    pub(crate) revert: bool,
    pub(crate) reload: bool,
    pub(crate) preset: bool,
}

impl Enablement {
    /// Derive enablement from the connection state, the dirty flag and
    /// whether the daemon reported a writable config path.
    pub(crate) fn compute(connected: bool, dirty: bool, has_target: bool) -> Self {
        Self {
            save: connected && dirty && has_target,
            save_as: connected && has_target,
            revert: connected && dirty,
            reload: connected,
            preset: connected,
        }
    }
}

impl WindowActions {
    /// Apply `enablement` to the actions (and thus to bound widgets).
    pub(crate) fn apply(&self, enablement: Enablement) {
        self.save.set_enabled(enablement.save);
        self.save_as.set_enabled(enablement.save_as);
        self.revert.set_enabled(enablement.revert);
        self.reload.set_enabled(enablement.reload);
        self.preset.set_enabled(enablement.preset);
    }
}

/// Register all actions on `window` / `app`, bind accelerators, and
/// return the state-dependent ones (initially disabled).
pub(crate) fn install(
    app: &adw::Application,
    window: &adw::ApplicationWindow,
    emit: &Emit<ActionEvent>,
) -> WindowActions {
    let save = event_action(window, ACTION_SAVE, emit, ActionEvent::Save);
    let save_as = event_action(window, ACTION_SAVE_AS, emit, ActionEvent::SaveAs);
    let revert = event_action(window, ACTION_REVERT, emit, ActionEvent::Revert);
    let reload = event_action(window, ACTION_RELOAD, emit, ActionEvent::Reload);
    event_action(window, ACTION_ABOUT, emit, ActionEvent::About);

    let preset = gio::SimpleAction::new(short_name(ACTION_PRESET), Some(glib::VariantTy::UINT32));
    {
        let emit = Emit::clone(emit);
        preset.connect_activate(move |_, param| {
            let slot = param
                .and_then(glib::Variant::get::<u32>)
                .and_then(|n| NonZeroUsize::new(n as usize));
            if let Some(slot) = slot {
                emit(ActionEvent::PresetSlot(slot));
            }
        });
    }
    window.add_action(&preset);

    let shortcuts = gio::SimpleAction::new(short_name(ACTION_SHORTCUTS), None);
    shortcuts.connect_activate(glib::clone!(
        #[weak]
        window,
        move |_, _| shortcuts_dialog().present(Some(&window))
    ));
    window.add_action(&shortcuts);

    // Quit by closing windows rather than `app.quit()`, so each
    // window's `close-request` (geometry persistence) still runs.
    let quit = gio::SimpleAction::new(short_name(ACTION_QUIT), None);
    quit.connect_activate(glib::clone!(
        #[weak]
        app,
        move |_, _| {
            for window in app.windows() {
                window.close();
            }
        }
    ));
    app.add_action(&quit);

    for (action, accels) in accelerators() {
        app.set_accels_for_action(&action, &accels);
    }

    let actions = WindowActions {
        save,
        save_as,
        revert,
        reload,
        preset,
    };
    actions.apply(Enablement::compute(false, false, false));
    actions
}

/// Accelerator table: detailed action name → accelerators.
fn accelerators() -> Vec<(String, Vec<&'static str>)> {
    let mut table = vec![
        (ACTION_SAVE.to_string(), vec!["<Control>s"]),
        (ACTION_SAVE_AS.to_string(), vec!["<Control><Shift>s"]),
        (ACTION_RELOAD.to_string(), vec!["<Control>r", "F5"]),
        (ACTION_SHORTCUTS.to_string(), vec!["<Control>question"]),
        (ACTION_QUIT.to_string(), vec!["<Control>q"]),
    ];
    table.extend((1..=PRESET_SLOTS).map(|slot| (preset_action(slot), vec![digit_accel(slot)])));
    table
}

/// Detailed `win.preset(<slot>)` action name.
fn preset_action(slot: u32) -> String {
    format!("{ACTION_PRESET}(uint32 {slot})")
}

fn digit_accel(slot: u32) -> &'static str {
    const DIGITS: [&str; 9] = [
        "<Control>1",
        "<Control>2",
        "<Control>3",
        "<Control>4",
        "<Control>5",
        "<Control>6",
        "<Control>7",
        "<Control>8",
        "<Control>9",
    ];
    DIGITS[(slot - 1) as usize]
}

/// Action name without its `win.` / `app.` group prefix.
fn short_name(detailed: &str) -> &str {
    detailed.split_once('.').map_or(detailed, |(_, name)| name)
}

/// Register a parameterless window action that emits `event`.
fn event_action(
    window: &adw::ApplicationWindow,
    detailed: &str,
    emit: &Emit<ActionEvent>,
    event: ActionEvent,
) -> gio::SimpleAction {
    let action = gio::SimpleAction::new(short_name(detailed), None);
    let emit = Emit::clone(emit);
    action.connect_activate(move |_, _| emit(event));
    window.add_action(&action);
    action
}

/// Keyboard shortcuts overview, driven by the registered accelerators.
fn shortcuts_dialog() -> adw::ShortcutsDialog {
    let dialog = adw::ShortcutsDialog::new();

    let presets = adw::ShortcutsSection::new(Some("Presets"));
    presets.add(adw::ShortcutsItem::from_action("Save", ACTION_SAVE));
    presets.add(adw::ShortcutsItem::from_action("Save As", ACTION_SAVE_AS));
    presets.add(adw::ShortcutsItem::new(
        "Switch to Preset 1–9",
        "<Control>1...9",
    ));
    presets.add(adw::ShortcutsItem::from_action(
        "Reload Configuration",
        ACTION_RELOAD,
    ));
    dialog.add(presets);

    let general = adw::ShortcutsSection::new(Some("General"));
    general.add(adw::ShortcutsItem::from_action(
        "Keyboard Shortcuts",
        ACTION_SHORTCUTS,
    ));
    general.add(adw::ShortcutsItem::from_action("Quit", ACTION_QUIT));
    dialog.add(general);

    dialog
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disconnected_disables_everything() {
        let e = Enablement::compute(false, true, true);
        assert_eq!(
            e,
            Enablement {
                save: false,
                save_as: false,
                revert: false,
                reload: false,
                preset: false,
            }
        );
    }

    #[test]
    fn save_needs_dirty_and_target() {
        assert!(Enablement::compute(true, true, true).save);
        assert!(!Enablement::compute(true, false, true).save);
        assert!(!Enablement::compute(true, true, false).save);
        assert!(Enablement::compute(true, false, true).save_as);
        assert!(Enablement::compute(true, true, false).revert);
    }

    // Accelerator strings themselves need `gtk::init` to parse, so only
    // the (display-free) detailed action names are checked here.
    #[test]
    fn accelerator_table_action_names_parse() {
        for (action, _) in accelerators() {
            assert!(
                gio::Action::parse_detailed_name(&action).is_ok(),
                "bad action name {action}"
            );
        }
        assert_eq!(preset_action(3), "win.preset(uint32 3)");
    }
}
