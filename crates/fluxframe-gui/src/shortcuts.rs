//! Keyboard shortcuts wired onto the main window.
//!
//! - `Ctrl+R` / `F5` — reload daemon TOML + refetch state.
//! - `Ctrl+Q` — quit.
//! - `Ctrl+1` .. `Ctrl+9` — switch to preset N (1-indexed in the
//!   currently-loaded list).
//!
//! Implemented via [`gtk::ShortcutController`] so the shortcuts are
//! captured by the window regardless of which widget has focus.

use gtk::glib;
use gtk::prelude::*;
use relm4::Sender;

use crate::app::AppMsg;

/// Attach the application shortcut controller to `window`.
///
/// `sender` is the AppModel input channel; shortcuts dispatch
/// [`AppMsg`] values onto it. The presets list is resolved at
/// trigger time inside the AppModel
/// ([`AppMsg::SetPresetByIndex`]), so the shortcut layer does not
/// need to track preset state.
pub(crate) fn install(window: &adw::ApplicationWindow, sender: &Sender<AppMsg>) {
    let controller = gtk::ShortcutController::new();
    controller.set_scope(gtk::ShortcutScope::Global);

    // Ctrl+R — reload.
    {
        let sender = sender.clone();
        controller.add_shortcut(callback_shortcut("<Control>r", move || {
            let _ = sender.send(AppMsg::Reload);
        }));
    }
    // F5 — also reload (gnome-shell convention).
    {
        let sender = sender.clone();
        controller.add_shortcut(callback_shortcut("F5", move || {
            let _ = sender.send(AppMsg::Reload);
        }));
    }

    // Ctrl+Q — quit.
    let window_for_quit = window.clone();
    controller.add_shortcut(callback_shortcut("<Control>q", move || {
        window_for_quit.close();
    }));

    // Ctrl+1 .. Ctrl+9 — switch preset by index.
    for digit in 1..=9 {
        let sender = sender.clone();
        let slot = std::num::NonZeroUsize::new(digit).expect("digit 1..=9 is non-zero");
        controller.add_shortcut(callback_shortcut(&format!("<Control>{digit}"), move || {
            let _ = sender.send(AppMsg::SetPresetByIndex { slot });
        }));
    }

    window.add_controller(controller);
}

/// Build a shortcut whose action is an arbitrary closure.
///
/// # Panics
///
/// Panics if `trigger` is not a valid GTK shortcut string. Callers
/// pass only static literals; the panic would surface at app startup,
/// which is the correct failure mode for a programming error.
fn callback_shortcut(trigger: &str, f: impl Fn() + 'static) -> gtk::Shortcut {
    let trigger = gtk::ShortcutTrigger::parse_string(trigger)
        .unwrap_or_else(|| panic!("static shortcut trigger '{trigger}' must parse"));
    let action = gtk::CallbackAction::new(move |_widget, _args| {
        f();
        glib::Propagation::Stop
    });
    gtk::Shortcut::builder()
        .trigger(&trigger)
        .action(&action)
        .build()
}
