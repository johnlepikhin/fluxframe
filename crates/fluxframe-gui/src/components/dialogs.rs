//! Modal dialogs presented over the main window.
//!
//! Each builder presents its dialog right away, anchored to `parent`
//! (`AdwDialog::present` walks up to the toplevel, so any widget in the
//! window will do), and reports the operator's choice through an
//! [`Emit`] sink.

use std::cell::Cell;

use adw::prelude::*;
use gtk::glib;

use crate::components::Emit;

/// Ask before a destructive action (switching presets, reverting or
/// reloading with unsaved changes). Choosing `discard_label` fires
/// `on_discard` once.
pub(crate) fn confirm_discard(
    parent: &impl IsA<gtk::Widget>,
    heading: &str,
    body: &str,
    discard_label: &str,
    on_discard: Emit<()>,
) {
    let dialog = adw::AlertDialog::new(Some(heading), Some(body));
    dialog.add_response("cancel", "Cancel");
    dialog.add_response("discard", discard_label);
    dialog.set_response_appearance("discard", adw::ResponseAppearance::Destructive);
    dialog.set_default_response(Some("cancel"));
    dialog.set_close_response("cancel");

    // `response` fires once per presentation; guard against a repeat
    // anyway so the destructive action can never run twice.
    let fired = Cell::new(false);
    dialog.connect_response(None, move |_dlg, response| {
        if response == "discard" && !fired.replace(true) {
            on_discard(());
        }
    });

    dialog.present(Some(parent));
}

/// Ask the operator for a new preset name and fire `on_save` with the
/// trimmed name. Save stays disabled while the name is blank.
pub(crate) fn save_as(parent: &impl IsA<gtk::Widget>, on_save: Emit<String>) {
    let entry = gtk::Entry::builder()
        .placeholder_text("Preset name")
        .activates_default(true)
        .build();

    let dialog = adw::AlertDialog::new(
        Some("Save Preset As"),
        Some("Choose a name for the new preset."),
    );
    dialog.add_response("cancel", "Cancel");
    dialog.add_response("save", "Save");
    dialog.set_response_appearance("save", adw::ResponseAppearance::Suggested);
    dialog.set_response_enabled("save", false);
    dialog.set_default_response(Some("save"));
    dialog.set_close_response("cancel");
    dialog.set_extra_child(Some(&entry));

    entry.connect_changed(glib::clone!(
        #[weak]
        dialog,
        move |entry| {
            dialog.set_response_enabled("save", !entry.text().trim().is_empty());
        }
    ));

    // `activates_default` on the entry + the default response make
    // Enter commit the dialog (only while the response is enabled).
    dialog.connect_response(
        None,
        glib::clone!(
            #[weak]
            entry,
            move |_dlg, response| {
                if response == "save" {
                    let name = entry.text().trim().to_string();
                    if !name.is_empty() {
                        on_save(name);
                    }
                }
            }
        ),
    );

    dialog.present(Some(parent));
}

/// About dialog; `debug_info` carries the daemon session details.
pub(crate) fn about(parent: &impl IsA<gtk::Widget>, debug_info: &str) {
    let dialog = adw::AboutDialog::builder()
        .application_name("FluxFrame")
        .application_icon("camera-video-symbolic")
        .developer_name("FluxFrame contributors")
        .version(env!("CARGO_PKG_VERSION"))
        .website(env!("CARGO_PKG_REPOSITORY"))
        .comments("Live editor for the FluxFrame video effects daemon.")
        .license_type(gtk::License::Custom)
        .license(env!("CARGO_PKG_LICENSE"))
        .debug_info(debug_info)
        .build();
    dialog.present(Some(parent));
}
