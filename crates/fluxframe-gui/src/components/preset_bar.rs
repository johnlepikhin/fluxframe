//! Header bar widgets: title (active preset name), preset
//! drop-down, Save / Save as / Revert, and the main menu.
//!
//! Buttons and menu items are bound to the `win.*` / `app.*` actions
//! registered by [`crate::shortcuts`], so their sensitivity follows the
//! actions' enabled state. Only the preset drop-down reports back
//! through an [`Emit`] sink.

use adw::prelude::*;
use gtk::{gio, glib};

use crate::components::Emit;
use crate::shortcuts::{
    ACTION_ABOUT, ACTION_RELOAD, ACTION_REVERT, ACTION_SAVE, ACTION_SAVE_AS, ACTION_SHORTCUTS,
};

/// Widgets the AppModel keeps around so it can update the displayed
/// state without rebuilding the header bar each tick.
pub(crate) struct PresetBar {
    /// The header bar itself.
    pub(crate) root: adw::HeaderBar,
    /// Revert button; hidden by the narrow-window breakpoint, where the
    /// main menu entry takes over.
    pub(crate) revert: gtk::Button,
    /// Title widget; the subtitle carries the active preset name and
    /// the dirty marker.
    title: adw::WindowTitle,
    /// DropDown listing all known preset names.
    dropdown: gtk::DropDown,
    /// Handler for the user-driven selection change, blocked while the
    /// selection is changed programmatically.
    dropdown_handler: glib::SignalHandlerId,
    /// Save-to-current-preset button; tooltip names the target file.
    save: gtk::Button,
    /// Save-as-new-preset button; tooltip names the target file.
    save_as: gtk::Button,
}

/// Construct the header bar. `on_preset_change` receives the preset
/// name only when the user picks one in the drop-down, never for
/// [`PresetBar::set_presets`]-driven selection changes.
pub(crate) fn build(on_preset_change: Emit<String>) -> PresetBar {
    let title = adw::WindowTitle::new("FluxFrame", "");
    let header = adw::HeaderBar::builder().title_widget(&title).build();

    // Preset DropDown — fed an empty StringList up front; AppModel
    // refreshes it when `Connected` lands.
    let model = gtk::StringList::new(&[]);
    let dropdown = gtk::DropDown::builder()
        .model(&model)
        .tooltip_text("Switch Active Preset")
        .sensitive(false)
        .build();
    dropdown.update_property(&[gtk::accessible::Property::Label("Preset")]);
    let dropdown_handler = dropdown.connect_selected_notify(move |dd| {
        let model = dd
            .model()
            .and_then(|m| m.downcast::<gtk::StringList>().ok());
        if let Some(name) = model.and_then(|m| m.string(dd.selected())) {
            on_preset_change(name.to_string());
        }
    });
    header.pack_start(&dropdown);

    // Save / Save as / Revert group — packed right after the preset
    // dropdown, because the operator's mental flow is "pick preset →
    // edit → save". Linked so they read as one cluster.
    let save = action_button("document-save-symbolic", "Save", ACTION_SAVE);
    let save_as = action_button("document-save-as-symbolic", "Save As…", ACTION_SAVE_AS);
    let revert = action_button(
        "document-revert-symbolic",
        "Discard Unsaved Changes",
        ACTION_REVERT,
    );
    let save_group = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    save_group.add_css_class("linked");
    save_group.append(&save);
    save_group.append(&save_as);
    save_group.append(&revert);
    header.pack_start(&save_group);

    header.pack_end(&main_menu_button());

    PresetBar {
        root: header,
        revert,
        title,
        dropdown,
        dropdown_handler,
        save,
        save_as,
    }
}

/// Icon-only button bound to `action`, with a tooltip and a matching
/// accessible label.
fn action_button(icon: &str, label: &str, action: &str) -> gtk::Button {
    let button = gtk::Button::from_icon_name(icon);
    button.set_tooltip_text(Some(label));
    button.update_property(&[gtk::accessible::Property::Label(label)]);
    button.set_action_name(Some(action));
    button
}

/// Primary menu (`open-menu-symbolic`) with the rarely used actions.
fn main_menu_button() -> gtk::MenuButton {
    let menu = gio::Menu::new();
    let config_section = gio::Menu::new();
    config_section.append(Some("_Discard Unsaved Changes"), Some(ACTION_REVERT));
    config_section.append(Some("_Reload Configuration"), Some(ACTION_RELOAD));
    menu.append_section(None, &config_section);
    let app_section = gio::Menu::new();
    app_section.append(Some("_Keyboard Shortcuts"), Some(ACTION_SHORTCUTS));
    app_section.append(Some("_About FluxFrame"), Some(ACTION_ABOUT));
    menu.append_section(None, &app_section);

    let button = gtk::MenuButton::builder()
        .icon_name("open-menu-symbolic")
        .tooltip_text("Main Menu")
        .primary(true)
        .menu_model(&menu)
        .build();
    button.update_property(&[gtk::accessible::Property::Label("Main Menu")]);
    button
}

impl PresetBar {
    /// Refresh the Save / Save as tooltips, the title-bar dirty marker
    /// and the preset drop-down sensitivity. Button sensitivity itself
    /// follows the bound actions.
    pub(crate) fn set_state(
        &self,
        connected: bool,
        is_dirty: bool,
        config_path: Option<&std::path::Path>,
        active_preset: Option<&str>,
    ) {
        self.dropdown.set_sensitive(connected);

        let (save_tooltip, save_as_tooltip) = match config_path {
            Some(p) => (
                format!("Save to {}", p.display()),
                format!("Save as New Preset in {}", p.display()),
            ),
            None => (
                "Save Unavailable: the Daemon Has No Writable Config".to_string(),
                "Save As Unavailable: the Daemon Has No Writable Config".to_string(),
            ),
        };
        self.save.set_tooltip_text(Some(&save_tooltip));
        self.save_as.set_tooltip_text(Some(&save_as_tooltip));

        // Dirty marker: append " — Unsaved changes" to the preset name
        // so the indicator sits where the operator's eye lands when
        // scanning header → editor.
        let subtitle = match (is_dirty, active_preset) {
            (true, Some(name)) => format!("{name} — Unsaved changes"),
            (true, None) => "Unsaved changes".into(),
            (false, Some(name)) => name.to_string(),
            (false, None) => String::new(),
        };
        self.title.set_subtitle(&subtitle);
    }

    /// Replace the preset list shown in the dropdown and select
    /// `active` (if present), without reporting the change as a user
    /// pick.
    pub(crate) fn set_presets(&self, names: &[String], active: Option<&str>) {
        let model = gtk::StringList::new(&names.iter().map(String::as_str).collect::<Vec<_>>());
        self.dropdown.block_signal(&self.dropdown_handler);
        self.dropdown.set_model(Some(&model));
        if let Some(idx) = active.and_then(|a| names.iter().position(|n| n == a)) {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "preset list length is far below u32::MAX"
            )]
            self.dropdown.set_selected(idx as u32);
        }
        self.dropdown.unblock_signal(&self.dropdown_handler);
    }
}
