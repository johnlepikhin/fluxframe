//! Header bar widgets: title (active preset name), preset
//! drop-down, Reload, and "Open preview" launcher.
//!
//! Builds an [`adw::HeaderBar`] preloaded with the controls; signal
//! handlers forward user events into the AppModel via closures.

use adw::prelude::*;

/// Widgets the AppModel keeps around so it can update the displayed
/// state without rebuilding the header bar each tick.
pub struct PresetBar {
    /// The header bar itself; attach as `set_titlebar` on the window.
    pub root: adw::HeaderBar,
    /// Title label (live-updated with the active preset name).
    pub title: adw::WindowTitle,
    /// DropDown listing all known preset names.
    pub dropdown: gtk::DropDown,
}

/// Callbacks the header bar invokes for user-driven events.
///
/// Each field is an owned closure (not borrowed) because GTK signal
/// handlers must outlive the function that wires them.
#[allow(
    clippy::struct_field_names,
    reason = "`on_*` prefix mirrors the event-handler naming used at call sites"
)]
pub struct PresetBarCallbacks {
    /// Fired when the preset DropDown selection changes.
    pub on_preset_change: Box<dyn Fn(String) + 'static>,
    /// Fired when the Reload button is clicked.
    pub on_reload: Box<dyn Fn() + 'static>,
    /// Fired when the "Open preview" button is clicked.
    pub on_preview: Box<dyn Fn() + 'static>,
}

/// Construct the header bar. See [`PresetBarCallbacks`] for the user
/// events forwarded to the AppModel.
pub fn build(callbacks: PresetBarCallbacks) -> PresetBar {
    let PresetBarCallbacks {
        on_preset_change,
        on_reload,
        on_preview,
    } = callbacks;
    let title = adw::WindowTitle::new("FluxFrame", "");
    let header = adw::HeaderBar::builder().title_widget(&title).build();

    // Preset DropDown — fed an empty StringList up front; AppModel
    // refreshes it when `Connected` lands.
    let model = gtk::StringList::new(&[]);
    let dropdown = gtk::DropDown::builder()
        .model(&model)
        .tooltip_text("Switch active preset")
        .build();
    dropdown.connect_selected_notify(move |dd| {
        let model = dd
            .model()
            .and_then(|m| m.downcast::<gtk::StringList>().ok());
        if let Some(model) = model {
            let idx = dd.selected();
            if let Some(name) = model.string(idx) {
                on_preset_change(name.to_string());
            }
        }
    });
    header.pack_start(&dropdown);

    let reload = gtk::Button::from_icon_name("view-refresh-symbolic");
    reload.set_tooltip_text(Some("Reload daemon TOML"));
    reload.connect_clicked(move |_| on_reload());
    header.pack_end(&reload);

    let preview = gtk::Button::from_icon_name("video-display-symbolic");
    preview.set_tooltip_text(Some(
        "Open a live preview of the output (spawns gst-launch on /dev/video10)",
    ));
    preview.connect_clicked(move |_| on_preview());
    header.pack_end(&preview);

    PresetBar {
        root: header,
        title,
        dropdown,
    }
}

/// Replace the preset list shown in the dropdown.
///
/// `active` selects the matching item (if found) so the displayed
/// selection matches the daemon's `current_preset`.
pub fn set_presets(bar: &PresetBar, names: &[String], active: Option<&str>) {
    let model = gtk::StringList::new(&names.iter().map(String::as_str).collect::<Vec<_>>());
    bar.dropdown.set_model(Some(&model));
    if let Some(active) = active {
        if let Some(idx) = names.iter().position(|n| n == active) {
            bar.dropdown.set_selected(idx as u32);
        }
    }
}

/// Set the title bar to show the active preset name.
pub fn set_active_preset(bar: &PresetBar, name: &str) {
    bar.title.set_title("FluxFrame");
    bar.title.set_subtitle(name);
}
