//! Header bar widgets: title (active preset name), preset
//! drop-down, Save / Save as / Revert, Reload, and "Open preview"
//! launcher.
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
    /// Save-to-current-preset button. Sensitive only when the
    /// active config has diverged from the daemon baseline AND the
    /// daemon reported a writable [`ConfigPath`].
    pub save: gtk::Button,
    /// Save-as-new-preset button. Sensitive whenever the daemon has
    /// a writable config path.
    pub save_as: gtk::Button,
    /// Revert button — re-syncs the in-memory state from the daemon
    /// baseline. Sensitive only when dirty.
    pub revert: gtk::Button,
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
    /// Fired when the Save button is clicked.
    pub on_save: Box<dyn Fn() + 'static>,
    /// Fired when the "Save as…" button is clicked.
    pub on_save_as: Box<dyn Fn() + 'static>,
    /// Fired when the Revert button is clicked.
    pub on_revert: Box<dyn Fn() + 'static>,
}

/// Construct the header bar. See [`PresetBarCallbacks`] for the user
/// events forwarded to the AppModel.
pub fn build(callbacks: PresetBarCallbacks) -> PresetBar {
    let PresetBarCallbacks {
        on_preset_change,
        on_reload,
        on_preview,
        on_save,
        on_save_as,
        on_revert,
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

    // Save / Save as / Revert group — packed at the start, right
    // after the preset dropdown, because the operator's mental flow
    // is "pick preset → edit → save". The buttons share a linked
    // CSS class so they read as one cluster.
    let save = gtk::Button::from_icon_name("document-save-symbolic");
    save.set_tooltip_text(Some("Save current settings to active preset"));
    save.set_sensitive(false);
    save.connect_clicked(move |_| on_save());

    let save_as = gtk::Button::from_icon_name("document-save-as-symbolic");
    save_as.set_tooltip_text(Some("Save current settings as a new preset…"));
    save_as.set_sensitive(false);
    save_as.connect_clicked(move |_| on_save_as());

    let revert = gtk::Button::from_icon_name("document-revert-symbolic");
    revert.set_tooltip_text(Some("Discard unsaved changes and reload from daemon"));
    revert.set_sensitive(false);
    revert.connect_clicked(move |_| on_revert());

    let save_group = gtk::Box::new(gtk::Orientation::Horizontal, 0);
    save_group.add_css_class("linked");
    save_group.append(&save);
    save_group.append(&save_as);
    save_group.append(&revert);
    header.pack_start(&save_group);

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
        save,
        save_as,
        revert,
    }
}

/// Refresh the Save / Save as / Revert button sensitivity and the
/// title-bar dirty marker. Called by the AppModel whenever
/// [`is_dirty`](crate::state::AppState::is_dirty) or the daemon's
/// writable config path may have changed (handshake, after a
/// `SetParam` reply, after Save, after Revert).
pub fn set_dirty_state(
    bar: &PresetBar,
    is_dirty: bool,
    config_path: Option<&std::path::Path>,
    active_preset: Option<&str>,
) {
    let has_target = config_path.is_some();
    bar.save.set_sensitive(is_dirty && has_target);
    bar.save_as.set_sensitive(has_target);
    bar.revert.set_sensitive(is_dirty);

    let tooltip = match config_path {
        Some(p) => format!(
            "Save current settings to active preset (target: {})",
            p.display()
        ),
        None => "Daemon was started without a writable config path; Save disabled".into(),
    };
    bar.save.set_tooltip_text(Some(&tooltip));
    bar.save_as.set_tooltip_text(Some(&match config_path {
        Some(p) => format!(
            "Save current settings as a new preset (target: {})",
            p.display()
        ),
        None => "Daemon was started without a writable config path; Save as disabled".into(),
    }));

    // Title-bar dirty marker: prepend "● " to the preset name when
    // unsaved edits are pending. Keeps the indicator close to where
    // the operator's eye lands when scanning header → editor.
    let subtitle = match (is_dirty, active_preset) {
        (true, Some(name)) => format!("● {name}"),
        (true, None) => "● (no preset)".into(),
        (false, Some(name)) => name.to_string(),
        (false, None) => String::new(),
    };
    bar.title.set_subtitle(&subtitle);
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
