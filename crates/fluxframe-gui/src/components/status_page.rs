//! Empty / disconnected fallback page.
//!
//! Built around [`adw::StatusPage`] so a user who launches the GUI
//! without a running daemon (or after the connection drops) sees an
//! actionable message rather than a blank window.

use adw::prelude::*;

/// Build a self-contained "no daemon" status page with an embedded
/// Retry button.
///
/// `on_retry` fires when the user clicks Retry; the AppModel uses
/// this to re-spawn the IPC worker.
pub fn build(
    socket_path: &std::path::Path,
    reason: &str,
    on_retry: impl Fn() + 'static,
) -> gtk::Box {
    let container = gtk::Box::new(gtk::Orientation::Vertical, 0);
    container.set_hexpand(true);
    container.set_vexpand(true);

    let page = adw::StatusPage::builder()
        .icon_name("network-offline-symbolic")
        .title("Daemon unreachable")
        .description(format!(
            "Could not connect to {}.\n\n{reason}\n\nStart fluxframe with [control].enabled = true in its TOML, then click Retry.",
            socket_path.display()
        ))
        .hexpand(true)
        .vexpand(true)
        .build();

    let retry = gtk::Button::with_label("Retry");
    retry.add_css_class("suggested-action");
    retry.add_css_class("pill");
    retry.set_halign(gtk::Align::Center);
    retry.connect_clicked(move |_| on_retry());
    page.set_child(Some(&retry));

    container.append(&page);
    container
}
