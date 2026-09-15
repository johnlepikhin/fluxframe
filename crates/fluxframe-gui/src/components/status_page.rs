//! Connecting / disconnected fallback pages.
//!
//! Built around [`adw::StatusPage`] so a user who launches the GUI
//! without a running daemon (or after the connection drops) sees an
//! actionable message rather than a blank window.

use adw::prelude::*;

use crate::components::Emit;

/// Build a self-contained "no daemon" status page announcing the next
/// automatic reconnect, with an embedded "Retry Now" button.
///
/// `retry_in` is the delay until the scheduled attempt; `on_retry`
/// fires when the user clicks the button, and the AppModel then
/// connects immediately.
pub(crate) fn build(
    socket_path: &std::path::Path,
    reason: &str,
    retry_in: std::time::Duration,
    on_retry: Emit<()>,
) -> adw::StatusPage {
    let page = adw::StatusPage::builder()
        .icon_name("network-offline-symbolic")
        .title("Daemon Unreachable")
        .description(format!(
            "Socket: {}\n\n{reason}\n\nStart fluxframe with [control].enabled = true in its TOML. Reconnecting automatically in {} s.",
            socket_path.display(),
            retry_in.as_secs()
        ))
        .hexpand(true)
        .vexpand(true)
        .build();

    let retry = gtk::Button::with_label("Retry Now");
    retry.add_css_class("suggested-action");
    retry.add_css_class("pill");
    retry.set_halign(gtk::Align::Center);
    retry.connect_clicked(move |_| on_retry(()));
    page.set_child(Some(&retry));
    page
}

/// Build a "Connecting…" status page with an animated spinner and the
/// supplied description.
pub(crate) fn connecting(description: &str) -> adw::StatusPage {
    let page = adw::StatusPage::builder()
        .title("Connecting…")
        .description(description)
        .hexpand(true)
        .vexpand(true)
        .build();
    page.set_paintable(Some(&adw::SpinnerPaintable::new(Some(&page))));
    page
}
