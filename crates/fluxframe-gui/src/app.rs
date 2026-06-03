//! Root relm4 [`Component`] gluing the IPC worker to the libadwaita
//! window.
//!
//! Owns the application state (`AppState`), reacts to `WorkerOutput`
//! messages, and forwards user intent (preset switching, reload,
//! preview) back to the worker.

use std::path::PathBuf;
use std::process::Command as ProcessCommand;

use adw::prelude::*;
use fluxframe_core::protocol::Command;
use relm4::prelude::{Component, ComponentParts, ComponentSender};
use relm4::{Sender, WorkerController};

use crate::components::{preset_bar, status_page};
use crate::ipc::{IpcWorker, WorkerInput, WorkerOutput};
use crate::state::{AppState, ConnectionStatus};

/// Messages the AppModel handles internally.
#[derive(Debug)]
pub enum AppMsg {
    /// IPC worker reply. Wrapped so the AppModel can pattern-match
    /// without depending on the worker's `Output` type directly.
    Ipc(WorkerOutput),
    /// User clicked the preset DropDown — switch to `name`.
    SetPreset(String),
    /// User clicked the Reload button.
    Reload,
    /// User clicked "Open preview" — spawn an external viewer.
    OpenPreview,
    /// User clicked Retry on the disconnected status page.
    Retry,
}

/// Root component.
pub struct AppModel {
    state: AppState,
    /// IPC worker handle. `take()`-replaced on Retry so a fresh
    /// connection can be opened without restarting the GUI.
    worker: Option<WorkerController<IpcWorker>>,
    /// Cloned input sender — needed by widget callbacks that have to
    /// re-enter the model after `update()` has returned. relm4
    /// `Sender` is `Clone`, unlike the `ComponentSender` itself.
    input_sender: Sender<AppMsg>,
    /// Monotonic request counter used to correlate worker replies.
    next_tag: u64,
    /// Widgets that need to live longer than `view!` — the preset
    /// bar's children are mutated in `update_view`.
    preset_bar: preset_bar::PresetBar,
    /// Body container — we swap between a `StatusPage` (disconnected)
    /// and a placeholder pane (connected; chain editor lands in
    /// Step 5).
    body_container: gtk::Box,
}

impl Component for AppModel {
    type Init = PathBuf;
    type Input = AppMsg;
    type Output = ();
    type CommandOutput = ();
    type Root = adw::ApplicationWindow;
    type Widgets = ();

    fn init_root() -> Self::Root {
        adw::ApplicationWindow::builder()
            .title("FluxFrame")
            .default_width(720)
            .default_height(540)
            .build()
    }

    fn init(
        socket_path: Self::Init,
        root: Self::Root,
        sender: ComponentSender<Self>,
    ) -> ComponentParts<Self> {
        // Spawn the IPC worker.
        let worker = IpcWorker::builder()
            .detach_worker(socket_path.clone())
            .forward(sender.input_sender(), AppMsg::Ipc);

        // Build the header bar with closures that re-enter via the
        // input sender.
        let bar_sender = sender.input_sender().clone();
        let reload_sender = sender.input_sender().clone();
        let preview_sender = sender.input_sender().clone();
        let bar = preset_bar::build(preset_bar::PresetBarCallbacks {
            on_preset_change: Box::new(move |name| {
                let _ = bar_sender.send(AppMsg::SetPreset(name));
            }),
            on_reload: Box::new(move || {
                let _ = reload_sender.send(AppMsg::Reload);
            }),
            on_preview: Box::new(move || {
                let _ = preview_sender.send(AppMsg::OpenPreview);
            }),
        });

        let body = gtk::Box::new(gtk::Orientation::Vertical, 0);
        body.set_hexpand(true);
        body.set_vexpand(true);

        // Initial body — a Connecting status. Replaced on
        // `Connected` / `Disconnected`.
        body.append(&connecting_page("Talking to the FluxFrame daemon."));

        let outer = gtk::Box::new(gtk::Orientation::Vertical, 0);
        outer.append(&bar.root);
        outer.append(&body);
        root.set_content(Some(&outer));

        let state = AppState::new(socket_path);
        let input_sender = sender.input_sender().clone();
        let model = Self {
            state,
            worker: Some(worker),
            input_sender,
            next_tag: 0,
            preset_bar: bar,
            body_container: body,
        };
        ComponentParts { model, widgets: () }
    }

    fn update(&mut self, msg: Self::Input, _sender: ComponentSender<Self>, _root: &Self::Root) {
        match msg {
            AppMsg::Ipc(out) => self.on_ipc(out),
            AppMsg::SetPreset(name) => {
                self.send_command(Command::SetPreset { name });
            }
            AppMsg::Reload => {
                self.send_command(Command::Reload);
                // After a successful reload the cached active_preset
                // / config may have drifted; refetch the state.
                self.send_command(Command::CurrentPreset);
                self.send_command(Command::GetConfig { path: None });
            }
            AppMsg::OpenPreview => open_preview(),
            AppMsg::Retry => self.retry(),
        }
    }
}

impl AppModel {
    fn next_tag(&mut self) -> u64 {
        let tag = self.next_tag;
        self.next_tag = self.next_tag.wrapping_add(1);
        tag
    }

    fn send_command(&mut self, command: Command) {
        if self.worker.is_none() {
            tracing::warn!("send_command dropped — worker not connected");
            return;
        }
        let tag = self.next_tag();
        // Worker presence re-checked just above; safe to unwrap is justified.
        let worker = self.worker.as_ref().expect("worker presence checked above");
        let _ = worker.sender().send(WorkerInput::Send { tag, command });
    }

    fn on_ipc(&mut self, out: WorkerOutput) {
        match out {
            WorkerOutput::Connected(initial) => {
                tracing::info!(presets = initial.presets.len(), "daemon handshake complete");
                self.state.status = ConnectionStatus::Connected;
                self.state.presets = initial.presets;
                self.state.active_preset = Some(initial.active_preset.clone());
                self.state.inventory = initial.inventory;

                preset_bar::set_presets(
                    &self.preset_bar,
                    &self.state.presets,
                    self.state.active_preset.as_deref(),
                );
                if let Some(active) = self.state.active_preset.as_deref() {
                    preset_bar::set_active_preset(&self.preset_bar, active);
                }
                self.replace_body_with_connected_placeholder();
            }
            WorkerOutput::Disconnected { reason } => {
                tracing::warn!(reason = %reason, "daemon disconnected");
                self.state.status = ConnectionStatus::Disconnected(reason.clone());
                self.replace_body_with_status_page(&reason);
            }
            WorkerOutput::Reply { tag, response } => self.on_reply(tag, response),
        }
    }

    fn on_reply(&mut self, tag: u64, response: fluxframe_core::protocol::Response) {
        match response {
            fluxframe_core::protocol::Response::Ok { data } => {
                tracing::debug!(tag, ?data, "reply ok");
                // CurrentPreset replies after Reload — refresh title.
                if let Some(name) = data.as_str() {
                    self.state.active_preset = Some(name.to_string());
                    preset_bar::set_active_preset(&self.preset_bar, name);
                }
            }
            fluxframe_core::protocol::Response::Err { error, hint } => {
                tracing::warn!(tag, error = %error, hint = ?hint, "reply err");
            }
        }
    }

    fn replace_body_with_status_page(&self, reason: &str) {
        clear_children(&self.body_container);
        let socket = self.state.socket_path.clone();
        let retry_sender = self.input_sender.clone();
        let page = status_page::build(&socket, reason, move || {
            let _ = retry_sender.send(AppMsg::Retry);
        });
        self.body_container.append(&page);
    }

    fn replace_body_with_connected_placeholder(&self) {
        clear_children(&self.body_container);
        let placeholder = adw::StatusPage::builder()
            .icon_name("emblem-ok-symbolic")
            .title("Connected")
            .description("Chain editor lands in Stage 14 step 5. For now, switch presets above.")
            .hexpand(true)
            .vexpand(true)
            .build();
        self.body_container.append(&placeholder);
    }

    fn retry(&mut self) {
        tracing::info!("retry: re-creating IPC worker");
        // Drop the old worker (its background thread tears down) and
        // create a fresh one.
        self.worker = None;
        let socket_path = self.state.socket_path.clone();
        let worker = IpcWorker::builder()
            .detach_worker(socket_path)
            .forward(&self.input_sender, AppMsg::Ipc);
        self.worker = Some(worker);
        self.state.status = ConnectionStatus::Connecting;
        clear_children(&self.body_container);
        self.body_container
            .append(&connecting_page("Re-establishing the daemon socket."));
    }
}

/// Remove every child from a `gtk::Box`. relm4 has no built-in
/// "replace contents" for raw widgets.
fn clear_children(container: &gtk::Box) {
    while let Some(child) = container.first_child() {
        container.remove(&child);
    }
}

/// Build a "Connecting…" status page with the supplied description.
/// Icon and title are constant; only the description differs between
/// the initial wait and the post-Retry wait.
fn connecting_page(description: &str) -> adw::StatusPage {
    adw::StatusPage::builder()
        .icon_name("content-loading-symbolic")
        .title("Connecting…")
        .description(description)
        .hexpand(true)
        .vexpand(true)
        .build()
}

/// Spawn `gst-launch-1.0` on `/dev/video10` to show the daemon's
/// live output. Detached — closing the GUI does not kill the viewer
/// (use case: open preview once, keep tuning).
///
/// Failures are logged but not surfaced to the user via a dialog
/// because the operation is best-effort.
fn open_preview() {
    let mut cmd = ProcessCommand::new("gst-launch-1.0");
    cmd.args([
        "v4l2src",
        "device=/dev/video10",
        "!",
        "videoconvert",
        "!",
        "autovideosink",
    ]);
    match cmd.spawn() {
        Ok(child) => tracing::info!(pid = child.id(), "spawned gst-launch preview"),
        Err(e) => tracing::warn!(error = %e, "failed to spawn gst-launch"),
    }
}
