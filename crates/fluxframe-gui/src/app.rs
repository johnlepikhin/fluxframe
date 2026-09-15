//! Root relm4 [`Component`] gluing the IPC worker to the libadwaita
//! window.
//!
//! Owns the application state (`AppState`), maps component events and
//! worker output into daemon requests, and applies the UI effects that
//! [`crate::reply::apply_reply`] derives from each reply.

use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use adw::prelude::*;
use fluxframe_core::SubchainKind;
use fluxframe_core::protocol::Command;
use gtk::glib;
use relm4::prelude::{Component, ComponentParts, ComponentSender};
use relm4::{Sender, WorkerController};

use crate::components::chain_page::{self, ChainEvent};
use crate::components::param_row::ParamChange;
use crate::components::{Emit, dialogs, preset_bar, preview, status_page};
use crate::debounce::Debouncer;
use crate::ipc::{IpcWorker, WorkerInput, WorkerOutput};
use crate::reply::{self, NoticeKind, PendingKind, PendingRequests, ReplyEffects};
use crate::shortcuts::{ActionEvent, Enablement, WindowActions};
use crate::state::{AppState, ChainEdit, ConfigSync, ConnectionStatus, LostSession, PresetSwitch};

/// Dwell time for error toasts. Long enough to read, short enough to
/// avoid stacking when a slider drags through several rejections.
const ERROR_TOAST_TIMEOUT_SECS: u32 = 5;

/// Dwell time for confirmation toasts (successful Save / Save as).
const CONFIRMATION_TOAST_TIMEOUT_SECS: u32 = 2;

/// Toast shown when a reconnect finds the operator's unsaved edits gone.
const RECONNECT_EDITS_LOST: &str =
    "Reconnected. Unsaved changes are gone: the daemon no longer has them.";

/// Smallest window size the layout is designed for (GNOME HIG mobile
/// minimum).
const MIN_WINDOW_WIDTH: i32 = 360;
/// See [`MIN_WINDOW_WIDTH`].
const MIN_WINDOW_HEIGHT: i32 = 294;

/// Below this width the header bar sheds the Revert button (the main
/// menu still offers it).
const NARROW_BREAKPOINT: &str = "max-width: 550sp";

/// Startup parameters of the root component.
#[derive(Debug)]
pub struct AppInit {
    /// Daemon control socket.
    pub socket_path: PathBuf,
    /// Whether `gstreamer::init()` succeeded; the preview stays inert
    /// otherwise.
    pub gst_ready: bool,
}

/// Messages the AppModel handles.
#[derive(Debug)]
pub enum AppMsg {
    /// IPC worker output.
    Ipc(WorkerOutput),
    /// User picked a preset in the drop-down or via `Ctrl+<digit>`.
    PresetPicked(String),
    /// The operator confirmed discarding edits to switch to the preset.
    SetPresetConfirmed(String),
    /// Chain page event.
    Chain(ChainEvent),
    /// Window action event.
    Action(ActionEvent),
    /// The operator confirmed the "Save As" dialog with a non-empty name.
    SaveAs(String),
    /// The operator confirmed discarding edits on Revert.
    RevertConfirmed,
    /// The operator confirmed discarding edits on Reload.
    ReloadConfirmed,
    /// User clicked "Retry Now" on the disconnected status page.
    Retry,
    /// A scheduled reconnect attempt is due. Carries the generation it
    /// was scheduled under; a stale one is ignored.
    ReconnectTick(u64),
}

/// Root component.
pub struct AppModel {
    state: AppState,
    /// IPC worker handle. Replaced on Retry so a fresh connection can
    /// be opened without restarting the GUI.
    worker: Option<WorkerController<IpcWorker>>,
    /// In-flight requests awaiting a reply.
    pending: PendingRequests,
    /// Cloned input sender for callbacks that re-enter the model after
    /// `update()` has returned.
    input_sender: Sender<AppMsg>,
    /// Event sink handed to every chain page build.
    chain_emit: Emit<ChainEvent>,
    /// Header bar widgets updated from the model.
    preset_bar: preset_bar::PresetBar,
    /// State-dependent window actions (Save, Revert, Reload, …).
    actions: WindowActions,
    /// Body container — we swap between a `StatusPage` (connecting /
    /// disconnected) and the chain editor (connected).
    body_container: gtk::Box,
    /// Chain page currently shown in `body_container`, if any. Kept
    /// to carry its view state across rebuilds; `None` while another
    /// page (status / connecting) occupies the body.
    chain_page: Option<chain_page::ChainPage>,
    /// Embedded live preview pane. Owns the preview GStreamer pipeline;
    /// replacing it (or dropping the model) tears that pipeline down.
    preview: preview::Preview,
    /// What `preview` shows; the pane is rebuilt only when this changes.
    preview_target: preview::PreviewTarget,
    /// Split holding the preview (start child) and the body.
    paned: gtk::Paned,
    /// Whether `gstreamer::init()` succeeded, for preview rebuilds.
    gst_ready: bool,
    /// Consecutive failed connection attempts; drives the backoff.
    reconnect_attempt: u32,
    /// Generation of the pending reconnect tick. Bumping it cancels a
    /// scheduled tick without removing its (possibly already fired)
    /// glib source.
    reconnect_generation: u64,
    /// The session shown before the link dropped, compared against the
    /// next handshake to tell whether unsaved edits survived.
    lost_session: Option<LostSession>,
    /// Per-parameter debounce queue shared by all param rows.
    debouncer: Debouncer,
    /// Toast overlay wrapping the window body.
    toast_overlay: adw::ToastOverlay,
}

/// Emitter forwarding a component's events into the AppModel as `wrap(event)`.
fn forward<E: 'static>(sender: &Sender<AppMsg>, wrap: impl Fn(E) -> AppMsg + 'static) -> Emit<E> {
    let sender = sender.clone();
    Rc::new(move |event| {
        let _ = sender.send(wrap(event));
    })
}

/// Emitter sending a fixed message (cloned per call) into the AppModel.
fn forward_unit(sender: &Sender<AppMsg>, msg: fn() -> AppMsg) -> Emit<()> {
    forward(sender, move |()| msg())
}

/// Spawn an IPC worker for `socket_path` whose output lands in the
/// AppModel.
fn spawn_worker(socket_path: PathBuf, sender: &Sender<AppMsg>) -> WorkerController<IpcWorker> {
    IpcWorker::builder()
        .detach_worker(socket_path)
        .forward(sender, AppMsg::Ipc)
}

impl Component for AppModel {
    type Init = AppInit;
    type Input = AppMsg;
    type Output = ();
    type CommandOutput = ();
    type Root = adw::ApplicationWindow;
    type Widgets = ();

    fn init_root() -> Self::Root {
        // Geometry is applied in `init()`, which also needs it for the
        // `gtk::Paned` split; the window is presented only afterwards.
        adw::ApplicationWindow::builder()
            .title("FluxFrame")
            .width_request(MIN_WINDOW_WIDTH)
            .height_request(MIN_WINDOW_HEIGHT)
            .build()
    }

    fn init(
        AppInit {
            socket_path,
            gst_ready,
        }: Self::Init,
        root: Self::Root,
        sender: ComponentSender<Self>,
    ) -> ComponentParts<Self> {
        let geometry = crate::persistence::load();
        root.set_default_size(geometry.width, geometry.height);
        if geometry.maximized {
            root.maximize();
        }

        let input_sender = sender.input_sender().clone();

        let worker = spawn_worker(socket_path.clone(), &input_sender);

        let actions = crate::shortcuts::install(
            &relm4::main_adw_application(),
            &root,
            &forward(&input_sender, AppMsg::Action),
        );
        let bar = preset_bar::build(forward(&input_sender, AppMsg::PresetPicked));

        let narrow = adw::Breakpoint::new(
            adw::BreakpointCondition::parse(NARROW_BREAKPOINT)
                .expect("static breakpoint condition parses"),
        );
        narrow.add_setter(&bar.revert, "visible", Some(&false.to_value()));
        root.add_breakpoint(narrow);

        let body = gtk::Box::new(gtk::Orientation::Vertical, 0);
        body.set_hexpand(true);
        body.set_vexpand(true);

        // Initial body — a Connecting status. Replaced on
        // `Connected` / `Disconnected`.
        body.append(&status_page::connecting("Talking to the FluxFrame daemon."));

        // The daemon's output is unknown until the handshake reports it.
        let preview_target = preview::PreviewTarget::Unavailable(preview::WAITING_FOR_DAEMON);
        let preview = preview::build(&preview_target, gst_ready);

        // `gtk::Paned` owns the split between preview (top) and the
        // chain editor (bottom). The Paned takes the responsibility
        // for the vertical allocation away from the outer Box, which
        // means the Picture inside `preview.root` can use
        // `content_fit = Contain` + `can_shrink = true` without
        // fighting `gtk::Picture`'s aspect-ratio-derived natural
        // size — the Paned tells the Picture exactly how tall it gets,
        // and the Picture letterboxes inside that allocation.
        //
        // `resize_*_child = true` makes the divider track the parent
        // size proportionally on resize; `shrink_*_child = false`
        // stops the user from collapsing either half to zero by
        // dragging the handle to an extreme.
        let paned = gtk::Paned::builder()
            .orientation(gtk::Orientation::Vertical)
            .vexpand(true)
            .hexpand(true)
            .resize_start_child(true)
            .resize_end_child(true)
            .shrink_start_child(false)
            .shrink_end_child(false)
            .start_child(&preview.root)
            .end_child(&body)
            .position(geometry.resolved_preview_split())
            .build();

        // Persist window geometry + Paned split on close. `close-request`
        // fires before teardown so the window and paned are still
        // queryable.
        let paned_for_close = paned.clone();
        root.connect_close_request(move |w| {
            let state = crate::persistence::WindowState {
                width: w.default_width(),
                height: w.default_height(),
                maximized: w.is_maximized(),
                preview_split: Some(paned_for_close.position()),
            };
            crate::persistence::save(&state);
            glib::Propagation::Proceed
        });

        // Toasts fly over the editor without disturbing the layout.
        let toast_overlay = adw::ToastOverlay::new();
        toast_overlay.set_child(Some(&paned));

        let toolbar_view = adw::ToolbarView::new();
        toolbar_view.add_top_bar(&bar.root);
        toolbar_view.set_content(Some(&toast_overlay));
        root.set_content(Some(&toolbar_view));

        let model = Self {
            state: AppState::new(socket_path),
            worker: Some(worker),
            pending: PendingRequests::default(),
            chain_emit: forward(&input_sender, AppMsg::Chain),
            input_sender,
            preset_bar: bar,
            actions,
            body_container: body,
            chain_page: None,
            preview,
            preview_target,
            paned,
            gst_ready,
            reconnect_attempt: 0,
            reconnect_generation: 0,
            lost_session: None,
            debouncer: Debouncer::new(),
            toast_overlay,
        };
        model.refresh_header();
        ComponentParts { model, widgets: () }
    }

    fn update(&mut self, msg: Self::Input, _sender: ComponentSender<Self>, _root: &Self::Root) {
        match msg {
            AppMsg::Ipc(out) => self.on_ipc(out),
            AppMsg::PresetPicked(name) => self.on_preset_picked(name),
            AppMsg::SetPresetConfirmed(name) => self.switch_preset(name),
            AppMsg::Chain(ChainEvent::Edit { section, edit }) => self.edit_chain(section, edit),
            AppMsg::Chain(ChainEvent::Param(change)) => self.send_set_param(change),
            AppMsg::Chain(ChainEvent::SetEnabled {
                section,
                effect,
                enabled,
            }) => self.send_kind(
                Command::SetEnabled {
                    section,
                    effect: effect.clone(),
                    enabled,
                },
                PendingKind::SetEnabled {
                    section,
                    effect,
                    enabled,
                },
            ),
            AppMsg::Action(action) => self.on_action(action),
            AppMsg::SaveAs(name) => {
                self.send_kind(
                    Command::SavePresetAs { name: name.clone() },
                    PendingKind::SaveAs { new_name: name },
                );
            }
            AppMsg::RevertConfirmed => self.revert(),
            AppMsg::ReloadConfirmed => self.reload(),
            AppMsg::Retry => self.reconnect(),
            AppMsg::ReconnectTick(generation) => {
                if generation == self.reconnect_generation
                    && self.state.status == ConnectionStatus::Disconnected
                {
                    self.reconnect();
                }
            }
        }
    }
}

impl AppModel {
    fn on_action(&mut self, action: ActionEvent) {
        match action {
            ActionEvent::Save => self.send_kind(Command::SavePreset, PendingKind::Save),
            ActionEvent::SaveAs => {
                dialogs::save_as(
                    &self.preset_bar.root,
                    forward(&self.input_sender, AppMsg::SaveAs),
                );
            }
            ActionEvent::Revert => dialogs::confirm_discard(
                &self.preset_bar.root,
                "Discard Changes?",
                "This will restore the preset as last saved, discarding all unsaved edits.",
                "Discard",
                forward_unit(&self.input_sender, || AppMsg::RevertConfirmed),
            ),
            ActionEvent::Reload => {
                if self.state.is_dirty() {
                    dialogs::confirm_discard(
                        &self.preset_bar.root,
                        "Reload Configuration?",
                        "You have unsaved changes in the current preset. Reloading the configuration file will discard them.",
                        "Discard and Reload",
                        forward_unit(&self.input_sender, || AppMsg::ReloadConfirmed),
                    );
                } else {
                    self.reload();
                }
            }
            ActionEvent::About => dialogs::about(&self.preset_bar.root, &self.debug_info()),
            ActionEvent::PresetSlot(slot) => {
                // 1-indexed slot → list position; `NonZeroUsize` makes
                // the subtraction safe.
                if let Some(name) = self.state.presets.get(slot.get() - 1).cloned() {
                    self.on_preset_picked(name);
                } else {
                    tracing::debug!(slot = slot.get(), "no preset at slot");
                }
            }
        }
    }

    fn on_preset_picked(&mut self, name: String) {
        match self.state.preset_switch(&name) {
            PresetSwitch::AlreadyActive => {}
            PresetSwitch::Proceed => self.switch_preset(name),
            PresetSwitch::NeedsConfirmation => {
                // Keep the drop-down on the active preset until the
                // operator confirms.
                self.refresh_presets();
                let target = name;
                dialogs::confirm_discard(
                    &self.preset_bar.root,
                    "Switch Preset?",
                    "You have unsaved changes in the current preset. Switching will discard them.",
                    "Discard and Switch",
                    forward(&self.input_sender, move |()| {
                        AppMsg::SetPresetConfirmed(target.clone())
                    }),
                );
            }
        }
    }

    /// Send a command to the worker, registering its reply kind.
    fn send_kind(&mut self, command: Command, kind: PendingKind) {
        let Some(worker) = self.worker.as_ref() else {
            tracing::warn!("send_kind dropped — worker not connected");
            return;
        };
        let tag = self.pending.register(kind);
        let _ = worker.sender().send(WorkerInput::Send { tag, command });
    }

    /// Activate preset `name` on the daemon. The resulting state is
    /// refetched once the daemon accepts the switch.
    fn switch_preset(&mut self, name: String) {
        // Pending debounced sends belong to the preset being left.
        self.debouncer.cancel_all();
        self.send_kind(Command::SetPreset { name }, PendingKind::SetPreset);
    }

    /// Drop unsaved edits by re-activating the active preset: the
    /// daemon rebuilds it from its persisted copy.
    fn revert(&mut self) {
        let Some(name) = self.state.active_preset.clone() else {
            tracing::warn!("revert requested without an active preset");
            return;
        };
        // Pending debounced sends would re-apply discarded edits.
        self.debouncer.cancel_all();
        self.send_kind(Command::SetPreset { name }, PendingKind::Revert);
    }

    /// Re-read the daemon's config file; follow-up refetches are
    /// issued once it succeeds.
    fn reload(&mut self) {
        self.debouncer.cancel_all();
        self.send_kind(Command::Reload, PendingKind::Reload);
    }

    /// Refetch the active preset name and config as a clean baseline.
    fn refetch_active(&mut self) {
        self.send_kind(Command::CurrentPreset, PendingKind::CurrentPreset);
        self.send_kind(
            Command::GetConfig { path: None },
            PendingKind::GetConfig(ConfigSync::ResyncBaseline),
        );
    }

    /// Apply `edit` to the chain of `section`, dispatch the result as a
    /// `Command::SetChain`, then refetch the config so widgets pick up
    /// the new chain shape (and any defaulting the daemon applied).
    fn edit_chain(&mut self, section: SubchainKind, edit: ChainEdit) {
        let mut chain = self.state.chain_for(section);
        if !edit.apply(&mut chain) {
            return;
        }
        // Pending debounced sends target rows the new chain may not have.
        self.debouncer.cancel_all();
        self.send_kind(Command::SetChain { section, chain }, PendingKind::Other);
        // The daemon answers with its in-memory preset, which still
        // carries any unsaved edits — so the baseline must stay put.
        self.send_kind(
            Command::GetConfig { path: None },
            PendingKind::GetConfig(ConfigSync::KeepBaseline),
        );
    }

    /// Dispatch a parameter change as a `Command::Set`.
    fn send_set_param(&mut self, change: ParamChange) {
        let ParamChange { path, value } = change;
        let command = Command::Set {
            path: path.to_string(),
            value: value.clone(),
        };
        self.send_kind(command, PendingKind::SetParam { path, value });
    }

    fn on_ipc(&mut self, out: WorkerOutput) {
        match out {
            WorkerOutput::Connected(initial) => {
                let initial = *initial;
                tracing::info!(presets = initial.presets.len(), "daemon handshake complete");
                self.state.status = ConnectionStatus::Connected;
                self.reconnect_attempt = 0;
                self.reconnect_generation += 1;
                self.state.presets = initial.presets;
                self.state.inventory = initial.inventory;
                self.state.config_path = initial.config_path;
                self.state.output = initial.output;
                let edits_lost = self.state.apply_handshake(
                    initial.active_preset,
                    initial.active_config,
                    self.lost_session.take().as_ref(),
                );
                self.show_preview(preview::target_for(self.state.output.as_ref()));
                self.refresh_presets();
                self.refresh_header();
                self.rebuild_chain_page();
                if edits_lost {
                    self.show_notice(&reply::Notice {
                        kind: NoticeKind::Error,
                        message: RECONNECT_EDITS_LOST.to_string(),
                    });
                }
            }
            WorkerOutput::Disconnected { reason } => {
                tracing::warn!(reason = %reason, "daemon disconnected");
                // Remember what the operator saw only when a live session
                // drops; a failed reconnect attempt must not overwrite it.
                if self.state.is_connected() {
                    self.lost_session = Some(self.state.lost_session());
                }
                self.state.status = ConnectionStatus::Disconnected;
                // Replies to these requests will never arrive.
                self.pending.clear();
                self.debouncer.cancel_all();
                self.refresh_header();
                let retry_in = self.schedule_reconnect();
                let page = status_page::build(
                    &self.state.socket_path,
                    &reason,
                    retry_in,
                    forward_unit(&self.input_sender, || AppMsg::Retry),
                );
                self.set_body(&page);
            }
            WorkerOutput::Reply { tag, response } => {
                let kind = self.pending.take(tag);
                let effects = reply::apply_reply(&mut self.state, kind, response);
                self.apply_effects(effects);
            }
        }
    }

    /// Perform the UI work a reply requires.
    fn apply_effects(&mut self, effects: ReplyEffects) {
        let ReplyEffects {
            rebuild_chain,
            reset_param,
            sync_toggle,
            refetch_config_quiet,
            presets_changed,
            refetch_active,
            refetch_presets,
            notice,
        } = effects;
        if refetch_presets {
            self.send_kind(Command::ListPresets, PendingKind::ListPresets);
        }
        if refetch_config_quiet {
            self.send_kind(
                Command::GetConfig { path: None },
                PendingKind::GetConfigQuiet,
            );
        }
        if refetch_active {
            self.refetch_active();
        }
        if presets_changed {
            self.refresh_presets();
        }
        self.refresh_header();
        if rebuild_chain {
            self.rebuild_chain_page();
        }
        if let (Some(path), Some(page)) = (reset_param, self.chain_page.as_ref()) {
            page.reset_param(&path, self.state.config_field(&path));
        }
        if let (Some((section, effect)), Some(page)) = (sync_toggle, self.chain_page.as_ref()) {
            page.set_toggle(
                section,
                &effect,
                self.state.effect_enabled(section, &effect),
            );
        }
        if let Some(notice) = notice {
            self.show_notice(&notice);
        }
    }

    /// Show `notice` as a toast over the window body.
    fn show_notice(&self, notice: &reply::Notice) {
        let timeout = match notice.kind {
            NoticeKind::Error => ERROR_TOAST_TIMEOUT_SECS,
            NoticeKind::Confirmation => CONFIRMATION_TOAST_TIMEOUT_SECS,
        };
        let toast = adw::Toast::builder()
            .title(&notice.message)
            .timeout(timeout)
            .build();
        self.toast_overlay.add_toast(toast);
    }

    /// Re-populate the preset drop-down from the state.
    fn refresh_presets(&self) {
        self.preset_bar
            .set_presets(&self.state.presets, self.state.active_preset.as_deref());
    }

    /// Push connection state, dirty flag and writable config path into
    /// the header bar and the window actions.
    fn refresh_header(&self) {
        let connected = self.state.is_connected();
        let dirty = self.state.is_dirty();
        self.preset_bar.set_state(
            connected,
            dirty,
            self.state.config_path.as_deref(),
            self.state.active_preset.as_deref(),
        );
        self.actions.apply(Enablement::compute(
            connected,
            dirty,
            self.state.config_path.is_some(),
        ));
    }

    /// Daemon session details for the About dialog.
    fn debug_info(&self) -> String {
        let features = &self.state.inventory.build_features;
        let config_path = self
            .state
            .config_path
            .as_deref()
            .map_or_else(|| "(none)".to_string(), |p| p.display().to_string());
        format!(
            "Socket: {}\nConnection: {:?}\nActive preset: {}\nConfig path: {config_path}\nDaemon build features: {}\n",
            self.state.socket_path.display(),
            self.state.status,
            self.state.active_preset.as_deref().unwrap_or("(none)"),
            if features.is_empty() {
                "(none)".to_string()
            } else {
                features.join(", ")
            },
        )
    }

    /// Drop the current body content and re-render the chain editor
    /// from the active state, preserving expanded rows, scroll and
    /// focus.
    fn rebuild_chain_page(&mut self) {
        tracing::debug!(
            preset = self.state.active_preset.as_deref().unwrap_or("(none)"),
            "rebuilt chain page"
        );
        // Pending debounced sends target widgets of the old page.
        self.debouncer.cancel_all();
        let view_state = self
            .chain_page
            .as_ref()
            .map(chain_page::ChainPage::view_state);
        let page = chain_page::build(&self.state, &self.debouncer, &self.chain_emit);
        self.set_body(&page.root);
        if let Some(view_state) = &view_state {
            page.restore(view_state);
        }
        self.chain_page = Some(page);
    }

    /// Replace the body content with `widget`, forgetting the chain
    /// page (callers showing a chain page set it again).
    fn set_body(&mut self, widget: &impl IsA<gtk::Widget>) {
        self.chain_page = None;
        while let Some(child) = self.body_container.first_child() {
            self.body_container.remove(&child);
        }
        self.body_container.append(widget);
    }

    /// Swap the preview pane when the daemon's output calls for a
    /// different one.
    fn show_preview(&mut self, target: preview::PreviewTarget) {
        if target == self.preview_target {
            return;
        }
        let preview = preview::build(&target, self.gst_ready);
        self.paned.set_start_child(Some(&preview.root));
        // Replacing the field drops the old pane and stops its pipeline.
        self.preview = preview;
        self.preview_target = target;
    }

    /// Schedule the next automatic reconnect with backoff; returns its
    /// delay for the status page.
    fn schedule_reconnect(&mut self) -> Duration {
        let delay = crate::reconnect::delay(self.reconnect_attempt);
        self.reconnect_attempt = self.reconnect_attempt.saturating_add(1);
        self.reconnect_generation += 1;
        let generation = self.reconnect_generation;
        let sender = self.input_sender.clone();
        // The source is never removed: a newer generation turns this
        // tick into a no-op instead, because removing a source that has
        // already fired panics.
        let _ = glib::timeout_add_local_once(delay, move || {
            let _ = sender.send(AppMsg::ReconnectTick(generation));
        });
        delay
    }

    /// Connect again right away, superseding any scheduled attempt.
    fn reconnect(&mut self) {
        tracing::info!(
            attempt = self.reconnect_attempt,
            "reconnecting: re-creating IPC worker"
        );
        self.reconnect_generation += 1;
        // Drop the old worker (its background thread tears down) and
        // create a fresh one; replies owed by the old one are gone.
        self.worker = None;
        self.pending.clear();
        self.worker = Some(spawn_worker(
            self.state.socket_path.clone(),
            &self.input_sender,
        ));
        self.state.status = ConnectionStatus::Connecting;
        self.refresh_header();
        self.set_body(&status_page::connecting(
            "Re-establishing the daemon socket.",
        ));
    }
}
