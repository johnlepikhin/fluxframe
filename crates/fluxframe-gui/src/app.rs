//! Root relm4 [`Component`] gluing the IPC worker to the libadwaita
//! window.
//!
//! Owns the application state (`AppState`), reacts to `WorkerOutput`
//! messages, and forwards user intent (preset switching, reload,
//! preview) back to the worker.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command as ProcessCommand;

use adw::prelude::*;
use fluxframe_core::protocol::Command;
use gtk::glib;
use relm4::prelude::{Component, ComponentParts, ComponentSender};
use relm4::{Sender, WorkerController};

use crate::components::{chain_page, preset_bar, preview, status_page};
use crate::debounce::Debouncer;
use crate::ipc::{IpcWorker, WorkerInput, WorkerOutput};
use crate::state::{AppState, ConnectionStatus};

/// Dwell time for daemon-error toasts. 5 seconds is long enough for
/// the operator to read but short enough to avoid stacking when a
/// slider drags through several rejection cases.
const TOAST_TIMEOUT_SECS: u32 = 5;

// TODO(stage-15): pull `[output].device` from the daemon's `GetConfig`
// round-trip instead of hardcoding /dev/video10. Same path used by both
// the embedded preview and the detached gst-launch viewer.
const DEFAULT_PREVIEW_DEVICE: &str = "/dev/video10";

/// Messages the AppModel handles internally.
#[derive(Debug)]
pub enum AppMsg {
    /// IPC worker reply. Wrapped so the AppModel can pattern-match
    /// without depending on the worker's `Output` type directly.
    Ipc(WorkerOutput),
    /// User clicked the preset DropDown — switch to `name`.
    SetPreset(String),
    /// Keyboard shortcut `Ctrl+<digit>` — switch to the preset at the
    /// 1-indexed slot, or no-op if the slot is empty.
    SetPresetByIndex {
        /// 1-indexed slot in the loaded preset list. `NonZeroUsize`
        /// rules out the `slot - 1` underflow at the type level.
        slot: std::num::NonZeroUsize,
    },
    /// User clicked the Reload button.
    Reload,
    /// User clicked "Open preview" — spawn an external viewer.
    OpenPreview,
    /// User clicked Retry on the disconnected status page.
    Retry,
    /// A parameter widget produced a new value — dispatch as
    /// `Command::Set { path, value }` to the daemon.
    SetParam {
        /// Dot-path `<section>.<effect>.<field>`.
        path: String,
        /// New value as a JSON `Value`.
        value: serde_json::Value,
    },
    /// User added an effect to a section's chain (via the section's
    /// add MenuButton). The new effect appends to the end.
    AddEffect {
        /// Sub-chain identifier as the typed [`fluxframe_core::SubchainKind`]
        /// enum (no longer accepts arbitrary strings).
        section: fluxframe_core::SubchainKind,
        /// Effect registry name to append.
        effect: String,
    },
    /// User removed an effect (✕ button on the chain item).
    RemoveEffect {
        /// Sub-chain identifier as the typed [`fluxframe_core::SubchainKind`]
        /// enum (no longer accepts arbitrary strings).
        section: fluxframe_core::SubchainKind,
        /// Position in the chain (0-indexed) to remove.
        index: usize,
    },
    /// User moved an effect one slot toward the chain head (`↑` button).
    /// No-op at index 0.
    MoveEffectUp {
        /// Sub-chain identifier as the typed [`fluxframe_core::SubchainKind`]
        /// enum (no longer accepts arbitrary strings).
        section: fluxframe_core::SubchainKind,
        /// Position to move (0-indexed); must be > 0.
        index: usize,
    },
    /// User moved an effect one slot toward the chain tail (`↓` button).
    /// No-op at the last index.
    MoveEffectDown {
        /// Sub-chain identifier as the typed [`fluxframe_core::SubchainKind`]
        /// enum (no longer accepts arbitrary strings).
        section: fluxframe_core::SubchainKind,
        /// Position to move (0-indexed).
        index: usize,
    },
}

/// Record of an in-flight `Command::Set` correlated by request tag.
///
/// `new` is the value the widget produced and the daemon is being
/// asked to apply; `previous` is the last-known-good fallback. Today
/// rollback is implemented by leaving `active_config` untouched on
/// `Err` and rebuilding the chain page — the widget re-reads
/// `active_config`. `previous` is kept on the struct so a future
/// targeted-widget rollback path (without a full rebuild) does not
/// need to re-derive it from the cache.
#[derive(Debug, Clone)]
struct PendingSet {
    path: String,
    new: serde_json::Value,
    #[allow(
        dead_code,
        reason = "reserved for future per-widget rollback (rebuild-from-active_config covers it today)"
    )]
    previous: serde_json::Value,
}

/// Classification of an in-flight request, used by `on_reply` to
/// dispatch on the reply payload without inspecting `data` shape
/// heuristics.
#[derive(Debug, Clone)]
enum PendingKind {
    /// A `Command::Set` issued by a widget change.
    SetParam(PendingSet),
    /// A `Command::GetConfig { path: None }` issued after Connected or
    /// Reload — the `Ok` payload is the full preset tree.
    GetConfig,
    /// A `Command::CurrentPreset` issued after SetPreset or Reload —
    /// the `Ok` payload is a JSON string with the active preset name.
    CurrentPreset,
    /// Anything else we send (Reload, SetPreset, SetChain, …). Tracked
    /// for completeness so unknown-tag warnings stay meaningful.
    Other,
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
    /// and the chain editor (connected).
    body_container: gtk::Box,
    /// Embedded live preview pane at the top of the window. Owns the
    /// preview GStreamer pipeline; dropping it tears the pipeline
    /// down. Kept as a field purely so its `Drop` runs at window
    /// close — the widget itself is already parented into `outer`.
    /// Underscore-prefixed so the compiler doesn't flag the
    /// never-read field.
    _preview: preview::Preview,
    /// Per-parameter debounce queue shared by all `param_row`
    /// instances on the chain page.
    debouncer: Debouncer,
    /// Toast overlay wrapping the window body. Used to surface daemon
    /// `Err` replies (e.g. parameter validation failures) without
    /// blocking the editor.
    toast_overlay: adw::ToastOverlay,
    /// Last-good values for each `<section>.<effect>.<field>` path.
    /// Updated when a `Set` succeeds; used to roll back the widget on
    /// failure. Keyed by the same dot-path the daemon accepts.
    last_known_good: HashMap<String, serde_json::Value>,
    /// In-flight requests indexed by their wire `tag`. Replies look up
    /// their pending kind here to decide how to interpret the payload
    /// (vs. fragile `data` shape heuristics).
    pending: HashMap<u64, PendingKind>,
}

impl Component for AppModel {
    type Init = PathBuf;
    type Input = AppMsg;
    type Output = ();
    type CommandOutput = ();
    type Root = adw::ApplicationWindow;
    type Widgets = ();

    fn init_root() -> Self::Root {
        // Only sizing + maximisation here; the close-request handler is
        // wired in `init()` once the `gtk::Paned` widget exists so we
        // can persist its split position alongside the window geometry.
        let geometry = crate::persistence::load();
        let window = adw::ApplicationWindow::builder()
            .title("FluxFrame")
            .default_width(geometry.width)
            .default_height(geometry.height)
            .build();
        if geometry.maximized {
            window.maximize();
        }
        window
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

        let preview = preview::build(std::path::Path::new(DEFAULT_PREVIEW_DEVICE));

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
        let geometry = crate::persistence::load();
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

        let outer = gtk::Box::new(gtk::Orientation::Vertical, 0);
        outer.append(&bar.root);
        outer.append(&paned);

        // Persist window geometry + Paned split on close. `close-request`
        // fires before teardown so the window and paned are still
        // queryable. `paned` is captured by move into the closure;
        // the closure outlives `init()` because GTK holds the signal.
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

        // Wrap the whole window body in a ToastOverlay so daemon
        // errors can fly over the chain page without disturbing the
        // layout. The overlay child has to be set before the window
        // content is attached.
        let toast_overlay = adw::ToastOverlay::new();
        toast_overlay.set_child(Some(&outer));
        root.set_content(Some(&toast_overlay));

        // Global keyboard shortcuts (Ctrl+R reload, Ctrl+Q quit,
        // Ctrl+1..9 preset switch).
        crate::shortcuts::install(&root, sender.input_sender());

        let state = AppState::new(socket_path);
        let input_sender = sender.input_sender().clone();
        let model = Self {
            state,
            worker: Some(worker),
            input_sender,
            next_tag: 0,
            preset_bar: bar,
            body_container: body,
            _preview: preview,
            debouncer: Debouncer::new(),
            toast_overlay,
            last_known_good: HashMap::new(),
            pending: HashMap::new(),
        };
        ComponentParts { model, widgets: () }
    }

    fn update(&mut self, msg: Self::Input, _sender: ComponentSender<Self>, _root: &Self::Root) {
        match msg {
            AppMsg::Ipc(out) => self.on_ipc(out),
            AppMsg::SetPreset(name) => {
                self.send_kind(Command::SetPreset { name }, PendingKind::Other);
            }
            AppMsg::SetPresetByIndex { slot } => {
                // 1-indexed slot → list position (slot - 1). `NonZeroUsize`
                // makes the subtraction safe at the type level.
                let Some(name) = self.state.presets.get(slot.get() - 1).cloned() else {
                    tracing::debug!(slot = slot.get(), "no preset at slot");
                    return;
                };
                let _ = self.input_sender.send(AppMsg::SetPreset(name));
            }
            AppMsg::Reload => {
                self.send_kind(Command::Reload, PendingKind::Other);
                // After a successful reload the cached active_preset
                // / config may have drifted; refetch the state.
                self.send_kind(Command::CurrentPreset, PendingKind::CurrentPreset);
                self.send_kind(Command::GetConfig { path: None }, PendingKind::GetConfig);
            }
            AppMsg::OpenPreview => open_preview(),
            AppMsg::Retry => self.retry(),
            AppMsg::SetParam { path, value } => {
                self.send_set_param(path, value);
            }
            AppMsg::AddEffect { section, effect } => {
                self.mutate_chain(section, |chain| chain.push(effect));
            }
            AppMsg::RemoveEffect { section, index } => {
                self.mutate_chain(section, |chain| {
                    if index < chain.len() {
                        chain.remove(index);
                    }
                });
            }
            AppMsg::MoveEffectUp { section, index } => {
                if index == 0 {
                    return;
                }
                self.mutate_chain(section, |chain| {
                    if index < chain.len() {
                        chain.swap(index - 1, index);
                    }
                });
            }
            AppMsg::MoveEffectDown { section, index } => {
                self.mutate_chain(section, |chain| {
                    if index + 1 < chain.len() {
                        chain.swap(index, index + 1);
                    }
                });
            }
        }
    }
}

impl AppModel {
    fn next_tag(&mut self) -> u64 {
        let tag = self.next_tag;
        self.next_tag = self.next_tag.wrapping_add(1);
        tag
    }

    /// Send a command to the worker and register its `tag` with a
    /// classification used by `on_reply` to dispatch on the reply.
    fn send_kind(&mut self, command: Command, kind: PendingKind) {
        if self.worker.is_none() {
            tracing::warn!("send_kind dropped — worker not connected");
            return;
        }
        let tag = self.next_tag();
        // Worker presence re-checked just above; safe to unwrap is justified.
        let worker = self.worker.as_ref().expect("worker presence checked above");
        self.pending.insert(tag, kind);
        let _ = worker.sender().send(WorkerInput::Send { tag, command });
    }

    /// Apply `mutate` to the chain array for `section`, dispatch a
    /// `Command::SetChain` with the result, then refetch the active
    /// config so widgets pick up the new chain shape (and any defaulting
    /// the daemon applied).
    fn mutate_chain(
        &mut self,
        section: fluxframe_core::SubchainKind,
        mutate: impl FnOnce(&mut Vec<String>),
    ) {
        let chain_str = section.as_str();
        let mut chain = self.state.chain_for(chain_str);
        mutate(&mut chain);
        self.send_kind(Command::SetChain { section, chain }, PendingKind::Other);
        // Refetch the active config so widgets pick up the new chain
        // shape (and any defaulting the daemon applied).
        self.send_kind(Command::GetConfig { path: None }, PendingKind::GetConfig);
    }

    /// Dispatch a parameter change as a `Command::Set` while capturing
    /// the prior value so the widget can be rolled back if the daemon
    /// rejects it.
    fn send_set_param(&mut self, path: String, value: serde_json::Value) {
        // Previous value preference order: the last-known-good cache
        // (only populated on accepted Sets) → the current active_config
        // entry → `Null`.
        let previous = self
            .last_known_good
            .get(&path)
            .cloned()
            .or_else(|| read_config_value(&self.state.active_config, &path))
            .unwrap_or(serde_json::Value::Null);
        let kind = PendingKind::SetParam(PendingSet {
            path: path.clone(),
            new: value.clone(),
            previous,
        });
        self.send_kind(Command::Set { path, value }, kind);
    }

    fn on_ipc(&mut self, out: WorkerOutput) {
        match out {
            WorkerOutput::Connected(initial) => {
                tracing::info!(presets = initial.presets.len(), "daemon handshake complete");
                self.state.status = ConnectionStatus::Connected;
                self.state.presets = initial.presets;
                self.state.active_preset = Some(initial.active_preset.clone());
                self.state.inventory = initial.inventory;
                self.state.active_config = initial.active_config;

                preset_bar::set_presets(
                    &self.preset_bar,
                    &self.state.presets,
                    self.state.active_preset.as_deref(),
                );
                if let Some(active) = self.state.active_preset.as_deref() {
                    preset_bar::set_active_preset(&self.preset_bar, active);
                }
                self.rebuild_chain_page();
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
        let kind = self.pending.remove(&tag);
        match response {
            fluxframe_core::protocol::Response::Ok { data } => {
                tracing::debug!(tag, ?data, "reply ok");
                match kind {
                    Some(PendingKind::SetParam(set)) => {
                        // Daemon accepted the write. Mirror it into
                        // active_config so subsequent rebuilds reflect
                        // the new value, and remember the value as the
                        // rollback target for future Sets on this path.
                        set_config_value(&mut self.state.active_config, &set.path, set.new.clone());
                        self.last_known_good.insert(set.path, set.new);
                    }
                    Some(PendingKind::CurrentPreset) => {
                        if let Some(name) = data.as_str() {
                            self.state.active_preset = Some(name.to_string());
                            preset_bar::set_active_preset(&self.preset_bar, name);
                        }
                    }
                    Some(PendingKind::GetConfig) => {
                        // Full preset tree — refresh and rebuild.
                        self.state.active_config = data;
                        self.rebuild_chain_page();
                    }
                    Some(PendingKind::Other) | None => {
                        // Reload / SetPreset / SetChain or an
                        // untracked tag — nothing further to do; the
                        // follow-up CurrentPreset/GetConfig will
                        // refresh the UI.
                    }
                }
            }
            fluxframe_core::protocol::Response::Err { error, hint } => {
                tracing::warn!(tag, error = %error, hint = ?hint, "reply err");
                match kind {
                    Some(PendingKind::SetParam(_set)) => {
                        // Daemon rejected the write. Surface the error
                        // via a toast and rebuild the chain page so
                        // every widget snaps back to active_config
                        // (which still holds the previous value because
                        // we only commit on Ok above).
                        self.show_error_toast(&error, hint.as_deref());
                        self.rebuild_chain_page();
                    }
                    Some(PendingKind::Other) => {
                        // Reload / SetPreset / SetChain — no widget
                        // rollback, but the user still deserves to
                        // know the daemon refused the request (e.g.
                        // "preset does not have a [post] sub-section"
                        // on AddEffect into an absent section).
                        self.show_error_toast(&error, hint.as_deref());
                    }
                    Some(PendingKind::CurrentPreset | PendingKind::GetConfig) | None => {
                        // Internal refetches and unknown tags
                        // — log only, the operator will see the warn.
                    }
                }
            }
            // `Response` is `#[non_exhaustive]`; tolerate future
            // variants without panicking the UI thread.
            _ => {
                tracing::warn!(tag, "unrecognised Response variant");
            }
        }
    }

    /// Surface a daemon error as a 5-second `AdwToast` overlay banner.
    ///
    /// Used by [`Self::on_reply`] for any [`PendingKind`] variant whose
    /// Err reply deserves operator attention. Internally builds a single
    /// formatted line via [`format_toast_error`] so error and hint share
    /// a consistent shape across rejection sources.
    fn show_error_toast(&self, error: &str, hint: Option<&str>) {
        let message = format_toast_error(error, hint);
        let toast = adw::Toast::builder()
            .title(&message)
            .timeout(TOAST_TIMEOUT_SECS)
            .build();
        self.toast_overlay.add_toast(toast);
    }

    /// Drop the current body content and re-render the chain editor
    /// from the active state. Called after Connected, SetPreset, and
    /// any successful refetch.
    fn rebuild_chain_page(&self) {
        tracing::debug!(
            sections = chain_page::SECTION_COUNT,
            preset = self.state.active_preset.as_deref().unwrap_or("(none)"),
            "rebuilt chain page"
        );
        clear_children(&self.body_container);
        // `chain_page::build` already returns a `ScrolledWindow` with
        // the right policy/expand flags — no need to wrap again.
        let page = chain_page::build(&self.state, &self.debouncer, &self.input_sender);
        self.body_container.append(&page);
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

/// Read the leaf value at `path` (`<section>.<effect>.<field>`) out of
/// the active config tree. Returns `None` if any segment is missing.
///
/// Mirrors the layout the daemon's `Set` command writes back: the path
/// is rewritten to `<section>.per_effect.<effect>.<field>` for the
/// lookup, because per-effect fields live under `per_effect`.
fn read_config_value(config: &serde_json::Value, path: &str) -> Option<serde_json::Value> {
    let mut parts = path.splitn(3, '.');
    let section = parts.next()?;
    let effect = parts.next()?;
    let field = parts.next()?;
    config
        .get(section)?
        .get("per_effect")?
        .get(effect)?
        .get(field)
        .cloned()
}

/// Write `value` at `<section>.per_effect.<effect>.<field>` inside the
/// active config tree, creating intermediate objects as needed. Used
/// to keep the cached config in sync when a `Set` is accepted by the
/// daemon so subsequent `rebuild_chain_page` calls render the new
/// value without a full `GetConfig` round-trip.
fn set_config_value(config: &mut serde_json::Value, path: &str, value: serde_json::Value) {
    let mut parts = path.splitn(3, '.');
    let (Some(section), Some(effect), Some(field)) = (parts.next(), parts.next(), parts.next())
    else {
        tracing::warn!(path, "set_config_value: malformed dot-path, skipped");
        return;
    };
    if !config.is_object() {
        *config = serde_json::Value::Object(serde_json::Map::new());
    }
    let root = config
        .as_object_mut()
        .expect("config coerced to object just above");
    let section_obj = root
        .entry(section.to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !section_obj.is_object() {
        *section_obj = serde_json::Value::Object(serde_json::Map::new());
    }
    let section_map = section_obj
        .as_object_mut()
        .expect("section coerced to object just above");
    let per_effect = section_map
        .entry("per_effect".to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !per_effect.is_object() {
        *per_effect = serde_json::Value::Object(serde_json::Map::new());
    }
    let per_effect_map = per_effect
        .as_object_mut()
        .expect("per_effect coerced to object just above");
    let effect_obj = per_effect_map
        .entry(effect.to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !effect_obj.is_object() {
        *effect_obj = serde_json::Value::Object(serde_json::Map::new());
    }
    let effect_map = effect_obj
        .as_object_mut()
        .expect("effect coerced to object just above");
    effect_map.insert(field.to_string(), value);
}

/// Compose a one-line toast title from the daemon's error reason and
/// optional hint. Kept short — toasts wrap awkwardly past one line.
fn format_toast_error(error: &str, hint: Option<&str>) -> String {
    match hint {
        Some(h) if !h.is_empty() => format!("{error} — {h}"),
        _ => error.to_string(),
    }
}

/// Spawn `gst-launch-1.0` on `/dev/video10` to show the daemon's
/// live output. Detached — closing the GUI does not kill the viewer
/// (use case: open preview once, keep tuning).
///
/// Failures are logged but not surfaced to the user via a dialog
/// because the operation is best-effort.
fn open_preview() {
    let mut cmd = ProcessCommand::new("gst-launch-1.0");
    let device_arg = format!("device={DEFAULT_PREVIEW_DEVICE}");
    cmd.args([
        "v4l2src",
        device_arg.as_str(),
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
