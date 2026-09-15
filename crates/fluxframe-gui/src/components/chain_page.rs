//! Single-page chain editor.
//!
//! Renders the active preset as one [`adw::PreferencesPage`] holding
//! four [`adw::PreferencesGroup`]s (mask / background / foreground /
//! post). Each group lists the effects currently in the chain as
//! [`adw::ExpanderRow`]s; expanding a row reveals one
//! [`crate::components::param_row`] per descriptor in the effect's
//! metadata.
//!
//! The chain page is rebuilt from scratch on preset switch, chain
//! edits and state refetch. Rebuilds are user-driven, not per-frame;
//! the caller carries expanded rows, scroll position and keyboard
//! focus across a rebuild via [`ChainPage::view_state`] /
//! [`ChainPage::restore`]. A refused parameter write resets only its
//! own widget via [`ChainPage::reset_param`]; an effect's enable switch
//! is updated in place via [`ChainPage::set_toggle`], without a rebuild.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use adw::prelude::*;
use fluxframe_core::{EffectSchema, SetPath, SubchainKind};
use gtk::glib;
use serde_json::Value;

use crate::components::Emit;
use crate::components::param_row::{self, DispatchCtx, ParamChange, ParamReset};
use crate::debounce::Debouncer;
use crate::state::{AppState, ChainEdit};

/// Cap on the add-effect popover height; longer inventories scroll.
const ADD_EFFECT_MENU_MAX_HEIGHT: i32 = 360;

/// User intent reported by the chain page.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ChainEvent {
    /// Change the effect list of `section`.
    Edit {
        /// Sub-chain being edited.
        section: SubchainKind,
        /// Requested change.
        edit: ChainEdit,
    },
    /// A param row produced a new value.
    Param(ParamChange),
    /// The operator flipped an effect's enable switch.
    SetEnabled {
        /// Sub-chain of the effect.
        section: SubchainKind,
        /// Effect name.
        effect: String,
        /// New switch position.
        enabled: bool,
    },
}

/// Which per-effect suffix control a focus key refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ChainButtonKind {
    Toggle,
    Up,
    Down,
    Remove,
}

/// Identity of a per-effect suffix control that survives a rebuild.
type ButtonKey = (SubchainKind, String, ChainButtonKind);

/// The enable switch of one effect row, with what [`ChainPage::set_toggle`]
/// needs to update it in place.
struct EffectToggle {
    section: SubchainKind,
    effect: String,
    switch: gtk::Switch,
    handler: glib::SignalHandlerId,
    row: adw::ExpanderRow,
    help: String,
}

/// A built chain page plus the handles needed to snapshot and
/// restore its view state across rebuilds.
pub(crate) struct ChainPage {
    /// Scroller wrapping the preferences page; attach it to the body
    /// container.
    pub(crate) root: gtk::ScrolledWindow,
    /// Effect expander rows keyed by (section, effect name).
    expanders: Vec<(SubchainKind, String, adw::ExpanderRow)>,
    /// Per-effect suffix controls, for focus carry-over.
    focusables: Vec<(ButtonKey, gtk::Widget)>,
    /// Enable switches of every effect row.
    toggles: Vec<EffectToggle>,
    /// Reset handles of every param row.
    resets: HashMap<SetPath, ParamReset>,
}

/// Transient view state of a chain page: which effect rows are
/// expanded, where the page is scrolled and which suffix button has
/// keyboard focus.
pub(crate) struct ChainViewState {
    expanded: HashSet<(SubchainKind, String)>,
    scroll: f64,
    focused: Option<ButtonKey>,
}

impl ChainPage {
    /// Snapshot expanded rows, the vertical scroll offset and the
    /// focused suffix button.
    pub(crate) fn view_state(&self) -> ChainViewState {
        let expanded = self
            .expanders
            .iter()
            .filter(|(_, _, row)| row.is_expanded())
            .map(|(section, name, _)| (*section, name.clone()))
            .collect();
        let focused = self
            .focusables
            .iter()
            .find(|(_, widget)| widget.has_focus())
            .map(|(key, _)| key.clone());
        ChainViewState {
            expanded,
            scroll: self.root.vadjustment().value(),
            focused,
        }
    }

    /// Put the widget of `path` back to `value` without emitting a
    /// change. No-op when the page has no row for `path`.
    pub(crate) fn reset_param(&self, path: &SetPath, value: &Value) {
        if let Some(reset) = self.resets.get(path) {
            reset(value);
        }
    }

    /// Show `effect` of `section` as `enabled` without emitting a
    /// change: switch position and row subtitle of every row with that
    /// name (a chain may list an effect twice).
    pub(crate) fn set_toggle(&self, section: SubchainKind, effect: &str, enabled: bool) {
        for toggle in self
            .toggles
            .iter()
            .filter(|t| t.section == section && t.effect == effect)
        {
            toggle.switch.block_signal(&toggle.handler);
            toggle.switch.set_active(enabled);
            toggle.switch.unblock_signal(&toggle.handler);
            toggle
                .row
                .set_subtitle(&effect_subtitle(&toggle.help, enabled));
        }
    }

    /// Re-expand rows present in `state`, restore the scroll offset and
    /// return keyboard focus to the same suffix button (or its row, when
    /// the button became insensitive, e.g. Move Up at the head).
    ///
    /// A freshly built page has no allocation yet, so the scroll offset
    /// is applied on the vertical adjustment's first `changed` emission
    /// with a non-zero page size; the handler then disconnects itself.
    /// Rows are expanded before the page is mapped, so the expander
    /// revealers open without animation and the first allocation
    /// already accounts for their content.
    pub(crate) fn restore(&self, state: &ChainViewState) {
        for (section, name, row) in &self.expanders {
            if state.expanded.contains(&(*section, name.clone())) {
                row.set_expanded(true);
            }
        }

        if let Some((section, name, kind)) = &state.focused {
            let button = self
                .focusables
                .iter()
                .find(|((s, n, k), _)| s == section && n == name && k == kind)
                .map(|(_, widget)| widget);
            if let Some(button) = button.filter(|b| b.is_sensitive()) {
                button.grab_focus();
            } else if let Some((_, _, row)) = self
                .expanders
                .iter()
                .find(|(s, n, _)| s == section && n == name)
            {
                row.grab_focus();
            }
        }

        let target = state.scroll;
        if target <= 0.0 {
            return;
        }
        let adjustment = self.root.vadjustment();
        if adjustment.page_size() > 0.0 {
            adjustment.set_value(target);
            return;
        }
        // The handler owns only its own id cell (no widget), so there
        // is no reference cycle; the id is taken exactly once.
        let handler: Rc<RefCell<Option<glib::SignalHandlerId>>> = Rc::new(RefCell::new(None));
        let handler_for_cb = Rc::clone(&handler);
        let id = adjustment.connect_changed(move |adj| {
            if adj.page_size() <= 0.0 {
                return;
            }
            adj.set_value(target.min(adj.upper() - adj.page_size()));
            if let Some(id) = handler_for_cb.borrow_mut().take() {
                adj.disconnect(id);
            }
        });
        *handler.borrow_mut() = Some(id);
    }
}

/// Display title + group description for one chain section.
fn section_labels(section: SubchainKind) -> (&'static str, &'static str) {
    match section {
        SubchainKind::Mask => (
            "Mask chain",
            "Effects applied to the segmentation mask before compositing.",
        ),
        SubchainKind::Background => (
            "Background chain",
            "Effects applied to pixels behind the subject.",
        ),
        SubchainKind::Foreground => (
            "Foreground chain",
            "Effects applied to the subject's pixels.",
        ),
        SubchainKind::Post => (
            "Post-composite chain",
            "Mask-aware effects applied after the composite.",
        ),
    }
}

/// Build the chain page from the current [`AppState`].
///
/// The returned [`ChainPage::root`] is a [`gtk::ScrolledWindow`]
/// wrapping the [`adw::PreferencesPage`] — callers attach it to the
/// body container. User intent is reported through `emit`; param rows
/// route their changes through `debouncer` first.
pub(crate) fn build(state: &AppState, debouncer: &Debouncer, emit: &Emit<ChainEvent>) -> ChainPage {
    let page = adw::PreferencesPage::new();
    let mut parts = PageParts::default();
    let param_emit: Emit<ParamChange> = {
        let emit = Emit::clone(emit);
        std::rc::Rc::new(move |change| emit(ChainEvent::Param(change)))
    };

    for section_kind in SubchainKind::ALL {
        let (title, description) = section_labels(section_kind);
        let group = adw::PreferencesGroup::builder()
            .title(title)
            .description(description)
            .build();
        // Add-effect MenuButton in the group header (suffix slot).
        let chain = state.chain_for(section_kind);
        let inventory_for_section = state.effects_for(section_kind);
        group.set_header_suffix(Some(&add_effect_menu_button(
            section_kind,
            inventory_for_section,
            emit,
        )));
        if chain.is_empty() {
            group.add(&placeholder_row());
        } else {
            let chain_len = chain.len();
            for (idx, effect_name) in chain.iter().enumerate() {
                let Some(meta) = find_effect(inventory_for_section, effect_name) else {
                    // The chain references an effect the registry no
                    // longer carries (e.g. a feature-gated effect on
                    // a slim daemon). Render a stub so the operator
                    // sees the divergence.
                    group.add(&unknown_effect_row(effect_name));
                    continue;
                };
                let ctx = EffectExpanderCtx {
                    section: section_kind,
                    index: idx,
                    chain_len,
                    state,
                    debouncer,
                    emit,
                    param_emit: &param_emit,
                };
                let row = effect_expander(ctx, meta, &mut parts);
                group.add(&row);
                parts.expanders.push((section_kind, meta.name.clone(), row));
            }
        }
        page.add(&group);
    }

    let root = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .hexpand(true)
        .vexpand(true)
        .child(&page)
        .build();
    let PageParts {
        expanders,
        focusables,
        toggles,
        resets,
    } = parts;
    ChainPage {
        root,
        expanders,
        focusables,
        toggles,
        resets,
    }
}

/// Handles collected while building the page.
#[derive(Default)]
struct PageParts {
    expanders: Vec<(SubchainKind, String, adw::ExpanderRow)>,
    focusables: Vec<(ButtonKey, gtk::Widget)>,
    toggles: Vec<EffectToggle>,
    resets: HashMap<SetPath, ParamReset>,
}

/// Row subtitle of an effect: its help text, prefixed when disabled so
/// the state is carried by text, not by the switch position alone.
fn effect_subtitle(help: &str, enabled: bool) -> String {
    if enabled {
        help.to_string()
    } else {
        format!("Disabled — {help}")
    }
}

fn placeholder_row() -> adw::ActionRow {
    adw::ActionRow::builder()
        .title("No effects added")
        .subtitle("Use the + button to add one.")
        .build()
}

fn unknown_effect_row(name: &str) -> adw::ActionRow {
    adw::ActionRow::builder()
        .title(name.to_string())
        .subtitle("Unknown effect — not in registry inventory.")
        .build()
}

fn find_effect<'a>(inventory: &'a [EffectSchema], name: &str) -> Option<&'a EffectSchema> {
    inventory.iter().find(|m| m.name == name)
}

/// Context bundled into [`effect_expander`] so the function does
/// not have to thread 6 unrelated params individually.
#[derive(Clone, Copy)]
struct EffectExpanderCtx<'a> {
    section: SubchainKind,
    index: usize,
    chain_len: usize,
    state: &'a AppState,
    debouncer: &'a Debouncer,
    emit: &'a Emit<ChainEvent>,
    param_emit: &'a Emit<ParamChange>,
}

fn effect_expander(
    ctx: EffectExpanderCtx<'_>,
    meta: &EffectSchema,
    parts: &mut PageParts,
) -> adw::ExpanderRow {
    let enabled = ctx.state.effect_enabled(ctx.section, &meta.name);
    let row = adw::ExpanderRow::builder()
        .title(meta.name.as_str())
        .subtitle(effect_subtitle(&meta.help, enabled))
        .build();

    // Enable switch. Its initial state is set by the builder, before the
    // handler is connected, so building or rebuilding a page never emits
    // a toggle.
    let switch = gtk::Switch::builder()
        .active(enabled)
        .valign(gtk::Align::Center)
        .tooltip_text("Enable Effect")
        .build();
    switch.update_property(&[gtk::accessible::Property::Label(&format!(
        "Enable {}",
        meta.name
    ))]);
    let handler = {
        let emit = Emit::clone(ctx.emit);
        let section = ctx.section;
        let effect = meta.name.clone();
        switch.connect_active_notify(move |switch| {
            emit(ChainEvent::SetEnabled {
                section,
                effect: effect.clone(),
                enabled: switch.is_active(),
            });
        })
    };
    row.add_suffix(&switch);
    parts.focusables.push((
        (ctx.section, meta.name.clone(), ChainButtonKind::Toggle),
        switch.clone().upcast(),
    ));
    parts.toggles.push(EffectToggle {
        section: ctx.section,
        effect: meta.name.clone(),
        switch,
        handler,
        row: row.clone(),
        help: meta.help.clone(),
    });

    // Suffix toolbar: ↑ ↓ ✕ — each emits a chain edit when clicked.
    // Disabled at the boundary positions so the user gets visual
    // feedback.
    let buttons = [
        (
            ChainButton {
                icon: "go-up-symbolic",
                tooltip: "Move Up",
                accessible_label: format!("Move {} Up", meta.name),
                enabled: ctx.index > 0,
            },
            ChainButtonKind::Up,
            ChainEdit::MoveUp(ctx.index),
        ),
        (
            ChainButton {
                icon: "go-down-symbolic",
                tooltip: "Move Down",
                accessible_label: format!("Move {} Down", meta.name),
                enabled: ctx.index + 1 < ctx.chain_len,
            },
            ChainButtonKind::Down,
            ChainEdit::MoveDown(ctx.index),
        ),
        (
            ChainButton {
                icon: "edit-delete-symbolic",
                tooltip: "Remove From Chain",
                accessible_label: format!("Remove {} From Chain", meta.name),
                enabled: true,
            },
            ChainButtonKind::Remove,
            ChainEdit::Remove(ctx.index),
        ),
    ];
    for (look, kind, edit) in buttons {
        let button = chain_action_button(look, ctx.section, edit, ctx.emit);
        if kind == ChainButtonKind::Remove {
            button.add_css_class("error");
        }
        row.add_suffix(&button);
        parts
            .focusables
            .push(((ctx.section, meta.name.clone(), kind), button.upcast()));
    }

    for desc in &meta.params {
        let path = SetPath {
            section: ctx.section,
            effect: meta.name.clone(),
            field: desc.name.clone(),
        };
        let initial = ctx.state.config_field(&path);
        let dispatch = DispatchCtx {
            debouncer: ctx.debouncer.clone(),
            emit: Emit::clone(ctx.param_emit),
        };
        let param = param_row::build(desc, path.clone(), initial, dispatch);
        row.add_row(&param.row);
        parts.resets.insert(path, param.reset);
    }
    row
}

/// Appearance of a per-effect suffix button.
struct ChainButton {
    icon: &'static str,
    tooltip: &'static str,
    /// Full accessible label naming the effect, so a screen reader
    /// distinguishes the buttons of different rows.
    accessible_label: String,
    enabled: bool,
}

/// Build one of the per-effect flat suffix buttons (↑/↓/✕) emitting
/// `chain_edit` for `section` when clicked.
fn chain_action_button(
    look: ChainButton,
    section: SubchainKind,
    chain_edit: ChainEdit,
    emit: &Emit<ChainEvent>,
) -> gtk::Button {
    let ChainButton {
        icon,
        tooltip,
        accessible_label,
        enabled,
    } = look;
    let button = gtk::Button::from_icon_name(icon);
    button.set_valign(gtk::Align::Center);
    button.set_tooltip_text(Some(tooltip));
    button.update_property(&[gtk::accessible::Property::Label(&accessible_label)]);
    button.add_css_class("flat");
    button.set_sensitive(enabled);
    let emit = Emit::clone(emit);
    button.connect_clicked(move |_| {
        emit(ChainEvent::Edit {
            section,
            edit: chain_edit.clone(),
        });
    });
    button
}

/// Build a `+` MenuButton listing every effect available in the
/// section's inventory. Clicking an item emits an append edit.
fn add_effect_menu_button(
    section: SubchainKind,
    inventory: &[EffectSchema],
    emit: &Emit<ChainEvent>,
) -> gtk::MenuButton {
    let popover = gtk::Popover::new();
    let list = gtk::Box::new(gtk::Orientation::Vertical, 4);
    list.set_margin_top(6);
    list.set_margin_bottom(6);
    list.set_margin_start(6);
    list.set_margin_end(6);
    if inventory.is_empty() {
        let lbl = gtk::Label::new(Some("No Effects Available"));
        list.append(&lbl);
    } else {
        for meta in inventory {
            let btn = gtk::Button::with_label(&meta.name);
            btn.set_has_frame(false);
            let emit = Emit::clone(emit);
            let effect = meta.name.clone();
            // The popover (transitively) owns this button, so hold it
            // weakly to avoid a reference cycle.
            btn.connect_clicked(glib::clone!(
                #[weak]
                popover,
                move |_| {
                    popover.popdown();
                    emit(ChainEvent::Edit {
                        section,
                        edit: ChainEdit::Append(effect.clone()),
                    });
                }
            ));
            list.append(&btn);
        }
    }
    // Long inventories scroll instead of growing past the window.
    let scroller = gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .propagate_natural_height(true)
        .propagate_natural_width(true)
        .max_content_height(ADD_EFFECT_MENU_MAX_HEIGHT)
        .child(&list)
        .build();
    popover.set_child(Some(&scroller));

    let (title, _) = section_labels(section);
    let menu = gtk::MenuButton::builder()
        .icon_name("list-add-symbolic")
        .tooltip_text("Add Effect to Chain")
        .popover(&popover)
        .valign(gtk::Align::Center)
        .build();
    menu.update_property(&[gtk::accessible::Property::Label(&format!(
        "Add Effect to {title}"
    ))]);
    menu.add_css_class("flat");
    menu
}
