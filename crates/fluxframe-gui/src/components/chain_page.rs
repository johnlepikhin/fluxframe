//! Single-page chain editor.
//!
//! Renders the active preset as one [`adw::PreferencesPage`] holding
//! four [`adw::PreferencesGroup`]s (mask / background / foreground /
//! post). Each group lists the effects currently in the chain as
//! [`adw::ExpanderRow`]s; expanding a row reveals one
//! [`crate::components::param_row`] per descriptor in the effect's
//! metadata.
//!
//! The page is rebuilt from scratch on every preset switch — chain
//! membership changes (Step 5.5+) will also trigger a rebuild. This
//! is acceptable because rebuilds are user-driven, not per-frame.

use std::str::FromStr;

use adw::prelude::*;
use fluxframe_core::{EffectMetadata, SubchainKind};
use relm4::Sender;

use crate::app::AppMsg;
use crate::components::param_row::{self, DispatchCtx};
use crate::debounce::Debouncer;
use crate::state::AppState;

/// Number of sections rendered by the chain page. Re-exported so
/// sibling modules that need to size collections by section count do
/// not have to hard-code `4`.
pub(crate) const SECTION_COUNT: usize = SECTION_LABELS.len();

/// Per-section identifier + display title + group description, in
/// page-display order. Single source of truth for the four chain
/// sections.
const SECTION_LABELS: [(&str, &str, &str); 4] = [
    (
        "mask",
        "Mask chain",
        "Effects applied to the segmentation mask before compositing.",
    ),
    (
        "background",
        "Background chain",
        "Effects applied to pixels behind the subject.",
    ),
    (
        "foreground",
        "Foreground chain",
        "Effects applied to the subject's pixels.",
    ),
    (
        "post",
        "Post-composite chain",
        "Mask-aware effects applied after the composite.",
    ),
];

/// Build the chain page from the current [`AppState`].
///
/// Returns a [`gtk::ScrolledWindow`] wrapping the
/// [`adw::PreferencesPage`] — callers attach the scroller to the body
/// container. The page captures `debouncer` and `sender` clones; each
/// param row drives them when the user moves a widget.
pub(crate) fn build(
    state: &AppState,
    debouncer: &Debouncer,
    sender: &Sender<AppMsg>,
) -> gtk::ScrolledWindow {
    let page = adw::PreferencesPage::new();

    for (section, title, description) in SECTION_LABELS {
        // SECTION_LABELS is hand-maintained alongside the
        // SubchainKind variants — the parse is infallible by
        // construction.
        let section_kind = SubchainKind::from_str(section)
            .expect("SECTION_LABELS entries are valid SubchainKind names");
        let group = adw::PreferencesGroup::builder()
            .title(title)
            .description(description)
            .build();
        // Add-effect MenuButton in the group header (suffix slot).
        let chain = state.chain_for(section);
        let inventory_for_section = state
            .inventory
            .sections
            .get(section)
            .map_or(&[][..], Vec::as_slice);
        group.set_header_suffix(Some(&add_effect_menu_button(
            section_kind,
            inventory_for_section,
            sender,
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
                    sender,
                };
                group.add(&effect_expander(ctx, meta));
            }
        }
        page.add(&group);
    }

    gtk::ScrolledWindow::builder()
        .hscrollbar_policy(gtk::PolicyType::Never)
        .vscrollbar_policy(gtk::PolicyType::Automatic)
        .hexpand(true)
        .vexpand(true)
        .child(&page)
        .build()
}

fn placeholder_row() -> adw::ActionRow {
    adw::ActionRow::builder()
        .title("(empty)")
        .subtitle("No effects in this section.")
        .build()
}

fn unknown_effect_row(name: &str) -> adw::ActionRow {
    adw::ActionRow::builder()
        .title(name.to_string())
        .subtitle("Unknown effect — not in registry inventory.")
        .build()
}

fn find_effect<'a>(inventory: &'a [EffectMetadata], name: &str) -> Option<&'a EffectMetadata> {
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
    sender: &'a Sender<AppMsg>,
}

fn effect_expander(ctx: EffectExpanderCtx<'_>, meta: &EffectMetadata) -> adw::ExpanderRow {
    let row = adw::ExpanderRow::builder()
        .title(meta.name)
        .subtitle(meta.help)
        .build();

    // Suffix toolbar: ↑ ↓ ✕ — each opens a fresh AppMsg send when
    // clicked. Disabled at the boundary positions so the user gets
    // visual feedback.
    let up = chain_action_button(
        "go-up-symbolic",
        "Move up",
        ctx.index > 0,
        ctx.section,
        ctx.index,
        ctx.sender,
        |section, index| AppMsg::MoveEffectUp { section, index },
    );
    row.add_suffix(&up);

    let down = chain_action_button(
        "go-down-symbolic",
        "Move down",
        ctx.index + 1 < ctx.chain_len,
        ctx.section,
        ctx.index,
        ctx.sender,
        |section, index| AppMsg::MoveEffectDown { section, index },
    );
    row.add_suffix(&down);

    let remove = chain_action_button(
        "edit-delete-symbolic",
        "Remove from chain",
        true,
        ctx.section,
        ctx.index,
        ctx.sender,
        |section, index| AppMsg::RemoveEffect { section, index },
    );
    remove.add_css_class("destructive-action");
    row.add_suffix(&remove);

    let section_str = ctx.section.as_str();
    for desc in meta.params {
        let path = format!("{section_str}.{}.{}", meta.name, desc.name);
        let initial = ctx.state.config_field(section_str, meta.name, desc.name);
        let dispatch = DispatchCtx {
            debouncer: ctx.debouncer.clone(),
            sender: ctx.sender.clone(),
        };
        // `ParamDescriptor: Copy` — pass by value so `param_row`
        // does not need a `'static` lifetime on the borrow.
        let param = param_row::build(*desc, path, initial, dispatch);
        row.add_row(&param);
    }
    row
}

/// Build one of the per-effect suffix buttons (↑/↓/✕).
///
/// `ctor` builds the [`AppMsg`] from `(section, index)` — pass
/// `|section, index| AppMsg::MoveEffectUp { section, index }`,
/// etc.
fn chain_action_button(
    icon: &str,
    tooltip: &str,
    enabled: bool,
    section: SubchainKind,
    index: usize,
    sender: &Sender<AppMsg>,
    ctor: impl Fn(SubchainKind, usize) -> AppMsg + 'static,
) -> gtk::Button {
    let button = gtk::Button::from_icon_name(icon);
    button.set_valign(gtk::Align::Center);
    button.set_tooltip_text(Some(tooltip));
    button.set_sensitive(enabled);
    let sender = sender.clone();
    button.connect_clicked(move |_| {
        let _ = sender.send(ctor(section, index));
    });
    button
}

/// Build a `+` MenuButton listing every effect available in the
/// section's inventory. Clicking an item dispatches
/// [`AppMsg::AddEffect`].
fn add_effect_menu_button(
    section: SubchainKind,
    inventory: &[EffectMetadata],
    sender: &Sender<AppMsg>,
) -> gtk::MenuButton {
    let popover = gtk::Popover::new();
    let list = gtk::Box::new(gtk::Orientation::Vertical, 4);
    list.set_margin_top(6);
    list.set_margin_bottom(6);
    list.set_margin_start(6);
    list.set_margin_end(6);
    if inventory.is_empty() {
        let lbl = gtk::Label::new(Some("(no effects available)"));
        list.append(&lbl);
    } else {
        for meta in inventory {
            let btn = gtk::Button::with_label(meta.name);
            btn.set_has_frame(false);
            let popover_for_btn = popover.clone();
            let sender = sender.clone();
            let effect = meta.name.to_string();
            btn.connect_clicked(move |_| {
                popover_for_btn.popdown();
                let _ = sender.send(AppMsg::AddEffect {
                    section,
                    effect: effect.clone(),
                });
            });
            list.append(&btn);
        }
    }
    popover.set_child(Some(&list));

    let menu = gtk::MenuButton::builder()
        .icon_name("list-add-symbolic")
        .tooltip_text("Add effect to chain")
        .popover(&popover)
        .valign(gtk::Align::Center)
        .build();
    menu.add_css_class("flat");
    menu
}
