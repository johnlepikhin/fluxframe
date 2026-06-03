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

use adw::prelude::*;
use fluxframe_core::EffectMetadata;
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
        let group = adw::PreferencesGroup::builder()
            .title(title)
            .description(description)
            .build();
        let chain = state.chain_for(section);
        let inventory_for_section = state
            .inventory
            .sections
            .get(section)
            .map_or(&[][..], Vec::as_slice);
        if chain.is_empty() {
            group.add(&placeholder_row());
        } else {
            for effect_name in &chain {
                let Some(meta) = find_effect(inventory_for_section, effect_name) else {
                    // The chain references an effect the registry no
                    // longer carries (e.g. a feature-gated effect on
                    // a slim daemon). Render a stub so the operator
                    // sees the divergence.
                    group.add(&unknown_effect_row(effect_name));
                    continue;
                };
                group.add(&effect_expander(section, meta, state, debouncer, sender));
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

fn effect_expander(
    section: &str,
    meta: &EffectMetadata,
    state: &AppState,
    debouncer: &Debouncer,
    sender: &Sender<AppMsg>,
) -> adw::ExpanderRow {
    let row = adw::ExpanderRow::builder()
        .title(meta.name)
        .subtitle(meta.help)
        .build();
    for desc in meta.params {
        let path = format!("{section}.{}.{}", meta.name, desc.name);
        let initial = state.config_field(section, meta.name, desc.name);
        let ctx = DispatchCtx {
            debouncer: debouncer.clone(),
            sender: sender.clone(),
        };
        // `ParamDescriptor: Copy` — pass by value so `param_row`
        // does not need a `'static` lifetime on the borrow.
        let param = param_row::build(*desc, path, initial, ctx);
        row.add_row(&param);
    }
    row
}
