//! Per-parameter editor row.
//!
//! Builds a single `AdwActionRow` (title = descriptor.name + tooltip
//! = descriptor.help) with a type-appropriate suffix widget:
//!
//! - `Float`     → `gtk::Scale` + linked `gtk::SpinButton`
//! - `Integer`   → `gtk::SpinButton`
//! - `Bool`      → `gtk::Switch`
//! - `Color`     → `gtk::ColorDialogButton`
//! - `Path`      → `gtk::Button` opening `gtk::FileDialog`
//! - `Enum`      → `gtk::DropDown::from_strings`
//!
//! Signal handlers forward changes through a `Debouncer` and the
//! supplied `Sender<AppMsg>`, so the AppModel ultimately issues a
//! `Command::Set` over the control socket.

use std::time::Duration;

use adw::prelude::*;
use fluxframe_core::{CommitStrategy, ParamDescriptor, ParamKind, Scale};
use gtk::gio;
use relm4::Sender;
use serde_json::Value;

use crate::app::AppMsg;
use crate::debounce::Debouncer;

/// Minimum horizontal pixel width for the param-row sliders.
const SLIDER_MIN_WIDTH: i32 = 180;

/// Number of discrete ticks the log-scale slider divides its
/// `[ln(min), ln(max)]` range into. 100 ticks gives one-percent
/// resolution at the visual level — fine enough for tuning, coarse
/// enough that the GTK adjustment step lands on a visible
/// quantisation.
const LOG_SLIDER_TICKS: f64 = 100.0;

/// Context shared by every param widget's signal handler.
///
/// Bundles the [`Debouncer`] queue and the AppModel input
/// [`Sender`] so the per-`ParamKind` `wire_*` helpers do not have to
/// re-thread both arguments individually. Cheap to clone — internal
/// fields are `Rc`/`Sender`-backed.
#[derive(Clone)]
pub(crate) struct DispatchCtx {
    pub(crate) debouncer: Debouncer,
    pub(crate) sender: Sender<AppMsg>,
}

/// Build an `AdwActionRow` for one parameter.
///
/// `path` is the dot-syntax `<section>.<effect>.<field>` used by the
/// `Command::Set` wire format. `initial` is the current value from
/// the active config (or `Null` if the daemon never set it — in that
/// case the widget falls back to the metadata default).
///
/// `descriptor` is taken by value (`ParamDescriptor: Copy`) so the
/// caller does not have to thread a `'static` lifetime.
#[allow(
    clippy::too_many_lines,
    reason = "single dispatch on ParamKind; splitting per-arm hurts readability"
)]
pub(crate) fn build(
    descriptor: ParamDescriptor,
    path: String,
    initial: &Value,
    ctx: DispatchCtx,
) -> adw::ActionRow {
    let row = adw::ActionRow::builder()
        .title(descriptor.name)
        .subtitle(descriptor.help)
        .build();

    match descriptor.kind {
        ParamKind::Float {
            default,
            min,
            max,
            step,
            scale,
        } => {
            let value = initial.as_f64().map_or(default, |v| v as f32);
            #[allow(
                clippy::match_same_arms,
                reason = "wildcard arm exists solely to satisfy non_exhaustive ParamKind"
            )]
            let widget = match scale {
                Scale::Linear => float_slider(
                    f64::from(min),
                    f64::from(max),
                    f64::from(step),
                    f64::from(value),
                ),
                Scale::Logarithmic => log_float_slider(min, max, value),
                // `Scale` is `#[non_exhaustive]`; fall back to linear.
                _ => float_slider(
                    f64::from(min),
                    f64::from(max),
                    f64::from(step),
                    f64::from(value),
                ),
            };
            wire_float(&widget, path, descriptor.commit, scale, ctx);
            row.add_suffix(&widget);
        }
        ParamKind::Integer {
            default,
            min,
            max,
            step,
            ..
        } => {
            let value = initial.as_i64().unwrap_or(default);
            #[allow(
                clippy::cast_precision_loss,
                reason = "i64 range fits a slider; rounding to f64 is intentional"
            )]
            let spin = gtk::SpinButton::with_range(min as f64, max as f64, step as f64);
            spin.set_value(value as f64);
            wire_integer(&spin, path, descriptor.commit, ctx);
            row.add_suffix(&spin);
        }
        ParamKind::Bool { default } => {
            let value = initial.as_bool().unwrap_or(default);
            let switch = gtk::Switch::new();
            switch.set_active(value);
            switch.set_valign(gtk::Align::Center);
            wire_bool(&switch, path, ctx.sender);
            row.add_suffix(&switch);
        }
        ParamKind::Color { default } => {
            let rgb = initial
                .as_array()
                .and_then(|arr| {
                    if arr.len() == 3 {
                        Some([
                            arr[0]
                                .as_u64()
                                .and_then(|n| u8::try_from(n).ok())
                                .unwrap_or(default[0]),
                            arr[1]
                                .as_u64()
                                .and_then(|n| u8::try_from(n).ok())
                                .unwrap_or(default[1]),
                            arr[2]
                                .as_u64()
                                .and_then(|n| u8::try_from(n).ok())
                                .unwrap_or(default[2]),
                        ])
                    } else {
                        None
                    }
                })
                .unwrap_or(default);
            let button = color_button(rgb);
            wire_color(&button, path, ctx.sender);
            row.add_suffix(&button);
        }
        ParamKind::Path {
            default,
            required,
            extensions,
        } => {
            let initial_path = initial
                .as_str()
                .or(default)
                .unwrap_or(if required {
                    "(unset — required)"
                } else {
                    "(unset)"
                })
                .to_string();
            let button = gtk::Button::with_label(&initial_path);
            wire_path(&button, path, extensions, ctx.sender);
            row.add_suffix(&button);
        }
        ParamKind::Enum { default, variants } => {
            let initial_str = initial.as_str().unwrap_or(default);
            let drop = gtk::DropDown::from_strings(variants);
            if let Some(idx) = variants.iter().position(|v| *v == initial_str) {
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "variants slice length bounded; idx fits u32"
                )]
                drop.set_selected(idx as u32);
            } else {
                tracing::warn!(
                    path = %path,
                    value = %initial_str,
                    ?variants,
                    "enum value not in inventory variants; defaulting to first"
                );
            }
            drop.set_valign(gtk::Align::Center);
            wire_enum(&drop, variants, path, ctx.sender);
            row.add_suffix(&drop);
        }
        // `ParamKind` is `#[non_exhaustive]`; future variants render
        // as a placeholder so the chain page does not blow up on a
        // newer daemon. Log so the operator sees something is off.
        _ => {
            tracing::warn!(
                path = %path,
                "unsupported ParamKind variant — rendering an empty row"
            );
        }
    }

    row
}

/// Construct a horizontal `gtk::Scale` with sensible defaults for a
/// float param.
fn float_slider(min: f64, max: f64, step: f64, value: f64) -> gtk::Scale {
    let adj = gtk::Adjustment::new(value, min, max, step, step, 0.0);
    let scale = gtk::Scale::new(gtk::Orientation::Horizontal, Some(&adj));
    scale.set_hexpand(true);
    scale.set_size_request(SLIDER_MIN_WIDTH, -1);
    scale.set_draw_value(true);
    scale.set_value_pos(gtk::PositionType::Right);
    scale.set_digits(2);
    scale
}

/// Build a `gtk::Scale` whose adjustment tracks `ln(value)` so the
/// slider feels evenly spaced across multiplicative ranges (blur
/// radius `1..256`, etc.). The displayed value is exponentiated via
/// [`gtk::Scale::set_format_value_func`].
fn log_float_slider(min: f32, max: f32, value: f32) -> gtk::Scale {
    let safe_min = min.max(1e-6);
    let ln_min = f64::from(safe_min.ln());
    let ln_max = f64::from(max.ln());
    let ln_value = f64::from(value.max(safe_min).ln());
    // Step in log space: `LOG_SLIDER_TICKS` ticks over the range.
    let ln_step = (ln_max - ln_min) / LOG_SLIDER_TICKS;
    let adj = gtk::Adjustment::new(ln_value, ln_min, ln_max, ln_step, ln_step, 0.0);
    let scale = gtk::Scale::new(gtk::Orientation::Horizontal, Some(&adj));
    scale.set_hexpand(true);
    scale.set_size_request(SLIDER_MIN_WIDTH, -1);
    scale.set_draw_value(true);
    scale.set_value_pos(gtk::PositionType::Right);
    scale.set_format_value_func(move |_, ln_v| {
        let v = ln_v.exp();
        format!("{v:.2}")
    });
    scale
}

fn wire_float(
    scale: &gtk::Scale,
    path: String,
    commit: CommitStrategy,
    value_scale: Scale,
    ctx: DispatchCtx,
) {
    scale.connect_value_changed(move |s| {
        let value = match value_scale {
            Scale::Logarithmic => s.value().exp(),
            // `Scale` is `#[non_exhaustive]`; treat unknown variants
            // as linear (the raw slider value).
            Scale::Linear | _ => s.value(),
        };
        let json = serde_json::Number::from_f64(value).map_or(Value::Null, Value::Number);
        dispatch(commit, path.clone(), json, &ctx);
    });
}

fn wire_integer(spin: &gtk::SpinButton, path: String, commit: CommitStrategy, ctx: DispatchCtx) {
    spin.connect_value_changed(move |s| {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "spin button range constrained to fit i64"
        )]
        let value = s.value() as i64;
        let json = Value::Number(serde_json::Number::from(value));
        dispatch(commit, path.clone(), json, &ctx);
    });
}

fn wire_bool(switch: &gtk::Switch, path: String, sender: Sender<AppMsg>) {
    // `connect_active_notify` fires on every state transition including
    // programmatic ones; the Bool arm in `build` sets the initial state
    // BEFORE calling `wire_bool` so the user never sees a spurious
    // dispatch.
    switch.connect_active_notify(move |s| {
        let _ = sender.send(AppMsg::SetParam {
            path: path.clone(),
            value: Value::Bool(s.is_active()),
        });
    });
}

fn color_button(rgb: [u8; 3]) -> gtk::ColorDialogButton {
    let dialog = gtk::ColorDialog::builder()
        .with_alpha(false)
        .modal(true)
        .build();
    let button = gtk::ColorDialogButton::new(Some(dialog));
    button.set_rgba(&gtk::gdk::RGBA::new(
        f32::from(rgb[0]) / 255.0,
        f32::from(rgb[1]) / 255.0,
        f32::from(rgb[2]) / 255.0,
        1.0,
    ));
    button.set_valign(gtk::Align::Center);
    button
}

fn wire_color(button: &gtk::ColorDialogButton, path: String, sender: Sender<AppMsg>) {
    button.connect_rgba_notify(move |b| {
        let rgba = b.rgba();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "rgba clamped to [0.0, 1.0] before scaling"
        )]
        let r = (rgba.red().clamp(0.0, 1.0) * 255.0) as u8;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "rgba clamped to [0.0, 1.0] before scaling"
        )]
        let g = (rgba.green().clamp(0.0, 1.0) * 255.0) as u8;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "rgba clamped to [0.0, 1.0] before scaling"
        )]
        let b = (rgba.blue().clamp(0.0, 1.0) * 255.0) as u8;
        let arr = Value::Array(vec![
            Value::Number(serde_json::Number::from(r)),
            Value::Number(serde_json::Number::from(g)),
            Value::Number(serde_json::Number::from(b)),
        ]);
        let _ = sender.send(AppMsg::SetParam {
            path: path.clone(),
            value: arr,
        });
    });
}

/// Wire a Path-parameter button to open a [`gtk::FileDialog`] and
/// send the chosen file's path back to the daemon.
///
/// The dialog is anchored to the button's window; cancellation is a
/// no-op (no `SetParam` is sent). Extension filters are advisory —
/// the dialog still lets the user override with "All files".
fn wire_path(
    button: &gtk::Button,
    path: String,
    extensions: &'static [&'static str],
    sender: Sender<AppMsg>,
) {
    let button_handle = button.clone();
    button.connect_clicked(move |_| {
        // Pull the last dot-segment of the path (the field name) so the
        // dialog title tells the user what they're picking, instead of a
        // generic "Pick a file".
        let field_name = path.rsplit('.').next().unwrap_or("file");
        let title = format!("Pick {field_name}");
        let dialog = gtk::FileDialog::builder().title(&title).modal(true).build();
        if !extensions.is_empty() {
            let filter = gtk::FileFilter::new();
            filter.set_name(Some("Supported types"));
            for ext in extensions {
                filter.add_suffix(ext);
            }
            let filters = gio::ListStore::new::<gtk::FileFilter>();
            filters.append(&filter);
            dialog.set_filters(Some(&filters));
            dialog.set_default_filter(Some(&filter));
        }
        let window = button_handle
            .root()
            .and_then(|w| w.downcast::<gtk::Window>().ok());
        let path_for_dialog = path.clone();
        let sender_for_dialog = sender.clone();
        let button_for_dialog = button_handle.clone();
        dialog.open(
            window.as_ref(),
            gio::Cancellable::NONE,
            move |result| match result {
                Ok(file) => {
                    let Some(picked) = file.path() else {
                        tracing::warn!("FileDialog returned a file without a path");
                        return;
                    };
                    let picked_str = picked.to_string_lossy().into_owned();
                    button_for_dialog.set_label(&picked_str);
                    let _ = sender_for_dialog.send(AppMsg::SetParam {
                        path: path_for_dialog.clone(),
                        value: Value::String(picked_str),
                    });
                }
                Err(e) => {
                    // gio::IOErrorEnum::Cancelled is the normal "user
                    // dismissed the dialog" path — log at debug only.
                    // Anything else is a real I/O failure and deserves
                    // warn-level visibility.
                    if e.kind::<gio::IOErrorEnum>() == Some(gio::IOErrorEnum::Cancelled) {
                        tracing::debug!("FileDialog cancelled by user");
                    } else {
                        tracing::warn!(error = %e, "FileDialog failed");
                    }
                }
            },
        );
    });
}

fn wire_enum(
    drop: &gtk::DropDown,
    variants: &'static [&'static str],
    path: String,
    sender: Sender<AppMsg>,
) {
    drop.connect_selected_notify(move |d| {
        let idx = d.selected() as usize;
        if let Some(name) = variants.get(idx) {
            let _ = sender.send(AppMsg::SetParam {
                path: path.clone(),
                value: Value::String((*name).to_string()),
            });
        }
    });
}

/// Route a value change through the [`Debouncer`] according to the
/// parameter's commit strategy.
fn dispatch(commit: CommitStrategy, path: String, value: Value, ctx: &DispatchCtx) {
    tracing::trace!(path = %path, ?value, ?commit, "param Set dispatched");
    debug_assert_eq!(
        path.matches('.').count(),
        2,
        "Set path must be <section>.<effect>.<field>, got '{path}'"
    );
    let DispatchCtx { debouncer, sender } = ctx;
    let sender = sender.clone();
    #[allow(
        clippy::single_match_else,
        reason = "wildcard arm exists solely to satisfy non_exhaustive CommitStrategy"
    )]
    match commit {
        CommitStrategy::Live { debounce_ms } => {
            let path_for_send = path.clone();
            debouncer.schedule(
                path,
                Duration::from_millis(u64::from(debounce_ms)),
                move || {
                    let _ = sender.send(AppMsg::SetParam {
                        path: path_for_send,
                        value,
                    });
                },
            );
        }
        // `CommitStrategy` is `#[non_exhaustive]`; treat unknown
        // variants like `OnCommit` (flush immediately).
        CommitStrategy::OnCommit | CommitStrategy::Instant | _ => {
            let path_for_send = path.clone();
            debouncer.flush(&path, || {
                let _ = sender.send(AppMsg::SetParam {
                    path: path_for_send,
                    value,
                });
            });
        }
    }
}
