//! Per-parameter editor row.
//!
//! Builds a single param row (title = descriptor.name, subtitle =
//! descriptor.help) whose shape depends on the parameter kind:
//!
//! - `Float`     → `AdwActionRow` + `gtk::Scale` suffix
//! - `Integer`   → `AdwSpinRow`
//! - `Bool`      → `AdwSwitchRow`
//! - `Color`     → `AdwActionRow` + `gtk::ColorDialogButton` suffix
//! - `Path`      → `AdwActionRow` + `gtk::Button` opening `gtk::FileDialog`
//! - `Enum`      → `AdwComboRow` over a `gtk::StringList`
//!
//! Suffix widgets on action rows are labelled by the row for
//! accessibility; the Adw spin/switch/combo rows label their control
//! natively.
//!
//! Signal handlers report a [`ParamChange`] through a `Debouncer` and
//! the supplied [`Emit`] sink. Every row sets its initial value BEFORE
//! connecting handlers so construction never emits a spurious change.
//! Each row also yields a [`ParamReset`] that puts the widget back to a
//! given value without emitting a change — used when the daemon refuses
//! a write.

use std::rc::Rc;
use std::time::Duration;

use adw::prelude::*;
use fluxframe_core::{CommitStrategy, ParamKindSchema, ParamSchema, Scale, SetPath};
use gtk::{gio, glib};
use serde_json::Value;

use crate::components::Emit;
use crate::debounce::Debouncer;

/// A parameter widget produced a new value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParamChange {
    /// Parameter the widget edits.
    pub(crate) path: SetPath,
    /// New value in wire form.
    pub(crate) value: Value,
}

/// Put a param widget back to a wire value (`Null` = the metadata
/// default) without reporting a [`ParamChange`].
pub(crate) type ParamReset = Rc<dyn Fn(&Value)>;

/// A built param row plus its reset handle.
pub(crate) struct BuiltParam {
    /// Row to add to the effect expander.
    pub(crate) row: adw::PreferencesRow,
    /// Resets the row's control; see [`ParamReset`].
    pub(crate) reset: ParamReset,
}

/// Minimum horizontal pixel width for the param-row sliders.
const SLIDER_MIN_WIDTH: i32 = 180;

/// Maximum width (in characters) of the Path button label; longer
/// paths are ellipsized at the start so the file name stays visible.
const PATH_LABEL_MAX_CHARS: i32 = 24;

/// Number of discrete ticks the log-scale slider divides its
/// `[ln(min), ln(max)]` range into. 100 ticks gives one-percent
/// resolution at the visual level — fine enough for tuning, coarse
/// enough that the GTK adjustment step lands on a visible
/// quantisation.
const LOG_SLIDER_TICKS: f64 = 100.0;

/// Decimal places shown by the log-scale slider label.
const LOG_SLIDER_DIGITS: usize = 2;

/// Lower clamp for log-scale ranges so `ln` stays finite.
const LOG_SLIDER_MIN: f32 = 1e-6;

/// Upper bound on the decimal places derived from a float `step`.
const MAX_SLIDER_DIGITS: i32 = 6;

/// Context shared by every param widget's signal handler.
///
/// Bundles the [`Debouncer`] queue and the change sink so the
/// per-`ParamKind` `wire_*` helpers do not have to re-thread both
/// arguments individually. Cheap to clone — both fields are `Rc`-backed.
#[derive(Clone)]
pub(crate) struct DispatchCtx {
    pub(crate) debouncer: Debouncer,
    pub(crate) emit: Emit<ParamChange>,
}

/// Build a param row for one parameter.
///
/// `path` addresses the parameter for the `Command::Set` wire format.
/// `initial` is the current value from the active config (or `Null`
/// if the daemon never set it — in that case the widget falls back to
/// the metadata default).
#[allow(
    clippy::too_many_lines,
    reason = "single dispatch on ParamKind; splitting per-arm hurts readability"
)]
pub(crate) fn build(
    descriptor: &ParamSchema,
    path: SetPath,
    initial: &Value,
    ctx: DispatchCtx,
) -> BuiltParam {
    match descriptor.kind.clone() {
        ParamKindSchema::Float {
            default,
            min,
            max,
            step,
            scale,
        } => {
            let row = action_row(descriptor);
            let to_f32 = move |v: &Value| v.as_f64().map_or(default, |v| v as f32);
            let widget = match scale {
                Scale::Logarithmic => log_float_slider(min, max, to_f32(initial)),
                // `Scale` is `#[non_exhaustive]`; fall back to linear.
                Scale::Linear | _ => float_slider(
                    f64::from(min),
                    f64::from(max),
                    f64::from(step),
                    f64::from(to_f32(initial)),
                ),
            };
            let handler = wire_float(&widget, path, descriptor.commit, scale, ctx);
            label_suffix(&widget, &row);
            row.add_suffix(&widget);
            let reset: ParamReset = Rc::new(move |v| {
                let position = match scale {
                    Scale::Logarithmic => log_position(min, to_f32(v)),
                    Scale::Linear | _ => f64::from(to_f32(v)),
                };
                widget.block_signal(&handler);
                widget.set_value(position);
                widget.unblock_signal(&handler);
            });
            BuiltParam {
                row: row.upcast(),
                reset,
            }
        }
        ParamKindSchema::Integer {
            default,
            min,
            max,
            step,
            ..
        } => {
            #[allow(
                clippy::cast_precision_loss,
                reason = "i64 range fits a spin row; rounding to f64 is intentional"
            )]
            let to_f64 = move |v: &Value| v.as_i64().unwrap_or(default) as f64;
            #[allow(
                clippy::cast_precision_loss,
                reason = "i64 range fits a spin row; rounding to f64 is intentional"
            )]
            let adjustment = gtk::Adjustment::new(
                to_f64(initial),
                min as f64,
                max as f64,
                step as f64,
                step as f64,
                0.0,
            );
            let spin = adw::SpinRow::builder()
                .title(descriptor.name.as_str())
                .subtitle(descriptor.help.as_str())
                .adjustment(&adjustment)
                .digits(0)
                .build();
            let handler = wire_integer(&spin, path, descriptor.commit, ctx);
            let widget = spin.clone();
            let reset: ParamReset = Rc::new(move |v| {
                widget.block_signal(&handler);
                widget.set_value(to_f64(v));
                widget.unblock_signal(&handler);
            });
            BuiltParam {
                row: spin.upcast(),
                reset,
            }
        }
        ParamKindSchema::Bool { default } => {
            let to_bool = move |v: &Value| v.as_bool().unwrap_or(default);
            let switch = adw::SwitchRow::builder()
                .title(descriptor.name.as_str())
                .subtitle(descriptor.help.as_str())
                .active(to_bool(initial))
                .build();
            let handler = wire_bool(&switch, path, ctx.emit);
            let widget = switch.clone();
            let reset: ParamReset = Rc::new(move |v| {
                widget.block_signal(&handler);
                widget.set_active(to_bool(v));
                widget.unblock_signal(&handler);
            });
            BuiltParam {
                row: switch.upcast(),
                reset,
            }
        }
        ParamKindSchema::Color { default } => {
            let row = action_row(descriptor);
            let button = color_button(rgb_from_json(initial, default));
            let handler = wire_color(&button, path, ctx.emit);
            label_suffix(&button, &row);
            row.add_suffix(&button);
            let reset: ParamReset = Rc::new(move |v| {
                button.block_signal(&handler);
                button.set_rgba(&rgb_to_rgba(rgb_from_json(v, default)));
                button.unblock_signal(&handler);
            });
            BuiltParam {
                row: row.upcast(),
                reset,
            }
        }
        ParamKindSchema::Path {
            default,
            required,
            extensions,
        } => {
            let row = action_row(descriptor);
            let display_path = move |v: &Value| {
                v.as_str()
                    .or(default.as_deref())
                    .unwrap_or(if required {
                        "(unset — required)"
                    } else {
                        "(unset)"
                    })
                    .to_string()
            };
            let button = path_button(&display_path(initial));
            wire_path(&button, path, extensions, ctx.emit);
            label_suffix(&button, &row);
            row.add_suffix(&button);
            // The path button only emits from its dialog callback, so
            // there is no handler to block.
            let reset: ParamReset = Rc::new(move |v| set_path_label(&button, &display_path(v)));
            BuiltParam {
                row: row.upcast(),
                reset,
            }
        }
        ParamKindSchema::Enum { default, variants } => {
            let model =
                gtk::StringList::new(&variants.iter().map(String::as_str).collect::<Vec<_>>());
            let combo = adw::ComboRow::builder()
                .title(descriptor.name.as_str())
                .subtitle(descriptor.help.as_str())
                .model(&model)
                .build();
            let position = {
                let variants = variants.clone();
                move |v: &Value| {
                    let wanted = v.as_str().unwrap_or(&default);
                    let idx = variants.iter().position(|name| name == wanted);
                    if idx.is_none() {
                        tracing::warn!(
                            value = %wanted,
                            ?variants,
                            "enum value not in inventory variants; defaulting to first"
                        );
                    }
                    #[allow(
                        clippy::cast_possible_truncation,
                        reason = "variants slice length bounded; idx fits u32"
                    )]
                    idx.map_or(0, |i| i as u32)
                }
            };
            combo.set_selected(position(initial));
            let handler = wire_enum(&combo, variants, path, ctx.emit);
            let widget = combo.clone();
            let reset: ParamReset = Rc::new(move |v| {
                widget.block_signal(&handler);
                widget.set_selected(position(v));
                widget.unblock_signal(&handler);
            });
            BuiltParam {
                row: combo.upcast(),
                reset,
            }
        }
        // `ParamKind` is `#[non_exhaustive]`; future variants render
        // as a placeholder so the chain page does not blow up on a
        // newer daemon. Log so the operator sees something is off.
        _ => {
            tracing::warn!(
                path = %path,
                "unsupported ParamKind variant — rendering an empty row"
            );
            BuiltParam {
                row: action_row(descriptor).upcast(),
                reset: Rc::new(|_| {}),
            }
        }
    }
}

/// Plain action row carrying the descriptor's title and help text.
fn action_row(descriptor: &ParamSchema) -> adw::ActionRow {
    adw::ActionRow::builder()
        .title(descriptor.name.as_str())
        .subtitle(descriptor.help.as_str())
        .build()
}

/// Associate a suffix widget with its row title so screen readers
/// announce the parameter name together with the control.
fn label_suffix(widget: &impl IsA<gtk::Accessible>, row: &adw::ActionRow) {
    widget.update_relation(&[gtk::accessible::Relation::LabelledBy(&[row.upcast_ref()])]);
}

/// Format a slider value with `digits` decimal places.
fn format_slider_value(v: f64, digits: usize) -> String {
    format!("{v:.digits$}")
}

/// Decimal places needed to represent multiples of `step` (e.g. `0.05`
/// → 2, `0.001` → 3, `1.0` → 0). Non-positive steps fall back to 2.
fn digits_for_step(step: f64) -> i32 {
    if step <= 0.0 || !step.is_finite() {
        return 2;
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "clamped to 0..=MAX_SLIDER_DIGITS before the cast"
    )]
    let digits = (-step.log10())
        .ceil()
        .clamp(0.0, f64::from(MAX_SLIDER_DIGITS)) as i32;
    digits
}

/// Horizontal slider with the layout shared by linear and log sliders.
fn styled_scale(adj: &gtk::Adjustment) -> gtk::Scale {
    let scale = gtk::Scale::new(gtk::Orientation::Horizontal, Some(adj));
    scale.set_hexpand(true);
    scale.set_valign(gtk::Align::Center);
    scale.set_size_request(SLIDER_MIN_WIDTH, -1);
    scale.set_draw_value(true);
    scale.set_value_pos(gtk::PositionType::Right);
    scale
}

/// Construct a horizontal `gtk::Scale` for a linear float param. The
/// displayed precision (and GTK's value rounding) follows `step`.
fn float_slider(min: f64, max: f64, step: f64, value: f64) -> gtk::Scale {
    let adj = gtk::Adjustment::new(value, min, max, step, step, 0.0);
    let scale = styled_scale(&adj);
    let digits = digits_for_step(step);
    // `digits` also drives value rounding while `draw-value` is on.
    scale.set_digits(digits);
    let label_digits = usize::try_from(digits).unwrap_or(2);
    scale.set_format_value_func(move |_, v| format_slider_value(v, label_digits));
    scale
}

/// Slider position (`ln` space) for `value` on a log slider over
/// `[min, ..]`.
fn log_position(min: f32, value: f32) -> f64 {
    f64::from(value.max(min.max(LOG_SLIDER_MIN)).ln())
}

/// Build a `gtk::Scale` whose adjustment tracks `ln(value)` so the
/// slider feels evenly spaced across multiplicative ranges (blur
/// radius `1..256`, etc.). The displayed and accessible values are
/// exponentiated.
fn log_float_slider(min: f32, max: f32, value: f32) -> gtk::Scale {
    let ln_min = log_position(min, min);
    let ln_max = f64::from(max.ln());
    // Step in log space: `LOG_SLIDER_TICKS` ticks over the range.
    let ln_step = (ln_max - ln_min) / LOG_SLIDER_TICKS;
    let adj = gtk::Adjustment::new(
        log_position(min, value),
        ln_min,
        ln_max,
        ln_step,
        ln_step,
        0.0,
    );
    let scale = styled_scale(&adj);
    // `GtkScale:digits` defaults to 1 and, with `draw-value` on, rounds
    // the adjustment to it — in ln space that would skip every other
    // tick. The label comes from the format func, so disable rounding.
    scale.set_round_digits(-1);
    scale.set_format_value_func(|_, ln_v| format_slider_value(ln_v.exp(), LOG_SLIDER_DIGITS));
    // Screen readers read the adjustment (ln space); publish the real
    // value as text instead.
    update_log_value_text(&scale);
    scale.connect_value_changed(update_log_value_text);
    scale
}

fn update_log_value_text(scale: &gtk::Scale) {
    let text = format_slider_value(scale.value().exp(), LOG_SLIDER_DIGITS);
    scale.update_property(&[gtk::accessible::Property::ValueText(&text)]);
}

fn wire_float(
    scale: &gtk::Scale,
    path: SetPath,
    commit: CommitStrategy,
    value_scale: Scale,
    ctx: DispatchCtx,
) -> glib::SignalHandlerId {
    scale.connect_value_changed(move |s| {
        let value = match value_scale {
            Scale::Logarithmic => s.value().exp(),
            // `Scale` is `#[non_exhaustive]`; treat unknown variants
            // as linear (the raw slider value).
            Scale::Linear | _ => s.value(),
        };
        let json = serde_json::Number::from_f64(value).map_or(Value::Null, Value::Number);
        dispatch(commit, &path, json, &ctx);
    })
}

fn wire_integer(
    spin: &adw::SpinRow,
    path: SetPath,
    commit: CommitStrategy,
    ctx: DispatchCtx,
) -> glib::SignalHandlerId {
    spin.connect_value_notify(move |s| {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "spin row range constrained to fit i64"
        )]
        let value = s.value() as i64;
        let json = Value::Number(serde_json::Number::from(value));
        dispatch(commit, &path, json, &ctx);
    })
}

fn wire_bool(
    switch: &adw::SwitchRow,
    path: SetPath,
    emit: Emit<ParamChange>,
) -> glib::SignalHandlerId {
    // `connect_active_notify` fires on every state transition including
    // programmatic ones; the Bool arm in `build` sets the initial state
    // BEFORE calling `wire_bool` so the user never sees a spurious
    // dispatch.
    switch.connect_active_notify(move |s| {
        emit(ParamChange {
            path: path.clone(),
            value: Value::Bool(s.is_active()),
        });
    })
}

/// Decode a wire `[r, g, b]` array, falling back to `default` for a
/// malformed value or channel.
fn rgb_from_json(value: &Value, default: [u8; 3]) -> [u8; 3] {
    let Some(arr) = value.as_array().filter(|arr| arr.len() == 3) else {
        return default;
    };
    let mut rgb = default;
    for (channel, item) in rgb.iter_mut().zip(arr) {
        if let Some(n) = item.as_u64().and_then(|n| u8::try_from(n).ok()) {
            *channel = n;
        }
    }
    rgb
}

fn rgb_to_rgba(rgb: [u8; 3]) -> gtk::gdk::RGBA {
    gtk::gdk::RGBA::new(
        f32::from(rgb[0]) / 255.0,
        f32::from(rgb[1]) / 255.0,
        f32::from(rgb[2]) / 255.0,
        1.0,
    )
}

/// Encode an RGBA colour as the wire `[r, g, b]` array.
fn rgba_to_json(rgba: &gtk::gdk::RGBA) -> Value {
    let channel = |c: f32| {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "channel clamped to [0.0, 1.0] before scaling"
        )]
        let byte = (c.clamp(0.0, 1.0) * 255.0) as u8;
        Value::Number(serde_json::Number::from(byte))
    };
    Value::Array(vec![
        channel(rgba.red()),
        channel(rgba.green()),
        channel(rgba.blue()),
    ])
}

fn color_button(rgb: [u8; 3]) -> gtk::ColorDialogButton {
    let dialog = gtk::ColorDialog::builder()
        .with_alpha(false)
        .modal(true)
        .build();
    let button = gtk::ColorDialogButton::new(Some(dialog));
    button.set_rgba(&rgb_to_rgba(rgb));
    button.set_valign(gtk::Align::Center);
    button
}

fn wire_color(
    button: &gtk::ColorDialogButton,
    path: SetPath,
    emit: Emit<ParamChange>,
) -> glib::SignalHandlerId {
    button.connect_rgba_notify(move |b| {
        emit(ParamChange {
            path: path.clone(),
            value: rgba_to_json(&b.rgba()),
        });
    })
}

/// Build the Path-parameter button: a start-ellipsized label so long
/// paths never force the window's minimum width, with the full path
/// in the tooltip.
fn path_button(initial_path: &str) -> gtk::Button {
    let label = gtk::Label::builder()
        .label(initial_path)
        .ellipsize(gtk::pango::EllipsizeMode::Start)
        .max_width_chars(PATH_LABEL_MAX_CHARS)
        .build();
    gtk::Button::builder()
        .child(&label)
        .tooltip_text(initial_path)
        .valign(gtk::Align::Center)
        .build()
}

/// Show `text` on a [`path_button`] and in its tooltip.
fn set_path_label(button: &gtk::Button, text: &str) {
    if let Some(label) = button.child().and_then(|c| c.downcast::<gtk::Label>().ok()) {
        label.set_label(text);
    }
    button.set_tooltip_text(Some(text));
}

/// Wire a Path-parameter button to open a [`gtk::FileDialog`] and
/// send the chosen file's path back to the daemon.
///
/// The dialog is anchored to the button's window; cancellation is a
/// no-op (no `SetParam` is sent). Extension filters are advisory —
/// the dialog still lets the user override with "All files".
///
/// The click handler uses its own button argument and the dialog
/// callback holds the button weakly, so no handler keeps its own
/// widget alive. A pick that lands after the chain page was rebuilt
/// is dropped on purpose, like the debounced sends of the old page.
fn wire_path(
    button: &gtk::Button,
    path: SetPath,
    extensions: Vec<String>,
    emit: Emit<ParamChange>,
) {
    button.connect_clicked(move |button| {
        // Name the field in the dialog title so the user knows what
        // they're picking, instead of a generic "Choose a file".
        let title = format!("Choose {}", path.field.replace('_', " "));
        let dialog = gtk::FileDialog::builder().title(&title).modal(true).build();
        if !extensions.is_empty() {
            let filter = gtk::FileFilter::new();
            filter.set_name(Some("Supported types"));
            for ext in &extensions {
                filter.add_suffix(ext);
            }
            let filters = gio::ListStore::new::<gtk::FileFilter>();
            filters.append(&filter);
            dialog.set_filters(Some(&filters));
            dialog.set_default_filter(Some(&filter));
        }
        let window = button.root().and_then(|w| w.downcast::<gtk::Window>().ok());
        let path_for_dialog = path.clone();
        let emit_for_dialog = Emit::clone(&emit);
        dialog.open(
            window.as_ref(),
            gio::Cancellable::NONE,
            glib::clone!(
                #[weak]
                button,
                #[upgrade_or_else]
                || tracing::debug!("path row rebuilt while the file dialog was open; pick dropped"),
                move |result| match result {
                    Ok(file) => {
                        let Some(picked) = file.path() else {
                            tracing::warn!("FileDialog returned a file without a path");
                            return;
                        };
                        let picked_str = picked.to_string_lossy().into_owned();
                        set_path_label(&button, &picked_str);
                        emit_for_dialog(ParamChange {
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
                }
            ),
        );
    });
}

fn wire_enum(
    combo: &adw::ComboRow,
    variants: Vec<String>,
    path: SetPath,
    emit: Emit<ParamChange>,
) -> glib::SignalHandlerId {
    combo.connect_selected_notify(move |c| {
        let idx = c.selected() as usize;
        if let Some(name) = variants.get(idx) {
            emit(ParamChange {
                path: path.clone(),
                value: Value::String(name.clone()),
            });
        }
    })
}

/// Route a value change through the [`Debouncer`] according to the
/// parameter's commit strategy.
fn dispatch(commit: CommitStrategy, path: &SetPath, value: Value, ctx: &DispatchCtx) {
    tracing::trace!(path = %path, ?value, ?commit, "param Set dispatched");
    let DispatchCtx { debouncer, emit } = ctx;
    let emit = Emit::clone(emit);
    let change = ParamChange {
        path: path.clone(),
        value,
    };
    #[allow(
        clippy::single_match_else,
        reason = "wildcard arm exists solely to satisfy non_exhaustive CommitStrategy"
    )]
    match commit {
        CommitStrategy::Live { debounce_ms } => {
            debouncer.schedule(
                path.to_string(),
                Duration::from_millis(u64::from(debounce_ms)),
                move || emit(change),
            );
        }
        // `CommitStrategy` is `#[non_exhaustive]`; treat unknown
        // variants like `OnCommit` (flush immediately).
        CommitStrategy::OnCommit | CommitStrategy::Instant | _ => {
            debouncer.flush(&path.to_string(), || emit(change));
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn digits_follow_step_precision() {
        assert_eq!(digits_for_step(1.0), 0);
        assert_eq!(digits_for_step(0.5), 1);
        assert_eq!(digits_for_step(0.05), 2);
        assert_eq!(digits_for_step(0.001), 3);
        assert_eq!(digits_for_step(1e-9), MAX_SLIDER_DIGITS);
        assert_eq!(digits_for_step(0.0), 2);
    }

    #[test]
    fn rgb_from_json_falls_back_per_channel_and_shape() {
        let default = [1, 2, 3];
        assert_eq!(rgb_from_json(&json!([10, 20, 30]), default), [10, 20, 30]);
        assert_eq!(rgb_from_json(&json!([10, 999, "x"]), default), [10, 2, 3]);
        assert_eq!(rgb_from_json(&json!([10, 20]), default), default);
        assert_eq!(rgb_from_json(&Value::Null, default), default);
    }

    #[test]
    fn log_position_clamps_below_min() {
        assert!((log_position(1.0, 0.0) - 0.0).abs() < 1e-9);
        assert!((log_position(1.0, std::f32::consts::E) - 1.0).abs() < 1e-6);
    }
}
