//! Shared helpers for plane effects: Rec.601 luminance derivation,
//! uniform range validators used by `configure()` methods, and
//! constructors for the common parameter descriptors.

use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::{CommitStrategy, ParamDescriptor, ParamKind, Scale};
use fluxframe_core::plane::{FramePlane, MaskPlane};

/// Rec.601 luminance of an 8-bit RGB triplet.
///
/// Uses the fixed-point weights `(77, 150, 29)` which sum to `256`, so
/// the final right-shift by 8 yields a value in `0..=255` without
/// further clamping. Centralised here because more than one effect
/// (currently `exposure_correct`, soon auto-WB and contrast) needs the
/// same byte-for-byte calculation, and the unit tests of those effects
/// re-derive luma in lockstep.
#[inline]
pub(crate) fn luma_rec601(rgb: [u8; 3]) -> u8 {
    let y = (u32::from(rgb[0]) * 77 + u32::from(rgb[1]) * 150 + u32::from(rgb[2]) * 29) >> 8;
    // Mathematically `y <= 255` because the weights sum to 256; the
    // `min` keeps the cast obviously sound to a reader.
    y.min(255) as u8
}

/// Reject `value` when it is not finite or falls outside `[min, max]`
/// (inclusive on both ends).
///
/// Produces an [`EffectError::InvalidConfig`] whose `reason` follows the
/// standard project phrasing (`"<field> must be in [<min>, <max>], got
/// <value>"`). Used by every plane-effect `configure()` that validates a
/// numeric field with inclusive bounds.
pub(crate) fn reject_out_of_range(
    effect: &'static str,
    field: &str,
    value: f32,
    min: f32,
    max: f32,
) -> Result<(), EffectError> {
    if !value.is_finite() {
        return Err(super::invalid_config(
            effect,
            format!("{field} must be finite, got {value}"),
        ));
    }
    if value < min || value > max {
        return Err(super::invalid_config(
            effect,
            format!("{field} must be in [{min}, {max}], got {value}"),
        ));
    }
    Ok(())
}

/// Reject an integer `value` outside `[min, max]` (inclusive), with the
/// same phrasing as [`reject_out_of_range`].
pub(crate) fn reject_int_out_of_range(
    effect: &'static str,
    field: &str,
    value: i64,
    min: i64,
    max: i64,
) -> Result<(), EffectError> {
    if value < min || value > max {
        return Err(super::invalid_config(
            effect,
            format!("{field} must be in [{min}, {max}], got {value}"),
        ));
    }
    Ok(())
}

/// Linear integer parameter with step 1, committed live.
pub(crate) const fn integer_param(
    name: &'static str,
    default: i64,
    min: i64,
    max: i64,
    debounce_ms: u32,
    help: &'static str,
) -> ParamDescriptor {
    ParamDescriptor {
        name,
        kind: ParamKind::Integer {
            default,
            min,
            max,
            step: 1,
            scale: Scale::Linear,
        },
        help,
        commit: CommitStrategy::Live { debounce_ms },
    }
}

/// Linear float parameter with step 0.05, committed live.
pub(crate) const fn float_param(
    name: &'static str,
    default: f32,
    min: f32,
    max: f32,
    debounce_ms: u32,
    help: &'static str,
) -> ParamDescriptor {
    ParamDescriptor {
        name,
        kind: ParamKind::Float {
            default,
            min,
            max,
            step: 0.05,
            scale: Scale::Linear,
        },
        help,
        commit: CommitStrategy::Live { debounce_ms },
    }
}

/// `ProcessFailed` unless `mask` has the plane's size. Effects that read
/// the mask call it first; the composite always passes matching sizes,
/// so a mismatch is a programming error surfaced as a failed frame.
pub(crate) fn check_mask_size(
    effect: &'static str,
    plane: &FramePlane<'_>,
    mask: &MaskPlane<'_>,
) -> Result<(), EffectError> {
    if (mask.width, mask.height) == (plane.width, plane.height) {
        return Ok(());
    }
    Err(EffectError::ProcessFailed {
        name: effect.to_string(),
        reason: format!(
            "mask {}x{} differs from plane {}x{}",
            mask.width, mask.height, plane.width, plane.height
        ),
    })
}
