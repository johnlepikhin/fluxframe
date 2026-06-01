//! Shared helpers for plane effects: Rec.601 luminance derivation and
//! a uniform finite/range validator used by `configure()` methods.

use fluxframe_core::error::EffectError;

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
