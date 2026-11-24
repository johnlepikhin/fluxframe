//! Helpers shared by the unit tests of the effect modules.

use fluxframe_core::context::FrameContext;
use fluxframe_core::error::EffectError;
use fluxframe_core::plane::{FramePlane, MaskPlane, PlaneEffect};

/// Fill `buf` with `value` at `width × height` and view it as a mask.
pub(crate) fn filled_mask(
    buf: &mut Vec<f32>,
    width: u32,
    height: u32,
    value: f32,
) -> MaskPlane<'_> {
    buf.clear();
    buf.resize((width as usize) * (height as usize), value);
    MaskPlane::new(buf, width, height)
}

/// Run a plane effect with a mask that marks the whole plane as
/// background — the case for effects that ignore the mask.
pub(crate) fn process_plane<E: PlaneEffect + ?Sized>(
    effect: &mut E,
    plane: &mut FramePlane<'_>,
    ctx: &mut FrameContext,
) -> Result<(), EffectError> {
    let mut buf = Vec::new();
    let mask = filled_mask(&mut buf, plane.width, plane.height, 0.0);
    effect.process(plane, &mask, ctx)
}
