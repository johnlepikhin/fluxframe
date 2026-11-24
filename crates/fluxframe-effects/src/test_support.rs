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

/// Frame-sized mask with `value(x, y)` at each pixel.
pub(crate) fn mask_from(width: u32, height: u32, value: impl Fn(u32, u32) -> f32) -> Vec<f32> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| value(x, y))
        .collect()
}

/// Run a plane effect over a uniform `fill` plane with `mask`; returns
/// the plane.
pub(crate) fn run_masked<E: PlaneEffect + ?Sized>(
    effect: &mut E,
    mask: &mut [f32],
    width: u32,
    height: u32,
    fill: [u8; 3],
) -> Vec<u8> {
    let mut data = fill.repeat((width * height) as usize);
    let mut plane = FramePlane::new(&mut data, width, height);
    let mask = MaskPlane::new(mask, width, height);
    effect
        .process(&mut plane, &mask, &mut FrameContext::default())
        .expect("process");
    data
}
