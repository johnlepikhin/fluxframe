//! Horizontal flip post effect. Swaps each row's pixels left ↔ right
//! on the already-composited frame.
//!
//! Lives in the post-composite chain (not background/foreground) so
//! mask, foreground subject, and background fill are all flipped as a
//! single unit — applying mirror in a plane sub-chain would flip one
//! side of the composite without the other and produce a torn image.
//!
//! Counters the self-view mirroring that conference apps apply, so
//! the operator sees text and UI in the scene right-way-round.
//!
//! Operates on packed RGB only (the composite path guarantees that by
//! the time a post effect runs). Odd-width rows leave the centre
//! pixel untouched, matching the algebraic identity of a horizontal
//! reflection.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::EffectMetadata;
use fluxframe_core::plane::{FramePlane, MaskPlane, PostEffect};
use fluxframe_core::traits::RawEffectParams;

/// In-place horizontal flip of the composited RGB frame.
///
/// Stateless: no `configure` parameters, no scratch buffer. `process`
/// runs `O(W·H)` time with zero heap allocations. The mask is ignored
/// — the flip is geometric, not mask-aware.
#[derive(Debug, Default)]
pub struct MirrorEffect;

impl MirrorEffect {
    /// Canonical (snake_case) name used by the registry and CLI.
    pub const NAME: &'static str = "mirror";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Flip the composited frame horizontally (left ↔ right). \
               Use to un-mirror the self-view in conference apps.",
        params: &[],
    };

    /// Construct a mirror effect ready to use — no configuration
    /// required.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PostEffect for MirrorEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, _params: RawEffectParams) -> Result<(), EffectError> {
        Ok(())
    }

    fn prepare(&mut self, _context: &ProcessingContext) -> Result<(), EffectError> {
        Ok(())
    }

    fn process(
        &mut self,
        plane: &mut FramePlane<'_>,
        _mask: &MaskPlane<'_>,
        _context: &mut FrameContext,
    ) -> Result<(), EffectError> {
        // Local guard against a future in-crate `FramePlane` literal
        // that bypasses `FramePlane::new`'s length assertion — a
        // mismatched buffer would let `chunks_exact_mut` silently
        // drop the tail row (no panic, no diagnostic) and produce a
        // partial flip.
        debug_assert_eq!(
            plane.data.len(),
            (plane.width as usize) * (plane.height as usize) * 3,
            "FramePlane invariant: data.len() == width * height * 3",
        );
        let width = plane.width as usize;
        if width < 2 {
            return Ok(());
        }
        let row_bytes = width * 3;
        let last_pixel_offset = (width - 1) * 3;
        for row in plane.data.chunks_exact_mut(row_bytes) {
            let mut left: usize = 0;
            let mut right: usize = last_pixel_offset;
            while left < right {
                row.swap(left, right);
                row.swap(left + 1, right + 1);
                row.swap(left + 2, right + 2);
                left += 3;
                right -= 3;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an all-1.0 mask plane matching the given dimensions.
    /// Mirror ignores the mask, but the trait still needs one.
    fn full_mask(buf: &mut Vec<f32>, width: u32, height: u32) -> MaskPlane<'_> {
        buf.clear();
        buf.resize((width as usize) * (height as usize), 1.0);
        MaskPlane::new(buf, width, height)
    }

    /// Even-width row: every pixel has a partner; whole row reverses.
    #[test]
    fn even_width_row_reverses() {
        // 4×1 RGB: R, G, B, Y.
        let mut data: Vec<u8> = vec![
            255, 0, 0, // R
            0, 255, 0, // G
            0, 0, 255, // B
            255, 255, 0, // Y
        ];
        let mut mask_buf = Vec::new();
        let mask = full_mask(&mut mask_buf, 4, 1);
        let mut plane = FramePlane::new(&mut data, 4, 1);
        let mut ctx = FrameContext::default();
        MirrorEffect::new()
            .process(&mut plane, &mask, &mut ctx)
            .expect("mirror process is infallible");
        assert_eq!(
            data,
            vec![
                // RGB triplets:
                255, 255, 0, // Y
                0, 0, 255, // B
                0, 255, 0, // G
                255, 0, 0, // R
            ],
        );
    }

    /// Minimum non-trivial width: one inner-loop iteration, then exit.
    /// Off-by-one boundary on the `left += 3; right -= 3` advance.
    #[test]
    fn width_two_swaps_once() {
        let mut data: Vec<u8> = vec![
            255, 0, 0, // R
            0, 255, 0, // G
        ];
        let mut mask_buf = Vec::new();
        let mask = full_mask(&mut mask_buf, 2, 1);
        let mut plane = FramePlane::new(&mut data, 2, 1);
        let mut ctx = FrameContext::default();
        MirrorEffect::new()
            .process(&mut plane, &mask, &mut ctx)
            .expect("mirror process is infallible");
        assert_eq!(
            data,
            vec![
                0, 255, 0, // G
                255, 0, 0, // R
            ],
        );
    }

    /// A 1×1 frame has no pixel to swap with — output equals input.
    #[test]
    fn single_pixel_is_identity() {
        let mut data: Vec<u8> = vec![42u8, 17, 99];
        let snapshot = data.clone();
        let mut mask_buf = Vec::new();
        let mask = full_mask(&mut mask_buf, 1, 1);
        let mut plane = FramePlane::new(&mut data, 1, 1);
        let mut ctx = FrameContext::default();
        MirrorEffect::new()
            .process(&mut plane, &mask, &mut ctx)
            .expect("mirror process is infallible");
        assert_eq!(data, snapshot);
    }

    /// Odd-width row: outer pixels swap, centre pixel keeps its place.
    #[test]
    fn odd_width_keeps_centre_pixel() {
        let mut data: Vec<u8> = vec![
            255, 0, 0, // R
            0, 255, 0, // G (centre)
            0, 0, 255, // B
        ];
        let mut mask_buf = Vec::new();
        let mask = full_mask(&mut mask_buf, 3, 1);
        let mut plane = FramePlane::new(&mut data, 3, 1);
        let mut ctx = FrameContext::default();
        MirrorEffect::new()
            .process(&mut plane, &mask, &mut ctx)
            .expect("mirror process is infallible");
        assert_eq!(
            data,
            vec![
                0, 0, 255, // B
                0, 255, 0, // G (untouched)
                255, 0, 0, // R
            ],
        );
    }

    /// Multi-row sanity: each row flips independently.
    #[test]
    fn multi_row_each_row_flips_independently() {
        // 2×2: row0=[R, B], row1=[G, Y] → row0=[B, R], row1=[Y, G].
        let mut data: Vec<u8> = vec![
            255, 0, 0, // R
            0, 0, 255, // B
            0, 255, 0, // G
            255, 255, 0, // Y
        ];
        let mut mask_buf = Vec::new();
        let mask = full_mask(&mut mask_buf, 2, 2);
        let mut plane = FramePlane::new(&mut data, 2, 2);
        let mut ctx = FrameContext::default();
        MirrorEffect::new()
            .process(&mut plane, &mask, &mut ctx)
            .expect("mirror process is infallible");
        assert_eq!(
            data,
            vec![
                0, 0, 255, // B
                255, 0, 0, // R
                255, 255, 0, // Y
                0, 255, 0, // G
            ],
        );
    }
}
