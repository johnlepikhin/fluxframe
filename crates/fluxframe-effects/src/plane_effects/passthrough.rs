//! Identity `PlaneEffect` — leaves the plane untouched.
//!
//! Useful as an explicit "no-op" entry in `[background.chain]` /
//! `[foreground.chain]`, equivalent to leaving the chain empty but
//! more discoverable from the operator's TOML.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::plane::{FramePlane, PlaneEffect};
use fluxframe_core::traits::RawEffectParams;

/// Identity plane effect: `process` returns immediately without
/// touching the plane data.
#[derive(Default)]
pub struct PassthroughPlaneEffect;

impl PassthroughPlaneEffect {
    /// Effect name as registered in the plane registry.
    pub const NAME: &'static str = "passthrough";

    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PlaneEffect for PassthroughPlaneEffect {
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
        _plane: &mut FramePlane<'_>,
        _ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaves_plane_untouched() {
        let mut effect = PassthroughPlaneEffect::new();
        let mut data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let snapshot = data.clone();
        let mut plane = FramePlane::new(&mut data, 2, 2);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("ok");
        assert_eq!(data, snapshot);
    }
}
