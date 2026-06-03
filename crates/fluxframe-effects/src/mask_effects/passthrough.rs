//! Identity `MaskEffect` — leaves the mask untouched.
//!
//! Symmetric counterpart of [`crate::plane_effects::PassthroughPlaneEffect`].
//! Most operators will use an empty `[mask].chain` instead, but listing
//! `"passthrough"` explicitly is a readable way to document intent.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::EffectMetadata;
use fluxframe_core::plane::{MaskEffect, MaskPlane};
use fluxframe_core::traits::RawEffectParams;

/// Identity mask effect: `process` does nothing.
#[derive(Default)]
pub struct PassthroughMaskEffect;

impl PassthroughMaskEffect {
    /// Effect name as registered in the mask registry.
    pub const NAME: &'static str = "passthrough";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Identity mask: leaves the mask untouched.",
        params: &[],
    };

    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl MaskEffect for PassthroughMaskEffect {
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
        _mask: &mut MaskPlane<'_>,
        _ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaves_mask_untouched() {
        let mut effect = PassthroughMaskEffect::new();
        let mut data = vec![0.1, 0.5, 0.9, 1.0];
        let snapshot = data.clone();
        let mut plane = MaskPlane::new(&mut data, 2, 2);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("ok");
        assert_eq!(data, snapshot);
    }
}
