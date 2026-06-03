//! Mask inversion: `value = 1 - value`.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::EffectMetadata;
use fluxframe_core::plane::{MaskEffect, MaskPlane};
use fluxframe_core::traits::RawEffectParams;

/// Inverts the mask so the operator can target the background instead
/// of the foreground. Stateless and parameter-less.
#[derive(Default)]
pub struct InvertMaskEffect;

impl InvertMaskEffect {
    /// Effect name as registered in the mask registry.
    pub const NAME: &'static str = "invert";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Invert the mask (value = 1 - value).",
        params: &[],
    };

    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl MaskEffect for InvertMaskEffect {
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
        mask: &mut MaskPlane<'_>,
        _ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        for v in mask.data.iter_mut() {
            *v = (1.0 - *v).clamp(0.0, 1.0);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverts_values() {
        let mut effect = InvertMaskEffect::new();
        let mut data = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let mut plane = MaskPlane::new(&mut data, 5, 1);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("ok");
        assert_eq!(data, vec![1.0, 0.75, 0.5, 0.25, 0.0]);
    }
}
