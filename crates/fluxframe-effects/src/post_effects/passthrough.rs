//! No-op post effect. Useful as a smoke-test default for an empty
//! `[presets.NAME.post]` chain — the composite runs the chain
//! unconditionally even when it contains only a passthrough.

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::EffectMetadata;
use fluxframe_core::plane::{FramePlane, MaskPlane, PostEffect};
use fluxframe_core::traits::RawEffectParams;

/// No-op post effect.
#[derive(Debug, Default)]
pub struct PassthroughPostEffect;

impl PassthroughPostEffect {
    /// Canonical (snake_case) name used by the registry and CLI.
    pub const NAME: &'static str = "passthrough";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Identity post-effect: leaves the composited frame untouched.",
        params: &[],
    };

    /// Construct a fresh passthrough effect.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PostEffect for PassthroughPostEffect {
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
        _mask: &MaskPlane<'_>,
        _context: &mut FrameContext,
    ) -> Result<(), EffectError> {
        Ok(())
    }
}
