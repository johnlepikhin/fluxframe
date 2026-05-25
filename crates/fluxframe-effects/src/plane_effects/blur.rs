//! Box-blur `PlaneEffect`. Wraps [`crate::backend::BlurBackend`] so the
//! same Auto / wgpu / CPU selection that drove the legacy
//! `background_blur` effect works inside the new composite pipeline.

use std::sync::Arc;

use fluxframe_core::Counters;
use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::plane::{FramePlane, PlaneEffect};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

use crate::backend::{BackendOverrides, BlurBackend, build_blur_backend};
use crate::processing::{resize_rgb_bilinear, resize_rgb_nearest};

const MAX_BLUR_RADIUS: u32 = 256;
const MAX_BLUR_PASSES: u32 = 16;
const MAX_BLUR_DOWNSCALE: u32 = 8;

/// TOML schema:
///
/// ```toml
/// [background.blur]
/// radius    = 20    # box-blur half-kernel radius (frame pixels)
/// passes    = 2     # number of box passes
/// downscale = 4     # blur at 1/N resolution; 1 disables downscale
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlurConfig {
    /// Box-blur half-kernel radius in *frame* pixels. Auto-scaled
    /// when `downscale > 1`.
    #[serde(default = "default_radius")]
    pub radius: u32,
    /// Number of box-blur passes. ≥ 2 approximates a Gaussian.
    #[serde(default = "default_passes")]
    pub passes: u32,
    /// Downscale factor for the blur pipeline. The frame is bilinear-
    /// downscaled, blurred at the smaller resolution and
    /// nearest-upscaled back. `1` disables the downscale path.
    #[serde(default = "default_downscale")]
    pub downscale: u32,
}

fn default_radius() -> u32 {
    20
}
fn default_passes() -> u32 {
    2
}
fn default_downscale() -> u32 {
    4
}

impl Default for BlurConfig {
    fn default() -> Self {
        Self {
            radius: default_radius(),
            passes: default_passes(),
            downscale: default_downscale(),
        }
    }
}

/// Plane-effect wrapper around `BlurBackend`.
pub struct BlurPlaneEffect {
    config: BlurConfig,
    backend: Option<Box<dyn BlurBackend + Send>>,
    counters: Option<Arc<Counters>>,
    // Scratch buffers (frame resolution):
    /// Full-resolution destination of the blur — what we then copy
    /// back into `plane.data`. Allocated once in `prepare`.
    blurred: Vec<u8>,
    // Downscale scratch (only used when downscale > 1):
    frame_down: Vec<u8>,
    blurred_down: Vec<u8>,
    // Negotiated dimensions:
    frame_w: u32,
    frame_h: u32,
    /// Effective downscale factor — clamped so the downscaled image
    /// is at least 2×2 (box-blur indexes `w-1`/`h-1`).
    downscale_effective: u32,
    blur_down_w: u32,
    blur_down_h: u32,
}

impl BlurPlaneEffect {
    /// Effect name as registered in the plane registry.
    pub const NAME: &'static str = "blur";

    /// Construct with defaults; configure/prepare before use.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: BlurConfig::default(),
            backend: None,
            counters: None,
            blurred: Vec::new(),
            frame_down: Vec::new(),
            blurred_down: Vec::new(),
            frame_w: 0,
            frame_h: 0,
            downscale_effective: 1,
            blur_down_w: 0,
            blur_down_h: 0,
        }
    }
}

impl Default for BlurPlaneEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl PlaneEffect for BlurPlaneEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: BlurConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        if cfg.radius > MAX_BLUR_RADIUS {
            return Err(super::invalid_config(
                Self::NAME,
                format!("radius must be <= {MAX_BLUR_RADIUS}, got {}", cfg.radius),
            ));
        }
        if cfg.passes > MAX_BLUR_PASSES {
            return Err(super::invalid_config(
                Self::NAME,
                format!("passes must be <= {MAX_BLUR_PASSES}, got {}", cfg.passes),
            ));
        }
        if cfg.downscale == 0 || cfg.downscale > MAX_BLUR_DOWNSCALE {
            return Err(super::invalid_config(
                Self::NAME,
                format!(
                    "downscale must be in 1..={MAX_BLUR_DOWNSCALE}, got {}",
                    cfg.downscale
                ),
            ));
        }
        self.config = cfg;
        Ok(())
    }

    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError> {
        self.frame_w = context.width;
        self.frame_h = context.height;
        self.counters.clone_from(&context.counters);
        let frame_bytes = (context.width as usize) * (context.height as usize) * 3;
        self.blurred = vec![0u8; frame_bytes];

        let max_downscale = (context.width.min(context.height) / 2).max(1);
        let downscale = self.config.downscale.max(1).min(max_downscale);
        self.downscale_effective = downscale;
        if downscale == 1 {
            self.blur_down_w = context.width;
            self.blur_down_h = context.height;
            self.frame_down.clear();
            self.blurred_down.clear();
        } else {
            self.blur_down_w = (context.width / downscale).max(1);
            self.blur_down_h = (context.height / downscale).max(1);
            let down_bytes = (self.blur_down_w as usize) * (self.blur_down_h as usize) * 3;
            self.frame_down = vec![0u8; down_bytes];
            self.blurred_down = vec![0u8; down_bytes];
        }

        let (prepare_w, prepare_h) = if downscale == 1 {
            (context.width, context.height)
        } else {
            (self.blur_down_w, self.blur_down_h)
        };
        let mut backend =
            build_blur_backend(BackendOverrides::current(), context.counters.clone())?;
        backend.prepare(prepare_w, prepare_h)?;
        self.backend = Some(backend);
        Ok(())
    }

    fn process(
        &mut self,
        plane: &mut FramePlane<'_>,
        _ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        if plane.width != self.frame_w || plane.height != self.frame_h {
            return Err(EffectError::ProcessFailed {
                name: Self::NAME.to_string(),
                reason: format!(
                    "plane {}x{} differs from prepared {}x{}",
                    plane.width, plane.height, self.frame_w, self.frame_h
                ),
            });
        }
        let backend = self
            .backend
            .as_mut()
            .ok_or_else(|| EffectError::ProcessFailed {
                name: Self::NAME.to_string(),
                reason: "process called before prepare".into(),
            })?;

        if self.downscale_effective <= 1 {
            backend.blur(
                plane.data,
                &mut self.blurred,
                self.frame_w,
                self.frame_h,
                self.config.radius,
                self.config.passes,
            )?;
        } else {
            resize_rgb_bilinear(
                plane.data,
                self.frame_w,
                self.frame_h,
                &mut self.frame_down,
                self.blur_down_w,
                self.blur_down_h,
            );
            let radius_down = self.config.radius.div_ceil(self.downscale_effective).max(1);
            backend.blur(
                &self.frame_down,
                &mut self.blurred_down,
                self.blur_down_w,
                self.blur_down_h,
                radius_down,
                self.config.passes,
            )?;
            resize_rgb_nearest(
                &self.blurred_down,
                self.blur_down_w,
                self.blur_down_h,
                &mut self.blurred,
                self.frame_w,
                self.frame_h,
            );
        }
        // Copy the freshly blurred plane back into the caller's
        // buffer so subsequent stages see it as the new "background".
        plane.data.copy_from_slice(&self.blurred);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configure_applies_defaults() {
        let mut effect = BlurPlaneEffect::new();
        let params: RawEffectParams = toml::Value::Table(toml::map::Map::new());
        effect.configure(params).expect("ok");
        assert_eq!(effect.config.radius, default_radius());
        assert_eq!(effect.config.passes, default_passes());
        assert_eq!(effect.config.downscale, default_downscale());
    }

    #[test]
    fn rejects_zero_downscale() {
        let mut effect = BlurPlaneEffect::new();
        let params: RawEffectParams = toml::from_str("downscale = 0").unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn rejects_excessive_radius() {
        let mut effect = BlurPlaneEffect::new();
        let raw = format!("radius = {}", MAX_BLUR_RADIUS + 1);
        let params: RawEffectParams = toml::from_str(&raw).unwrap();
        assert!(effect.configure(params).is_err());
    }

    #[test]
    fn prepare_and_process_mutate_plane() {
        let mut effect = BlurPlaneEffect::new();
        let params: RawEffectParams =
            toml::from_str("radius = 1\npasses = 1\ndownscale = 1").unwrap();
        effect.configure(params).expect("configure");

        let context = ProcessingContext {
            width: 4,
            height: 4,
            format: fluxframe_core::PixelFormat::Rgb,
            fps: 30,
            counters: None,
        };
        effect.prepare(&context).expect("prepare");

        // 4×4 RGB checkerboard so the blur produces visibly different
        // output from the input.
        let mut data = Vec::with_capacity(48);
        for y in 0..4 {
            for x in 0..4 {
                let v = if (x + y) % 2 == 0 { 255 } else { 0 };
                data.extend_from_slice(&[v, v, v]);
            }
        }
        let original = data.clone();
        let mut plane = FramePlane::new(&mut data, 4, 4);
        let mut ctx = FrameContext::default();
        effect.process(&mut plane, &mut ctx).expect("process");
        assert_ne!(data, original, "blur must change the plane");
    }
}
