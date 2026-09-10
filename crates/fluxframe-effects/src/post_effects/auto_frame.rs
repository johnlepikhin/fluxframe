//! `auto_frame` post-effect: smart cropping + recentering around the
//! person detected by the segmentation mask. Compensates for badly-
//! placed laptop webcams ("person sits in the lower-right corner")
//! without requiring physical recamering.
//!
//! Algorithm:
//! 1. Walk the upscaled mask, compute the bounding box of pixels above
//!    `threshold`.
//! 2. Expand the bbox by `padding` on every side.
//! 3. EMA-smooth the (centre, half-extent) tuple across frames using
//!    `smoothing` to prevent jitter.
//! 4. Constrain the crop's aspect ratio to match the frame, clamp the
//!    minimum size by `zoom_max`, clamp the crop window to the frame
//!    bounds.
//! 5. Copy the crop region into a scratch buffer, then bilinear-upscale
//!    that scratch back into the frame.
//!
//! Cost per frame: O(width × height) — one mask scan + one crop copy +
//! one bilinear resize. The mask scan and copy are tight integer loops;
//! the resize matches the cost of `image_fill`'s startup resize but
//! runs every frame. Profile at 640×480: ~1-2 ms (single-thread CPU).

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::{
    CommitStrategy, DEBOUNCE_STANDARD_MS, EffectMetadata, ParamDescriptor, ParamKind, Scale,
};
use fluxframe_core::plane::{FramePlane, MaskPlane, PostEffect};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

use crate::plane_effects::helpers::reject_out_of_range;
use crate::processing::resize::resize_rgb_bilinear;

/// Upper bound on `zoom_max`. Beyond 4× the resize blows up pixels
/// noticeably and the framing looks fake (people lean closer to fit
/// their own face, defeating the auto-frame).
const MAX_ZOOM: f32 = 4.0;
/// Upper bound on `smoothing`. Exactly `1.0` would mean the EMA never
/// updates (state stuck at the first sample). Cap below `1.0` so the
/// operator can't accidentally freeze the crop window.
const MAX_SMOOTHING: f32 = 0.99;

/// Default mask-confidence threshold for the `threshold` field.
pub const DEFAULT_THRESHOLD: f32 = 0.5;
/// Default symmetric bbox expansion fraction for the `padding` field.
pub const DEFAULT_PADDING: f32 = 0.15;
/// Default EMA inertia for the `smoothing` field.
pub const DEFAULT_SMOOTHING: f32 = 0.85;
/// Default maximum zoom factor for the `zoom_max` field.
pub const DEFAULT_ZOOM_MAX: f32 = 1.6;

/// TOML schema:
///
/// ```toml
/// [post.auto_frame]
/// threshold = 0.5
/// padding = 0.15
/// smoothing = 0.85
/// zoom_max = 1.6
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutoFrameConfig {
    /// Mask-confidence threshold above which a pixel counts toward the
    /// bounding box. `0.5` is a sane default for a mid-feathered mask.
    /// Validated to be a finite number in `[0.0, 1.0]`.
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    /// Bbox expansion factor applied symmetrically before smoothing.
    /// `0.15` adds a 15 % margin around the silhouette so the crop
    /// does not clip hair / shoulders. Validated to be in `[0.0, 1.0]`.
    #[serde(default = "default_padding")]
    pub padding: f32,
    /// EMA inertia. `0.0` means snap to the new bbox instantly,
    /// `0.99` means almost no movement. `0.85` is the recommended
    /// default for video calls — visible re-centering on real motion
    /// but no per-frame jitter.
    #[serde(default = "default_smoothing")]
    pub smoothing: f32,
    /// Maximum zoom factor. `1.0` disables the effect (no crop).
    /// `1.6` is the recommended default — enough to fix a person
    /// sitting in the corner, not enough to feel like a face zoom-in.
    /// Validated to be in `[1.0, 4.0]`.
    #[serde(default = "default_zoom_max")]
    pub zoom_max: f32,
}

fn default_threshold() -> f32 {
    DEFAULT_THRESHOLD
}
fn default_padding() -> f32 {
    DEFAULT_PADDING
}
fn default_smoothing() -> f32 {
    DEFAULT_SMOOTHING
}
fn default_zoom_max() -> f32 {
    DEFAULT_ZOOM_MAX
}

impl Default for AutoFrameConfig {
    fn default() -> Self {
        Self {
            threshold: default_threshold(),
            padding: default_padding(),
            smoothing: default_smoothing(),
            zoom_max: default_zoom_max(),
        }
    }
}

/// Floating-point bounding box centred at `(cx, cy)` with half-extents
/// `(hw, hh)`. Stored as floats so the EMA stays smooth across frames.
#[derive(Debug, Clone, Copy)]
struct SmoothBox {
    cx: f32,
    cy: f32,
    hw: f32,
    hh: f32,
}

/// `PostEffect` performing mask-driven auto-framing.
pub struct AutoFrameEffect {
    config: AutoFrameConfig,
    state: Option<SmoothBox>,
    /// Scratch buffer holding the cropped sub-region of the frame,
    /// before the bilinear upscale back into `plane.data`. Sized to
    /// the full frame in `prepare()` because the crop window can be
    /// as large as the whole frame (no-zoom case).
    crop_buf: Vec<u8>,
    frame_w: u32,
    frame_h: u32,
}

impl AutoFrameEffect {
    /// Effect name as registered in the post-effect registry.
    pub const NAME: &'static str = "auto_frame";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Mask-driven auto-framing: crop and recenter around the subject.",
        params: &[
            ParamDescriptor {
                name: "threshold",
                kind: ParamKind::Float {
                    default: DEFAULT_THRESHOLD,
                    min: 0.0,
                    max: 1.0,
                    step: 0.05,
                    scale: Scale::Linear,
                },
                help: "Mask confidence cutoff for bbox extraction.",
                commit: CommitStrategy::Live {
                    debounce_ms: DEBOUNCE_STANDARD_MS,
                },
            },
            ParamDescriptor {
                name: "padding",
                kind: ParamKind::Float {
                    default: DEFAULT_PADDING,
                    min: 0.0,
                    max: 1.0,
                    step: 0.05,
                    scale: Scale::Linear,
                },
                help: "Symmetric bbox expansion fraction.",
                commit: CommitStrategy::Live {
                    debounce_ms: DEBOUNCE_STANDARD_MS,
                },
            },
            ParamDescriptor {
                name: "smoothing",
                kind: ParamKind::Float {
                    default: DEFAULT_SMOOTHING,
                    min: 0.0,
                    max: MAX_SMOOTHING,
                    step: 0.05,
                    scale: Scale::Linear,
                },
                help: "EMA inertia for the crop window; 0 snaps, 0.99 freezes.",
                commit: CommitStrategy::Live {
                    debounce_ms: DEBOUNCE_STANDARD_MS,
                },
            },
            ParamDescriptor {
                name: "zoom_max",
                kind: ParamKind::Float {
                    default: DEFAULT_ZOOM_MAX,
                    min: 1.0,
                    max: MAX_ZOOM,
                    step: 0.1,
                    scale: Scale::Linear,
                },
                help: "Maximum allowed zoom; 1.0 disables cropping.",
                commit: CommitStrategy::Live {
                    debounce_ms: DEBOUNCE_STANDARD_MS,
                },
            },
        ],
    };

    /// Construct with defaults. `configure()` and `prepare()` must be
    /// called before `process()`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: AutoFrameConfig::default(),
            state: None,
            crop_buf: Vec::new(),
            frame_w: 0,
            frame_h: 0,
        }
    }
}

impl Default for AutoFrameEffect {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the bounding box of mask pixels above `threshold`. Returns
/// `None` when no pixel exceeds the threshold (i.e. nobody detected).
fn extract_bbox(mask: &[f32], width: usize, height: usize, threshold: f32) -> Option<SmoothBox> {
    if width == 0 || height == 0 {
        return None;
    }
    let mut min_x = usize::MAX;
    let mut max_x = 0usize;
    let mut min_y = usize::MAX;
    let mut max_y = 0usize;
    // Per row, only the first and last hot pixel matter: `position` /
    // `rposition` are tight scans that stop early, and a fully cold
    // row costs one pass with no per-pixel branching on the min/max.
    for (y, row) in mask.chunks_exact(width).take(height).enumerate() {
        let Some(first) = row.iter().position(|&v| v > threshold) else {
            continue;
        };
        // `rposition` cannot fail once `position` succeeded.
        let last = row.iter().rposition(|&v| v > threshold).unwrap_or(first);
        min_x = min_x.min(first);
        max_x = max_x.max(last);
        min_y = min_y.min(y);
        max_y = y;
    }
    if min_x == usize::MAX {
        return None;
    }
    let cx = (min_x as f32 + max_x as f32 + 1.0) * 0.5;
    let cy = (min_y as f32 + max_y as f32 + 1.0) * 0.5;
    let hw = (max_x as f32 - min_x as f32 + 1.0) * 0.5;
    let hh = (max_y as f32 - min_y as f32 + 1.0) * 0.5;
    Some(SmoothBox { cx, cy, hw, hh })
}

/// Apply EMA smoothing to a fresh observation. `smoothing` in
/// `[0, 1)` — higher means more inertia.
fn ema_blend(prev: SmoothBox, current: SmoothBox, smoothing: f32) -> SmoothBox {
    let s = smoothing;
    let inv = 1.0 - s;
    SmoothBox {
        cx: prev.cx * s + current.cx * inv,
        cy: prev.cy * s + current.cy * inv,
        hw: prev.hw * s + current.hw * inv,
        hh: prev.hh * s + current.hh * inv,
    }
}

/// Compute the integer crop rectangle. Output `(x, y, w, h)` is
/// guaranteed to lie inside `[0, frame_w) × [0, frame_h)` and to
/// preserve the frame's aspect ratio so the subsequent resize back to
/// `frame_w × frame_h` does not distort.
fn compute_crop_rect(
    bbox: SmoothBox,
    cfg: &AutoFrameConfig,
    frame_w: u32,
    frame_h: u32,
) -> (u32, u32, u32, u32) {
    // 1. Apply padding around the smoothed bbox.
    let pad = cfg.padding;
    let hw = bbox.hw * (1.0 + pad);
    let hh = bbox.hh * (1.0 + pad);

    // 2. Constrain aspect ratio to the frame's. Expand the shorter
    //    axis until ratio matches.
    let fw = frame_w as f32;
    let fh = frame_h as f32;
    let frame_aspect = fw / fh;
    let cur_aspect = (2.0 * hw) / (2.0 * hh).max(1.0);
    let (mut hw, mut hh) = if cur_aspect > frame_aspect {
        // Wider than the frame — grow vertically.
        let new_hh = hw / frame_aspect;
        (hw, new_hh)
    } else {
        // Taller (or same) — grow horizontally.
        let new_hw = hh * frame_aspect;
        (new_hw, hh)
    };

    // 3. Enforce the minimum crop size implied by `zoom_max`. The
    //    smallest permitted crop is `frame_w / zoom_max` wide (which
    //    upscales to `frame_w` after resize).
    let min_w = fw / cfg.zoom_max;
    let min_h = fh / cfg.zoom_max;
    if 2.0 * hw < min_w {
        hw = min_w * 0.5;
    }
    if 2.0 * hh < min_h {
        hh = min_h * 0.5;
    }

    // 4. Enforce the maximum (cannot crop more than the full frame).
    if 2.0 * hw > fw {
        hw = fw * 0.5;
    }
    if 2.0 * hh > fh {
        hh = fh * 0.5;
    }

    // 5. Clamp the centre so the crop rect fits inside the frame.
    let cx = bbox.cx.clamp(hw, fw - hw);
    let cy = bbox.cy.clamp(hh, fh - hh);

    // 6. Quantise to integer pixels.
    let crop_w = (2.0 * hw).round() as u32;
    let crop_h = (2.0 * hh).round() as u32;
    let crop_x = (cx - hw).round() as u32;
    let crop_y = (cy - hh).round() as u32;
    // Re-clamp in case rounding pushed the rect off-frame.
    let crop_w = crop_w.min(frame_w);
    let crop_h = crop_h.min(frame_h);
    let crop_x = crop_x.min(frame_w - crop_w);
    let crop_y = crop_y.min(frame_h - crop_h);
    (crop_x, crop_y, crop_w, crop_h)
}

impl PostEffect for AutoFrameEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: AutoFrameConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        reject_out_of_range(Self::NAME, "threshold", cfg.threshold, 0.0, 1.0)?;
        reject_out_of_range(Self::NAME, "padding", cfg.padding, 0.0, 1.0)?;
        reject_out_of_range(Self::NAME, "smoothing", cfg.smoothing, 0.0, MAX_SMOOTHING)?;
        reject_out_of_range(Self::NAME, "zoom_max", cfg.zoom_max, 1.0, MAX_ZOOM)?;
        self.config = cfg;
        Ok(())
    }

    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError> {
        self.frame_w = context.width;
        self.frame_h = context.height;
        let frame_bytes = (context.width as usize) * (context.height as usize) * 3;
        self.crop_buf = vec![0u8; frame_bytes];
        // Reset smoothing state — a new prepare() call signals a fresh
        // pipeline negotiation; the previous state's coordinates are
        // meaningless at a different resolution.
        self.state = None;
        Ok(())
    }

    fn process(
        &mut self,
        plane: &mut FramePlane<'_>,
        mask: &MaskPlane<'_>,
        _ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        if plane.width != self.frame_w || plane.height != self.frame_h {
            return Err(EffectError::ProcessFailed {
                name: Self::NAME.to_string(),
                reason: format!(
                    "plane dimensions {}x{} differ from prepared {}x{}",
                    plane.width, plane.height, self.frame_w, self.frame_h
                ),
            });
        }
        if mask.width != self.frame_w || mask.height != self.frame_h {
            return Err(EffectError::ProcessFailed {
                name: Self::NAME.to_string(),
                reason: format!(
                    "mask dimensions {}x{} differ from frame {}x{}",
                    mask.width, mask.height, self.frame_w, self.frame_h
                ),
            });
        }
        // zoom_max == 1.0 is a configured no-op: skip the work
        // entirely. Saves a mask scan + a resize when the operator
        // wants to keep the effect registered but switched off.
        if self.config.zoom_max <= 1.0 {
            return Ok(());
        }

        let observed = extract_bbox(
            mask.data,
            mask.width as usize,
            mask.height as usize,
            self.config.threshold,
        );
        let Some(observed) = observed else {
            // No subject detected this frame. Hold whatever crop the
            // state already settled on so the video does not pop;
            // when state is also `None` (e.g. first frame is empty)
            // we simply pass through.
            if self.state.is_none() {
                return Ok(());
            }
            // Fall through with the previous state as the bbox.
            // (We could also fade back to no-zoom; leaving the last
            // crop is the least surprising behaviour.)
            let prev = self.state.expect("checked is_none above");
            self.crop_and_resize(plane, prev);
            return Ok(());
        };

        let smoothed = match self.state {
            Some(prev) => ema_blend(prev, observed, self.config.smoothing),
            // First frame after prepare/empty-period: seed state
            // directly with the observation (no inertia to a stale
            // value).
            None => observed,
        };
        self.state = Some(smoothed);
        self.crop_and_resize(plane, smoothed);
        Ok(())
    }
}

impl AutoFrameEffect {
    /// Copy the configured crop region out of `plane.data`, then
    /// bilinear-upscale it back into `plane.data`. Infallible — all
    /// error paths are caught at `configure`/`prepare` time and the
    /// in-place resize cannot fail.
    fn crop_and_resize(&mut self, plane: &mut FramePlane<'_>, bbox: SmoothBox) {
        let (cx, cy, cw, ch) = compute_crop_rect(bbox, &self.config, self.frame_w, self.frame_h);
        if cw == self.frame_w && ch == self.frame_h && cx == 0 && cy == 0 {
            // No-op crop: the rect already covers the whole frame.
            return;
        }
        if cw == 0 || ch == 0 {
            return;
        }
        // Copy the crop region row-by-row into `crop_buf`. The buffer is
        // sized to `frame_w * frame_h * 3` but we only fill the first
        // `cw * ch * 3` bytes — they form a tightly-packed sub-image
        // that `resize_rgb_bilinear` can consume.
        let row_stride = self.frame_w as usize * 3;
        let crop_row_bytes = cw as usize * 3;
        let needed = cw as usize * ch as usize * 3;
        for y in 0..ch as usize {
            let src_off = (cy as usize + y) * row_stride + cx as usize * 3;
            let dst_off = y * crop_row_bytes;
            self.crop_buf[dst_off..dst_off + crop_row_bytes]
                .copy_from_slice(&plane.data[src_off..src_off + crop_row_bytes]);
        }
        resize_rgb_bilinear(
            &self.crop_buf[..needed],
            cw,
            ch,
            plane.data,
            self.frame_w,
            self.frame_h,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(w: u32, h: u32) -> ProcessingContext {
        ProcessingContext {
            width: w,
            height: h,
            format: fluxframe_core::PixelFormat::Rgb,
            fps: 30,
            counters: None,
        }
    }

    fn parse(text: &str) -> RawEffectParams {
        toml::from_str(text).unwrap()
    }

    #[test]
    fn defaults_when_no_params() {
        let mut effect = AutoFrameEffect::new();
        effect
            .configure(toml::Value::Table(toml::map::Map::new()))
            .expect("ok");
        assert!((effect.config.threshold - default_threshold()).abs() < 1e-6);
        assert!((effect.config.smoothing - default_smoothing()).abs() < 1e-6);
        assert!((effect.config.zoom_max - default_zoom_max()).abs() < 1e-6);
    }

    #[test]
    fn rejects_threshold_out_of_range() {
        let mut effect = AutoFrameEffect::new();
        assert!(effect.configure(parse("threshold = 1.5")).is_err());
        assert!(effect.configure(parse("threshold = -0.1")).is_err());
    }

    #[test]
    fn rejects_padding_out_of_range() {
        let mut effect = AutoFrameEffect::new();
        assert!(effect.configure(parse("padding = -0.1")).is_err());
        assert!(effect.configure(parse("padding = 1.5")).is_err());
    }

    #[test]
    fn rejects_smoothing_out_of_range() {
        let mut effect = AutoFrameEffect::new();
        // 1.0 exactly is rejected — would freeze the EMA.
        assert!(effect.configure(parse("smoothing = 1.0")).is_err());
        assert!(effect.configure(parse("smoothing = -0.01")).is_err());
    }

    #[test]
    fn rejects_zoom_max_below_one() {
        let mut effect = AutoFrameEffect::new();
        // < 1.0 would *shrink* the crop relative to the frame, which
        // is nonsensical for "zoom max".
        assert!(effect.configure(parse("zoom_max = 0.5")).is_err());
        // > MAX_ZOOM is also rejected.
        assert!(effect.configure(parse("zoom_max = 10.0")).is_err());
    }

    #[test]
    fn extract_bbox_finds_central_region() {
        // 8×8 mask with a 4×4 hot region centred at (4, 4).
        let mut mask = vec![0.0_f32; 64];
        for y in 2..6 {
            for x in 2..6 {
                mask[y * 8 + x] = 1.0;
            }
        }
        let bbox = extract_bbox(&mask, 8, 8, 0.5).expect("present");
        // Centre = midpoint of [2..=5] inclusive: 3.5..=4.5 average = 4.0
        assert!((bbox.cx - 4.0).abs() < 0.1, "cx={}", bbox.cx);
        assert!((bbox.cy - 4.0).abs() < 0.1, "cy={}", bbox.cy);
        // Half-width = (5 - 2 + 1) / 2 = 2.0
        assert!((bbox.hw - 2.0).abs() < 0.1, "hw={}", bbox.hw);
        assert!((bbox.hh - 2.0).abs() < 0.1, "hh={}", bbox.hh);
    }

    #[test]
    fn extract_bbox_returns_none_on_empty_mask() {
        let mask = vec![0.0_f32; 64];
        assert!(extract_bbox(&mask, 8, 8, 0.5).is_none());
    }

    #[test]
    fn ema_converges_to_observation() {
        // Repeated lerp toward a fixed target must converge.
        let target = SmoothBox {
            cx: 100.0,
            cy: 50.0,
            hw: 20.0,
            hh: 10.0,
        };
        let mut state = SmoothBox {
            cx: 0.0,
            cy: 0.0,
            hw: 0.0,
            hh: 0.0,
        };
        for _ in 0..200 {
            state = ema_blend(state, target, 0.85);
        }
        assert!((state.cx - target.cx).abs() < 0.5, "cx={}", state.cx);
        assert!((state.cy - target.cy).abs() < 0.5, "cy={}", state.cy);
        assert!((state.hw - target.hw).abs() < 0.1, "hw={}", state.hw);
        assert!((state.hh - target.hh).abs() < 0.1, "hh={}", state.hh);
    }

    #[test]
    fn process_no_op_on_empty_mask_and_no_state() {
        let mut effect = AutoFrameEffect::new();
        effect
            .configure(toml::Value::Table(toml::map::Map::new()))
            .expect("configure");
        effect.prepare(&ctx(8, 8)).expect("prepare");
        let mut frame = vec![0u8; 8 * 8 * 3];
        for (i, b) in frame.iter_mut().enumerate() {
            *b = i as u8;
        }
        let original = frame.clone();
        let mut plane = FramePlane::new(&mut frame, 8, 8);
        let mut mask_data = vec![0.0_f32; 64];
        let mask = MaskPlane::new(&mut mask_data, 8, 8);
        let mut fctx = FrameContext::default();
        effect.process(&mut plane, &mask, &mut fctx).expect("ok");
        assert_eq!(frame, original, "empty mask + no state must be a no-op");
    }

    #[test]
    fn process_full_mask_is_approximately_identity() {
        // Mask fully on → bbox covers the whole frame → crop is the
        // whole frame → resize is a no-op (src == dst dimensions).
        let mut effect = AutoFrameEffect::new();
        effect
            .configure(toml::Value::Table(toml::map::Map::new()))
            .expect("configure");
        effect.prepare(&ctx(16, 16)).expect("prepare");
        let mut frame: Vec<u8> = (0..(16 * 16 * 3)).map(|i| (i % 256) as u8).collect();
        let original = frame.clone();
        let mut plane = FramePlane::new(&mut frame, 16, 16);
        let mut mask_data = vec![1.0_f32; 16 * 16];
        let mask = MaskPlane::new(&mut mask_data, 16, 16);
        let mut fctx = FrameContext::default();
        effect.process(&mut plane, &mask, &mut fctx).expect("ok");
        // Full-frame crop is short-circuited to a true no-op, so the
        // buffer is byte-identical to the input.
        assert_eq!(frame, original);
    }

    #[test]
    fn process_zoom_max_one_is_no_op() {
        let mut effect = AutoFrameEffect::new();
        effect
            .configure(parse("zoom_max = 1.0"))
            .expect("configure");
        effect.prepare(&ctx(8, 8)).expect("prepare");
        let mut frame = vec![123u8; 8 * 8 * 3];
        let original = frame.clone();
        let mut plane = FramePlane::new(&mut frame, 8, 8);
        let mut mask_data = vec![1.0_f32; 64];
        let mask = MaskPlane::new(&mut mask_data, 8, 8);
        let mut fctx = FrameContext::default();
        effect.process(&mut plane, &mask, &mut fctx).expect("ok");
        assert_eq!(frame, original);
    }

    #[test]
    fn rejects_process_on_dimension_mismatch() {
        let mut effect = AutoFrameEffect::new();
        effect
            .configure(toml::Value::Table(toml::map::Map::new()))
            .expect("configure");
        effect.prepare(&ctx(8, 8)).expect("prepare");
        let mut frame = vec![0u8; 4 * 4 * 3];
        let mut plane = FramePlane::new(&mut frame, 4, 4);
        let mut mask_data = vec![0.0_f32; 16];
        let mask = MaskPlane::new(&mut mask_data, 4, 4);
        let mut fctx = FrameContext::default();
        let err = effect
            .process(&mut plane, &mask, &mut fctx)
            .expect_err("dim mismatch");
        let msg = format!("{err}");
        assert!(msg.contains("plane dimensions"), "got: {msg}");
    }

    #[test]
    fn configure_after_prepare_is_safe() {
        // Live-reconfig contract (Stage 13): configure() may be re-called
        // after prepare(). For this stateful post-effect, the smoothing
        // state (`state: Option<SmoothBox>`) is preserved across
        // re-configure (only prepare() resets it). The next process()
        // recomputes the observed bbox against the new `threshold` and
        // must not panic.
        let mut effect = AutoFrameEffect::new();
        effect
            .configure(toml::Value::Table(toml::map::Map::new()))
            .expect("configure 1 (defaults)");
        effect.prepare(&ctx(8, 8)).expect("prepare");

        // 8×8 mask with a 4×4 hot region (values = 0.7) in the upper-
        // left. With default threshold 0.5 those pixels count; with
        // threshold 0.8 (post-reconfigure) they do not.
        let make_mask = || {
            let mut m = vec![0.0_f32; 64];
            for y in 0..4 {
                for x in 0..4 {
                    m[y * 8 + x] = 0.7;
                }
            }
            m
        };

        let mut frame = vec![100u8; 8 * 8 * 3];
        let mut mask_data = make_mask();
        {
            let mut plane = FramePlane::new(&mut frame, 8, 8);
            let mask = MaskPlane::new(&mut mask_data, 8, 8);
            let mut fctx = FrameContext::default();
            effect
                .process(&mut plane, &mask, &mut fctx)
                .expect("process 1");
        }
        // With threshold=0.5 the bbox is detected and `state` is set.
        assert!(effect.state.is_some(), "default threshold must detect bbox");

        // Re-configure with a higher threshold post-prepare. State must
        // be preserved (only prepare() clears it).
        let state_before = effect.state;
        effect
            .configure(parse(
                "threshold = 0.8\npadding = 0.0\nsmoothing = 0.0\nzoom_max = 2.0",
            ))
            .expect("re-configure ok");
        assert!(
            effect.state.is_some()
                && state_before.is_some()
                && (effect.state.unwrap().cx - state_before.unwrap().cx).abs() < 1e-6,
            "configure() must NOT reset smoothing state (only prepare() does)",
        );
        assert!((effect.config.threshold - 0.8).abs() < 1e-6);

        // Second process(): observed bbox is None (no pixel > 0.8), so
        // the effect falls through with the previous state — does not
        // panic.
        let mut frame = vec![100u8; 8 * 8 * 3];
        let mut mask_data = make_mask();
        let mut plane = FramePlane::new(&mut frame, 8, 8);
        let mask = MaskPlane::new(&mut mask_data, 8, 8);
        let mut fctx = FrameContext::default();
        effect
            .process(&mut plane, &mask, &mut fctx)
            .expect("process 2 must not panic on new threshold");
    }
}
