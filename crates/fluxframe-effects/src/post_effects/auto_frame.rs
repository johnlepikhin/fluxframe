//! `auto_frame` post-effect: smart cropping + recentering around the
//! person detected by the segmentation mask. Compensates for badly-
//! placed laptop webcams ("person sits in the lower-right corner")
//! without requiring physical recamering.
//!
//! The effect does not touch pixels.  It steers the runtime's framing
//! stage, which cuts every processed frame out of the captured one
//! ([`fluxframe_core::Framing`]): the crop happens *before*
//! segmentation and the other effects, so they run at the processing
//! size while a zoomed-in window keeps the capture's full detail.
//!
//! Algorithm, per frame:
//! 1. Measure the subject box on the mask, trimming the outer share of
//!    the mask mass so stray pixels and hands do not count
//!    ([`SubjectMeter`]), and map it to captured-frame coordinates
//!    through the region this frame was cut from.
//! 2. Turn the box into a steady window ([`SubjectTracker`]): the
//!    window stays still while the target is inside a dead zone, moves
//!    only after the target has stayed outside it for `hold_secs`, and
//!    then glides there over `glide_secs`.  After a few seconds without
//!    a subject it returns to the full view.
//! 3. Request the window for the next frame.
//!
//! A subject cut off at an edge of the window (and not at the edge of
//! the captured frame) is followed at once: its true extent lies
//! beyond what this frame shows.
//!
//! Cost per frame: one sampled mask scan.

use fluxframe_core::context::{FrameContext, FramingSource, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::metadata::{
    CommitStrategy, DEBOUNCE_STANDARD_MS, EffectMetadata, ParamDescriptor, ParamKind, Scale,
};
use fluxframe_core::plane::{FramePlane, MaskPlane, PostEffect};
use fluxframe_core::traits::RawEffectParams;
use serde::Deserialize;

use crate::plane_effects::helpers::reject_out_of_range;
use crate::processing::subject_tracker::{SubjectBox, SubjectMeter, SubjectTracker, TrackerParams};

/// Upper bound on `zoom_max`. Beyond 4× the resize blows up pixels
/// noticeably and the framing looks fake (people lean closer to fit
/// their own face, defeating the auto-frame).
const MAX_ZOOM: f32 = 4.0;
/// Upper bound on `smoothing`. Exactly `1.0` would mean the EMA never
/// updates (state stuck at the first sample). Cap below `1.0` so the
/// operator can't accidentally freeze the crop window.
const MAX_SMOOTHING: f32 = 0.99;
/// Upper bound on `deadband`: beyond half the window the subject could
/// leave the frame without the window ever following.
const MAX_DEADBAND: f32 = 0.5;
/// Upper bound on `hold_secs`.
const MAX_HOLD_SECS: f32 = 10.0;
/// Upper bound on `glide_secs`.
const MAX_GLIDE_SECS: f32 = 5.0;
/// Seconds without a subject before the window returns to the full
/// frame. Long enough to ride out a model dropout or a quick lean out
/// of view.
const LOST_SECS: f32 = 2.0;
/// A subject edge within this fraction of the frame size from a frame
/// edge counts as cut off there.
const EDGE_TOUCH: f32 = 0.02;
/// A window edge within this fraction of the captured size from the
/// captured frame's edge shows all there is: a few pixels beyond it are
/// not worth revealing.  The full view itself stops short of the edges
/// when its aspect differs from the capture's (a 1280 x 720 capture shown
/// at 800 x 448 leaves 1.6 px above and below), and a subject touching
/// such an edge must not count as cut off: every frame would then be an
/// urgent move from a guess, and the subject would never be measured.
const WINDOW_AT_EDGE: f32 = 0.02;
/// How far beyond a cut-off edge the subject is assumed to extend, as a
/// fraction of the window size on that axis.  Generous on purpose: the
/// window then moves out far enough to reveal the real extent, after
/// which the measurement is exact again.
const CUT_OFF_EXTENSION: f32 = 0.5;

/// Default mask-confidence threshold for the `threshold` field.
pub const DEFAULT_THRESHOLD: f32 = 0.5;
/// Default symmetric bbox expansion fraction for the `padding` field.
pub const DEFAULT_PADDING: f32 = 0.15;
/// Default EMA inertia for the `smoothing` field.
pub const DEFAULT_SMOOTHING: f32 = 0.85;
/// Default maximum zoom factor for the `zoom_max` field.
pub const DEFAULT_ZOOM_MAX: f32 = 1.6;
/// Default dead zone for the `deadband` field.
pub const DEFAULT_DEADBAND: f32 = 0.12;
/// Default hold time for the `hold_secs` field.
pub const DEFAULT_HOLD_SECS: f32 = 1.0;
/// Default move duration for the `glide_secs` field.
pub const DEFAULT_GLIDE_SECS: f32 = 0.8;

/// TOML schema:
///
/// ```toml
/// [post.auto_frame]
/// threshold = 0.5
/// padding = 0.15
/// smoothing = 0.85
/// zoom_max = 1.6
/// deadband = 0.12
/// hold_secs = 1.0
/// glide_secs = 0.8
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutoFrameConfig {
    /// Mask-confidence threshold above which a pixel counts toward the
    /// subject box. Validated to be a finite number in `[0.0, 1.0]`.
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    /// Box expansion factor applied symmetrically. `0.15` adds a 15 %
    /// margin around the silhouette so the crop does not clip hair /
    /// shoulders. Validated to be in `[0.0, 1.0]`.
    #[serde(default = "default_padding")]
    pub padding: f32,
    /// EMA inertia of the measured box, per frame. `0.0` follows the
    /// raw measurement; higher values calm it. Validated to be in
    /// `[0.0, 0.99]`.
    #[serde(default = "default_smoothing")]
    pub smoothing: f32,
    /// Maximum zoom factor. `1.0` disables the effect (no crop).
    /// Validated to be in `[1.0, 4.0]`.
    #[serde(default = "default_zoom_max")]
    pub zoom_max: f32,
    /// How far the target may drift from the current window, as a
    /// fraction of the window size, before a move is considered.
    /// Validated to be in `[0.0, 0.5]`.
    #[serde(default = "default_deadband")]
    pub deadband: f32,
    /// Seconds the target must stay outside the dead zone before the
    /// window moves. Validated to be in `[0.0, 10.0]`.
    #[serde(default = "default_hold_secs")]
    pub hold_secs: f32,
    /// Seconds a move takes. `0` jumps. Validated to be in `[0.0, 5.0]`.
    #[serde(default = "default_glide_secs")]
    pub glide_secs: f32,
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
fn default_deadband() -> f32 {
    DEFAULT_DEADBAND
}
fn default_hold_secs() -> f32 {
    DEFAULT_HOLD_SECS
}
fn default_glide_secs() -> f32 {
    DEFAULT_GLIDE_SECS
}

impl Default for AutoFrameConfig {
    fn default() -> Self {
        Self {
            threshold: default_threshold(),
            padding: default_padding(),
            smoothing: default_smoothing(),
            zoom_max: default_zoom_max(),
            deadband: default_deadband(),
            hold_secs: default_hold_secs(),
            glide_secs: default_glide_secs(),
        }
    }
}

impl AutoFrameConfig {
    /// Tracker tunables with times converted to frames at `fps`.
    fn tracker_params(&self, fps: u32) -> TrackerParams {
        let frames = |secs: f32| (secs * fps as f32).round() as u32;
        TrackerParams {
            padding: self.padding,
            smoothing: self.smoothing,
            zoom_max: self.zoom_max,
            deadband: self.deadband,
            hold_frames: frames(self.hold_secs),
            glide_frames: frames(self.glide_secs),
            lost_frames: frames(LOST_SECS).max(1),
        }
    }
}

/// `PostEffect` performing mask-driven auto-framing.
pub struct AutoFrameEffect {
    config: AutoFrameConfig,
    meter: SubjectMeter,
    /// Tracks windows in captured-frame coordinates.  Built on the first
    /// framed frame (the captured size is only known then) and rebuilt
    /// when that size changes.
    tracker: Option<SubjectTracker>,
    /// Captured size the tracker was built for.
    tracker_space: (u32, u32),
    frame_w: u32,
    frame_h: u32,
    fps: u32,
}

/// Float slider descriptor with the standard live-commit debounce.
const fn float_param(
    name: &'static str,
    default: f32,
    min: f32,
    max: f32,
    step: f32,
    help: &'static str,
) -> ParamDescriptor {
    ParamDescriptor {
        name,
        kind: ParamKind::Float {
            default,
            min,
            max,
            step,
            scale: Scale::Linear,
        },
        help,
        commit: CommitStrategy::Live {
            debounce_ms: DEBOUNCE_STANDARD_MS,
        },
    }
}

impl AutoFrameEffect {
    /// Effect name as registered in the post-effect registry.
    pub const NAME: &'static str = "auto_frame";

    /// Self-describing metadata for the registry and the GUI.
    pub const METADATA: EffectMetadata = EffectMetadata {
        name: Self::NAME,
        help: "Mask-driven auto-framing: crop and recenter around the subject.",
        params: &[
            float_param(
                "threshold",
                DEFAULT_THRESHOLD,
                0.0,
                1.0,
                0.05,
                "Mask confidence cutoff for measuring the subject.",
            ),
            float_param(
                "padding",
                DEFAULT_PADDING,
                0.0,
                1.0,
                0.05,
                "Symmetric margin around the subject.",
            ),
            float_param(
                "smoothing",
                DEFAULT_SMOOTHING,
                0.0,
                MAX_SMOOTHING,
                0.05,
                "Per-frame inertia of the measured subject; 0 follows it raw.",
            ),
            float_param(
                "zoom_max",
                DEFAULT_ZOOM_MAX,
                1.0,
                MAX_ZOOM,
                0.1,
                "Maximum allowed zoom; 1.0 disables cropping.",
            ),
            float_param(
                "deadband",
                DEFAULT_DEADBAND,
                0.0,
                MAX_DEADBAND,
                0.01,
                "Drift tolerated before reframing, as a fraction of the frame.",
            ),
            float_param(
                "hold_secs",
                DEFAULT_HOLD_SECS,
                0.0,
                MAX_HOLD_SECS,
                0.1,
                "How long the subject must stay off-centre before reframing.",
            ),
            float_param(
                "glide_secs",
                DEFAULT_GLIDE_SECS,
                0.0,
                MAX_GLIDE_SECS,
                0.1,
                "Duration of a reframing move; 0 jumps.",
            ),
        ],
    };

    /// Construct with defaults. `configure()` and `prepare()` must be
    /// called before `process()`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: AutoFrameConfig::default(),
            meter: SubjectMeter::new(),
            tracker: None,
            tracker_space: (0, 0),
            frame_w: 0,
            frame_h: 0,
            fps: 0,
        }
    }
}

impl Default for AutoFrameEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoFrameEffect {
    /// The tracker for `source`'s captured size, built on first use.
    fn tracker_for(&mut self, source: &FramingSource) -> &mut SubjectTracker {
        let space = (source.width, source.height);
        if self.tracker.is_none() || self.tracker_space != space {
            let aspect = self.frame_w as f32 / self.frame_h.max(1) as f32;
            self.tracker = Some(SubjectTracker::new(
                self.config.tracker_params(self.fps),
                source.width,
                source.height,
                aspect,
            ));
            self.tracker_space = space;
        }
        self.tracker.as_mut().expect("tracker built above")
    }
}

/// Map `subject`, measured on a `frame_w × frame_h` frame cut from
/// `source`, to captured-frame coordinates.  Returns the box and whether
/// it is cut off at a window edge that is not a captured-frame edge; such
/// a side is extended by [`CUT_OFF_EXTENSION`].
fn to_capture(
    subject: SubjectBox,
    source: &FramingSource,
    frame_w: u32,
    frame_h: u32,
) -> (SubjectBox, bool) {
    let (fw, fh) = (frame_w as f32, frame_h as f32);
    let r = &source.region;
    let (mut left, mut top) = r.map_from_local(
        subject.cx - subject.hw,
        subject.cy - subject.hh,
        frame_w,
        frame_h,
    );
    let (mut right, mut bottom) = r.map_from_local(
        subject.cx + subject.hw,
        subject.cy + subject.hh,
        frame_w,
        frame_h,
    );
    let (tol_x, tol_y) = (EDGE_TOUCH * fw, EDGE_TOUCH * fh);
    // A window edge clearly inside the captured frame has more picture
    // beyond it.
    let (cw, ch) = (source.width as f32, source.height as f32);
    let (gap_x, gap_y) = (WINDOW_AT_EDGE * cw, WINDOW_AT_EDGE * ch);
    let more_left = r.x > gap_x;
    let more_top = r.y > gap_y;
    let more_right = r.x + r.width < cw - gap_x;
    let more_bottom = r.y + r.height < ch - gap_y;
    let mut cut_off = false;
    if more_left && subject.cx - subject.hw <= tol_x {
        left -= CUT_OFF_EXTENSION * r.width;
        cut_off = true;
    }
    if more_right && subject.cx + subject.hw >= fw - tol_x {
        right += CUT_OFF_EXTENSION * r.width;
        cut_off = true;
    }
    if more_top && subject.cy - subject.hh <= tol_y {
        top -= CUT_OFF_EXTENSION * r.height;
        cut_off = true;
    }
    if more_bottom && subject.cy + subject.hh >= fh - tol_y {
        bottom += CUT_OFF_EXTENSION * r.height;
        cut_off = true;
    }
    let boxed = SubjectBox {
        cx: f32::midpoint(left, right),
        cy: f32::midpoint(top, bottom),
        hw: (right - left) * 0.5,
        hh: (bottom - top) * 0.5,
    };
    (boxed, cut_off)
}

impl PostEffect for AutoFrameEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn reset_state(&mut self) {
        // Start tracking from the current subject box instead of gliding
        // from where the subject was before the effect was disabled.
        if let Some(tracker) = self.tracker.as_mut() {
            tracker.reset();
        }
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: AutoFrameConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        reject_out_of_range(Self::NAME, "threshold", cfg.threshold, 0.0, 1.0)?;
        reject_out_of_range(Self::NAME, "padding", cfg.padding, 0.0, 1.0)?;
        reject_out_of_range(Self::NAME, "smoothing", cfg.smoothing, 0.0, MAX_SMOOTHING)?;
        reject_out_of_range(Self::NAME, "zoom_max", cfg.zoom_max, 1.0, MAX_ZOOM)?;
        reject_out_of_range(Self::NAME, "deadband", cfg.deadband, 0.0, MAX_DEADBAND)?;
        reject_out_of_range(Self::NAME, "hold_secs", cfg.hold_secs, 0.0, MAX_HOLD_SECS)?;
        reject_out_of_range(
            Self::NAME,
            "glide_secs",
            cfg.glide_secs,
            0.0,
            MAX_GLIDE_SECS,
        )?;
        // Live reconfig keeps the tracking state: a slider drag must not
        // make the window jump.
        if let Some(tracker) = self.tracker.as_mut() {
            tracker.set_params(cfg.tracker_params(self.fps));
        }
        self.config = cfg;
        Ok(())
    }

    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError> {
        self.frame_w = context.width;
        self.frame_h = context.height;
        self.fps = context.fps;
        // A new geometry makes any previous window meaningless.
        self.tracker = None;
        Ok(())
    }

    fn process(
        &mut self,
        plane: &mut FramePlane<'_>,
        mask: &MaskPlane<'_>,
        ctx: &mut FrameContext,
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
        // Without a framing stage there is nothing to steer.
        let Some(source) = ctx.framing.source else {
            return Ok(());
        };
        // zoom_max == 1.0 is a configured no-op: no request means the
        // full view.
        if self.config.zoom_max <= 1.0 {
            return Ok(());
        }
        let observed = self.meter.measure(
            mask.data,
            mask.width as usize,
            mask.height as usize,
            self.config.threshold,
        );
        let (observed, cut_off) = match observed {
            Some(subject) => {
                let (boxed, cut_off) = to_capture(subject, &source, self.frame_w, self.frame_h);
                (Some(boxed), cut_off)
            }
            None => (None, false),
        };
        let window = self.tracker_for(&source).update(observed, cut_off);
        ctx.framing.request = Some(window);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use fluxframe_core::{Framing, Region};

    use super::*;

    const W: u32 = 64;
    const H: u32 = 36;

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

    fn empty() -> RawEffectParams {
        toml::Value::Table(toml::map::Map::new())
    }

    /// Frame context of a frame cut from `region` of a `cw × ch` capture.
    fn framed(cw: u32, ch: u32, region: Region) -> FrameContext {
        FrameContext {
            framing: Framing {
                source: Some(FramingSource {
                    width: cw,
                    height: ch,
                    region,
                }),
                request: None,
            },
            ..FrameContext::default()
        }
    }

    /// Mask with a hot block `x0..x1 × y0..y1`.
    fn block(x0: u32, y0: u32, x1: u32, y1: u32) -> Vec<f32> {
        let mut m = vec![0.0_f32; (W * H) as usize];
        for y in y0..y1 {
            for x in x0..x1 {
                m[(y * W + x) as usize] = 1.0;
            }
        }
        m
    }

    fn prepared(cfg: &str) -> AutoFrameEffect {
        let mut effect = AutoFrameEffect::new();
        effect.configure(parse(cfg)).expect("configure");
        effect.prepare(&ctx(W, H)).expect("prepare");
        effect
    }

    /// Run one frame; return the request it left in the context.
    fn step(
        effect: &mut AutoFrameEffect,
        mask: &mut [f32],
        fctx: &mut FrameContext,
    ) -> Option<Region> {
        let mut frame = vec![7u8; (W * H * 3) as usize];
        let original = frame.clone();
        let mut plane = FramePlane::new(&mut frame, W, H);
        let mask = MaskPlane::new(mask, W, H);
        fctx.framing.request = None;
        effect.process(&mut plane, &mask, fctx).expect("ok");
        assert_eq!(frame, original, "auto_frame must not touch pixels");
        fctx.framing.request
    }

    #[test]
    fn defaults_when_no_params() {
        let mut effect = AutoFrameEffect::new();
        effect.configure(empty()).expect("ok");
        assert!((effect.config.threshold - default_threshold()).abs() < 1e-6);
        assert!((effect.config.smoothing - default_smoothing()).abs() < 1e-6);
        assert!((effect.config.zoom_max - default_zoom_max()).abs() < 1e-6);
        assert!((effect.config.deadband - default_deadband()).abs() < 1e-6);
        assert!((effect.config.hold_secs - default_hold_secs()).abs() < 1e-6);
        assert!((effect.config.glide_secs - default_glide_secs()).abs() < 1e-6);
    }

    #[test]
    fn metadata_lists_every_config_field() {
        let names: Vec<&str> = AutoFrameEffect::METADATA
            .params
            .iter()
            .map(|p| p.name)
            .collect();
        assert_eq!(
            names,
            [
                "threshold",
                "padding",
                "smoothing",
                "zoom_max",
                "deadband",
                "hold_secs",
                "glide_secs"
            ]
        );
    }

    #[test]
    fn rejects_out_of_range_values() {
        let mut effect = AutoFrameEffect::new();
        for bad in [
            "threshold = 1.5",
            "threshold = -0.1",
            "padding = -0.1",
            "padding = 1.5",
            // 1.0 exactly is rejected — would freeze the EMA.
            "smoothing = 1.0",
            "smoothing = -0.01",
            // < 1.0 would *shrink* the crop relative to the frame.
            "zoom_max = 0.5",
            "zoom_max = 10.0",
            "deadband = 0.6",
            "hold_secs = -1.0",
            "hold_secs = 11.0",
            "glide_secs = 6.0",
        ] {
            assert!(effect.configure(parse(bad)).is_err(), "{bad} accepted");
        }
    }

    #[test]
    fn tracker_params_convert_seconds_to_frames() {
        let cfg = AutoFrameConfig {
            hold_secs: 1.0,
            glide_secs: 0.5,
            ..AutoFrameConfig::default()
        };
        let p = cfg.tracker_params(30);
        assert_eq!(p.hold_frames, 30);
        assert_eq!(p.glide_frames, 15);
        assert_eq!(p.lost_frames, 60);
    }

    #[test]
    fn without_framing_stage_nothing_is_requested() {
        let mut effect = prepared("zoom_max = 2.0");
        let mut mask = block(28, 10, 36, 26);
        let mut fctx = FrameContext::default();
        assert_eq!(step(&mut effect, &mut mask, &mut fctx), None);
    }

    #[test]
    fn zoom_max_one_requests_nothing() {
        let mut effect = prepared("zoom_max = 1.0");
        let mut mask = block(28, 10, 36, 26);
        let mut fctx = framed(128, 72, Region::full(128, 72));
        assert_eq!(step(&mut effect, &mut mask, &mut fctx), None);
    }

    #[test]
    fn requests_a_window_in_capture_coordinates() {
        let mut effect = prepared("zoom_max = 2.0");
        // Small subject in the middle of the processed frame, which is
        // the full view of a 2× larger capture.
        let mut mask = block(30, 14, 34, 22);
        let mut fctx = framed(128, 72, Region::full(128, 72));
        let window = step(&mut effect, &mut mask, &mut fctx).expect("request");
        // zoom_max 2 → 64×36 window of the 128×72 capture, centred on
        // the subject (x ≈ 64, y ≈ 36 in capture pixels).
        assert!((window.width - 64.0).abs() < 0.5, "{window:?}");
        assert!((window.center_x() - 64.0).abs() < 3.0, "{window:?}");
        assert!((window.center_y() - 36.0).abs() < 3.0, "{window:?}");
    }

    #[test]
    fn no_subject_requests_the_full_view() {
        let mut effect = prepared("zoom_max = 2.0");
        let mut mask = vec![0.0_f32; (W * H) as usize];
        let mut fctx = framed(128, 72, Region::full(128, 72));
        let window = step(&mut effect, &mut mask, &mut fctx).expect("request");
        assert!(window.approx_eq(&Region::full(128, 72), 1e-3), "{window:?}");
    }

    #[test]
    fn cut_off_subject_is_extended_beyond_the_window() {
        let source = FramingSource {
            width: 128,
            height: 72,
            region: Region {
                x: 32.0,
                y: 18.0,
                width: 64.0,
                height: 36.0,
            },
        };
        // Touches the left edge of the processed frame.
        let subject = SubjectBox {
            cx: 6.0,
            cy: 18.0,
            hw: 6.0,
            hh: 6.0,
        };
        let (boxed, cut_off) = to_capture(subject, &source, W, H);
        assert!(cut_off);
        // Left edge 32 in capture, extended by half the window width.
        assert!((boxed.cx - boxed.hw - 0.0).abs() < 1e-3, "{boxed:?}");
        // The same subject in a full-view frame touches the capture edge
        // itself: nothing lies beyond, no extension.
        let full = FramingSource {
            width: 64,
            height: 36,
            region: Region::full(64, 36),
        };
        let (boxed, cut_off) = to_capture(subject, &full, W, H);
        assert!(!cut_off);
        assert!((boxed.cx - 6.0).abs() < 1e-3, "{boxed:?}");
    }

    /// The full view of a capture whose aspect differs slightly from the
    /// processed frame's stops 1.6 px short of the top and bottom.  A
    /// seated subject touching the bottom of that view is not cut off:
    /// there is nothing to reveal, and treating it as cut off kept the
    /// window from ever zooming in.
    #[test]
    fn subject_at_the_edge_of_a_banded_full_view_is_not_cut_off() {
        let full_band = FramingSource {
            width: 1280,
            height: 720,
            region: Region {
                x: 0.0,
                y: 1.6,
                width: 1280.0,
                height: 716.8,
            },
        };
        // Touches the bottom of the 800 x 448 processed frame.
        let seated = SubjectBox {
            cx: 400.0,
            cy: 300.0,
            hw: 150.0,
            hh: 148.0,
        };
        let (boxed, cut_off) = to_capture(seated, &full_band, 800, 448);
        assert!(!cut_off, "{boxed:?}");
        assert!((boxed.cy + boxed.hh - 718.4).abs() < 0.5, "{boxed:?}");
    }

    /// Closed loop, as in the runtime: each frame is cut from the window
    /// the previous one requested.  Live geometry: a 1280 x 720 capture
    /// processed at 800 x 448, whose full view stops 1.6 px short of the
    /// capture's top and bottom.  The person first sits, then leaves
    /// (the window returns to that full view), then sits down again,
    /// filling the capture down to its bottom edge.  The window must zoom
    /// in on the person again and stay there.
    #[test]
    fn seated_subject_at_the_bottom_edge_is_zoomed_in_on() {
        const CW: u32 = 1280;
        const CH: u32 = 720;
        const FW: u32 = 800;
        const FH: u32 = 448;
        // The person in capture coordinates, as measured live; the body
        // continues below the capture.
        let (px0, py0, px1, py1) = (403.0_f32, 223.0_f32, 891.0_f32, CH as f32);
        let mut effect = AutoFrameEffect::new();
        effect
            .configure(parse("padding = 0.1\nsmoothing = 0.85\nzoom_max = 1.3"))
            .expect("configure");
        effect.prepare(&ctx(FW, FH)).expect("prepare");
        let mut region = Region {
            x: 0.0,
            y: 1.6,
            width: 1280.0,
            height: 716.8,
        };
        let mut frame = vec![0u8; (FW * FH * 3) as usize];
        let mut widths = Vec::new();
        for i in 0..900 {
            let present = !(200..400).contains(&i);
            // Render the person as this window shows it.
            let to_x = |x: f32| ((x - region.x) / region.width * FW as f32).clamp(0.0, FW as f32);
            let to_y = |y: f32| ((y - region.y) / region.height * FH as f32).clamp(0.0, FH as f32);
            let (x0, x1) = (to_x(px0).round() as u32, to_x(px1).round() as u32);
            let (y0, y1) = (to_y(py0).round() as u32, to_y(py1).round() as u32);
            let mut mask = vec![0.0_f32; (FW * FH) as usize];
            for y in (y0..y1).filter(|_| present) {
                for x in x0..x1 {
                    mask[(y * FW + x) as usize] = 1.0;
                }
            }
            let mut fctx = framed(CW, CH, region);
            let mut plane = FramePlane::new(&mut frame, FW, FH);
            effect
                .process(&mut plane, &MaskPlane::new(&mut mask, FW, FH), &mut fctx)
                .expect("ok");
            region = fctx.framing.request.expect("a request");
            widths.push(region.width);
        }
        let zoomed = CW as f32 / 1.3;
        assert!(
            (region.width - zoomed).abs() < 2.0,
            "not zoomed in to 1.3: {region:?}"
        );
        assert!(
            widths[399] > CW as f32 - 2.0,
            "the window must return to the full view while nobody is there"
        );
        // Settled: the last few seconds hold still.
        let tail = &widths[widths.len() - 150..];
        assert!(
            tail.iter().all(|w| (w - tail[0]).abs() < 1e-3),
            "the window keeps moving"
        );
    }

    #[test]
    fn configure_after_prepare_keeps_the_window() {
        // Live-reconfig contract (Stage 13): configure() may be re-called
        // after prepare() and must not reset the tracking state, or a
        // slider drag in the GUI would make the window jump.
        let mut effect = prepared("zoom_max = 2.0");
        let mut mask = block(28, 10, 36, 26);
        let mut fctx = framed(128, 72, Region::full(128, 72));
        let before = step(&mut effect, &mut mask, &mut fctx).expect("request");
        effect
            .configure(parse("zoom_max = 2.0\nthreshold = 0.8\nhold_secs = 3.0"))
            .expect("re-configure ok");
        fctx.framing.source = Some(FramingSource {
            width: 128,
            height: 72,
            region: before,
        });
        // Same subject, now seen through the window: still centred.
        let mut mask = block(16, 0, 48, 36);
        let after = step(&mut effect, &mut mask, &mut fctx).expect("request");
        assert!(after.approx_eq(&before, 1e-3), "{before:?} → {after:?}");
    }

    #[test]
    fn rejects_process_on_dimension_mismatch() {
        let mut effect = prepared("");
        let mut frame = vec![0u8; 4 * 4 * 3];
        let mut plane = FramePlane::new(&mut frame, 4, 4);
        let mut mask_data = vec![0.0_f32; 16];
        let mask = MaskPlane::new(&mut mask_data, 4, 4);
        let mut fctx = FrameContext::default();
        let err = effect
            .process(&mut plane, &mask, &mut fctx)
            .expect_err("dim mismatch");
        assert!(format!("{err}").contains("plane dimensions"), "got: {err}");
    }
}
