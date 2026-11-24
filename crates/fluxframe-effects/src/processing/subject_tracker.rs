//! Subject measurement and crop-window tracking for auto-framing.
//!
//! Two independent pieces:
//!
//! * [`SubjectMeter`] turns a segmentation mask into the box the person
//!   occupies.  It trims a small share of the mask mass from each side
//!   instead of taking the outermost hot pixels, so stray pixels, a
//!   raised hand or a small false blob do not move the box.
//! * [`SubjectTracker`] turns the per-frame box into a crop window that
//!   stays **still** while the person merely fidgets.  The silhouette
//!   changes on every frame, so "the box changed" is no reason to move;
//!   the window moves only when the smoothed target has stayed outside a
//!   dead zone for a hold time, and then glides to it.
//!
//! Both are pure: no frame buffers, no I/O — fully unit-testable.

use fluxframe_core::Region;
use tracing::debug;

/// Share of the mask mass trimmed from the left and from the right.
/// Cuts arms and hands held to the side.
const TRIM_SIDE: f32 = 0.03;
/// Share trimmed from the top.  Small: the narrow crown of the head
/// holds little mass and must survive.
const TRIM_TOP: f32 = 0.005;
/// Share trimmed from the bottom.
const TRIM_BOTTOM: f32 = 0.02;
/// Minimum share of sampled pixels that must be hot for a subject to
/// count as present; below it the mask holds only noise.
const MIN_COVERAGE: f32 = 0.002;
/// The meter samples roughly this many columns per row, whatever the
/// mask width (the mask is an upscaled model output, so denser sampling
/// adds no information).
const SAMPLE_COLUMNS: usize = 320;
/// Smallest deviation worth a move for a cut-off subject, which skips
/// the dead zone; below it the window already shows what it can.
const URGENT_MIN_DEVIATION: f32 = 0.005;

/// Box a subject occupies, as centre and half-extents in mask pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubjectBox {
    /// Horizontal centre.
    pub cx: f32,
    /// Vertical centre.
    pub cy: f32,
    /// Half width.
    pub hw: f32,
    /// Half height.
    pub hh: f32,
}

/// Measures the subject box of a mask.  Holds the projection scratch so
/// measuring allocates nothing after the first frame.
#[derive(Debug, Default)]
pub struct SubjectMeter {
    cols: Vec<u32>,
    rows: Vec<u32>,
}

impl SubjectMeter {
    /// Construct with empty scratch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Box of the pixels above `threshold` in a `width × height` mask,
    /// after trimming the outer share of their mass on every side.
    /// `None` when the mask holds (almost) no subject.
    pub fn measure(
        &mut self,
        mask: &[f32],
        width: usize,
        height: usize,
        threshold: f32,
    ) -> Option<SubjectBox> {
        if width == 0 || height == 0 || mask.len() < width * height {
            return None;
        }
        let step = (width / SAMPLE_COLUMNS).max(1);
        self.cols.clear();
        self.cols.resize(width.div_ceil(step), 0);
        self.rows.clear();
        for row in mask.chunks_exact(width).take(height).step_by(step) {
            let mut hot = 0;
            for (col, &v) in self.cols.iter_mut().zip(row.iter().step_by(step)) {
                if v > threshold {
                    *col += 1;
                    hot += 1;
                }
            }
            self.rows.push(hot);
        }
        let total: u32 = self.rows.iter().sum();
        let sampled = (self.cols.len() * self.rows.len()) as f32;
        if total == 0 || (total as f32) < MIN_COVERAGE * sampled {
            return None;
        }
        let (x0, x1) = trimmed_span(&self.cols, total, TRIM_SIDE, TRIM_SIDE);
        let (y0, y1) = trimmed_span(&self.rows, total, TRIM_TOP, TRIM_BOTTOM);
        // Sample indices → pixel edges; the last sample covers up to
        // the next one (or the mask edge).
        let left = (x0 * step) as f32;
        let right = ((x1 + 1) * step).min(width) as f32;
        let top = (y0 * step) as f32;
        let bottom = ((y1 + 1) * step).min(height) as f32;
        Some(SubjectBox {
            cx: f32::midpoint(left, right),
            cy: f32::midpoint(top, bottom),
            hw: (right - left) * 0.5,
            hh: (bottom - top) * 0.5,
        })
    }
}

/// First and last bin of `hist` once `lo` × `total` of the mass is cut
/// from the start and `hi` × `total` from the end.
fn trimmed_span(hist: &[u32], total: u32, lo: f32, hi: f32) -> (usize, usize) {
    let first = edge_bin(hist.iter().copied(), lo * total as f32);
    let last = hist.len() - 1 - edge_bin(hist.iter().rev().copied(), hi * total as f32);
    (first.min(last), first.max(last))
}

/// Index of the first bin at which the running sum exceeds `cut`.
fn edge_bin(bins: impl Iterator<Item = u32>, cut: f32) -> usize {
    let mut sum = 0u32;
    for (i, n) in bins.enumerate() {
        sum += n;
        if sum as f32 > cut {
            return i;
        }
    }
    0
}

/// Tunables of [`SubjectTracker`]; times are already in frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackerParams {
    /// Margin added around the subject box, as a fraction of its size.
    pub padding: f32,
    /// EMA inertia of the measured box, `[0, 1)`.
    pub smoothing: f32,
    /// Maximum zoom relative to the full view, `>= 1`.
    pub zoom_max: f32,
    /// Allowed deviation of the target from the current window, as a
    /// fraction of the window size, before a move is considered.
    pub deadband: f32,
    /// Frames the target must stay outside the dead zone before the
    /// window moves.
    pub hold_frames: u32,
    /// Frames a move takes.
    pub glide_frames: u32,
    /// Frames without a subject after which the window returns to the
    /// full view.
    pub lost_frames: u32,
}

/// A move in progress from `from` towards the (live) target.
#[derive(Debug, Clone, Copy)]
struct Glide {
    from: Region,
    frame: u32,
}

/// Turns per-frame subject boxes into a steady crop window.
///
/// Coordinates are those of a `space_w × space_h` frame; windows always
/// have the aspect ratio of the full view and never leave the frame.
#[derive(Debug)]
pub struct SubjectTracker {
    params: TrackerParams,
    space_w: f32,
    space_h: f32,
    /// Largest window: the whole frame, or its centred band when the
    /// requested aspect differs from the frame's.
    full: Region,
    filtered: Option<SubjectBox>,
    /// Window shown while not gliding.  `None` until the first update.
    locked: Option<Region>,
    glide: Option<Glide>,
    /// Consecutive frames the target has been outside the dead zone.
    hold: u32,
    /// Consecutive frames without a subject.
    lost: u32,
    /// The window was placed for a cut-off subject, from a guess at its
    /// extent rather than a measurement.  The dead zone does not hold
    /// such a window: once the subject is measured in full, the window
    /// settles on it even when the correction is small.
    provisional: bool,
}

impl SubjectTracker {
    /// Tracker for a `space_w × space_h` frame whose windows have aspect
    /// ratio `aspect` (width / height).
    #[must_use]
    pub fn new(params: TrackerParams, space_w: u32, space_h: u32, aspect: f32) -> Self {
        Self {
            params,
            space_w: space_w as f32,
            space_h: space_h as f32,
            full: Region::centered_with_aspect(space_w, space_h, aspect),
            filtered: None,
            locked: None,
            glide: None,
            hold: 0,
            lost: 0,
            provisional: false,
        }
    }

    /// Replace the tunables, keeping the tracking state (live reconfig).
    pub fn set_params(&mut self, params: TrackerParams) {
        self.params = params;
    }

    /// Forget all history: the next update snaps to its target.
    pub fn reset(&mut self) {
        self.filtered = None;
        self.locked = None;
        self.glide = None;
        self.hold = 0;
        self.lost = 0;
        self.provisional = false;
    }

    /// The largest window (no zoom).
    #[must_use]
    pub fn full_view(&self) -> Region {
        self.full
    }

    /// Whether a move is in progress.
    #[must_use]
    pub fn is_gliding(&self) -> bool {
        self.glide.is_some()
    }

    /// Feed this frame's measurement (`None`: no subject) and get the
    /// window to show.  `urgent` skips the hold time: the caller knows
    /// the subject is leaving the window (it is cut off at an edge), so
    /// waiting would only show it cut off for longer.
    pub fn update(&mut self, observed: Option<SubjectBox>, urgent: bool) -> Region {
        let lost_long = if let Some(obs) = observed {
            if self.lost >= self.params.lost_frames {
                // Back after a long absence: start from the fresh
                // position, not from where the subject left.
                self.filtered = None;
            }
            self.lost = 0;
            let s = self.params.smoothing;
            self.filtered = Some(self.filtered.map_or(obs, |prev| ema(prev, obs, s)));
            false
        } else {
            self.lost = self.lost.saturating_add(1);
            self.lost >= self.params.lost_frames
        };
        let target = match self.filtered {
            Some(subject) if !lost_long => self.window_for(subject),
            _ => self.full,
        };

        let Some(locked) = self.locked else {
            self.locked = Some(target);
            return target;
        };

        if let Some(glide) = self.glide.as_mut() {
            glide.frame += 1;
            if glide.frame >= self.params.glide_frames {
                self.glide = None;
                self.locked = Some(target);
                return target;
            }
            let t = glide.frame as f32 / self.params.glide_frames as f32;
            return glide.from.lerp(&target, smoothstep(t));
        }

        // The dead zone applies to a window placed from a measurement.
        // A cut-off subject's true extent is unknown until the window
        // reveals it, and a window placed from a guess at that extent
        // has to settle once the subject is measured: either kept still
        // by the dead zone would stay off for good.
        let dev = deviation(&locked, &target);
        let dead_zone = if urgent || self.provisional {
            URGENT_MIN_DEVIATION
        } else {
            self.params.deadband
        };
        if dev <= dead_zone {
            // Only a full measurement confirms the window.
            if !urgent {
                self.provisional = false;
            }
            self.hold = 0;
            return locked;
        }
        self.hold += 1;
        // Losing the subject needs no hold: the hold already elapsed as
        // `lost_frames`.
        if self.hold < self.params.hold_frames && !lost_long && !urgent {
            return locked;
        }
        self.hold = 0;
        self.provisional = urgent;
        debug!(
            from = ?locked,
            to = ?target,
            lost = lost_long,
            urgent,
            "auto_frame: moving the crop window"
        );
        if self.params.glide_frames == 0 {
            self.locked = Some(target);
            return target;
        }
        self.glide = Some(Glide {
            from: locked,
            frame: 0,
        });
        locked
    }

    /// Window that frames `subject`: padded, with the full view's
    /// aspect, no smaller than `zoom_max` allows, inside the frame.
    fn window_for(&self, subject: SubjectBox) -> Region {
        let p = &self.params;
        let aspect = self.full.width / self.full.height;
        // Grow the shorter axis of the padded box to the window aspect.
        let half_width = subject.hw * (1.0 + p.padding);
        let half_height = subject.hh * (1.0 + p.padding);
        let min_hw = self.full.width * 0.5 / p.zoom_max.max(1.0);
        let max_hw = self.full.width * 0.5;
        let hw = half_width.max(half_height * aspect).clamp(min_hw, max_hw);
        let hh = hw / aspect;
        let cx = subject.cx.clamp(hw, self.space_w - hw);
        let cy = subject.cy.clamp(hh, self.space_h - hh);
        Region {
            x: cx - hw,
            y: cy - hh,
            width: 2.0 * hw,
            height: 2.0 * hh,
        }
    }
}

fn ema(prev: SubjectBox, current: SubjectBox, smoothing: f32) -> SubjectBox {
    let mix = |a: f32, b: f32| a * smoothing + b * (1.0 - smoothing);
    SubjectBox {
        cx: mix(prev.cx, current.cx),
        cy: mix(prev.cy, current.cy),
        hw: mix(prev.hw, current.hw),
        hh: mix(prev.hh, current.hh),
    }
}

/// How far `target` is from `window`: the larger of the centre shift
/// (per axis, relative to the window size) and the relative size change.
fn deviation(window: &Region, target: &Region) -> f32 {
    let dx = (target.center_x() - window.center_x()).abs() / window.width;
    let dy = (target.center_y() - window.center_y()).abs() / window.height;
    let ds = (target.width - window.width).abs() / window.width;
    dx.max(dy).max(ds)
}

/// Ease-in/ease-out curve on `[0, 1]`.
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 800;
    const H: u32 = 448;

    fn params() -> TrackerParams {
        TrackerParams {
            padding: 0.15,
            smoothing: 0.5,
            zoom_max: 2.0,
            deadband: 0.12,
            hold_frames: 30,
            glide_frames: 24,
            lost_frames: 60,
        }
    }

    fn tracker() -> SubjectTracker {
        SubjectTracker::new(params(), W, H, W as f32 / H as f32)
    }

    fn subject(cx: f32, cy: f32) -> SubjectBox {
        // Padded and fitted to 16:9 this is narrower than the 400 px
        // minimum `zoom_max = 2` allows, so windows are 400 × 224.
        SubjectBox {
            cx,
            cy,
            hw: 60.0,
            hh: 90.0,
        }
    }

    /// Feed `frames` observations from `f(frame)` and count the frames
    /// on which the window differs from the previous one.
    fn run(
        t: &mut SubjectTracker,
        frames: u32,
        f: impl Fn(u32) -> Option<SubjectBox>,
    ) -> (u32, Region) {
        let mut moves = 0;
        let mut last = t.update(f(0), false);
        for i in 1..frames {
            let r = t.update(f(i), false);
            if !r.approx_eq(&last, 1e-3) {
                moves += 1;
            }
            last = r;
        }
        (moves, last)
    }

    fn assert_inside(r: &Region) {
        assert!(r.x >= -1e-3 && r.y >= -1e-3, "{r:?}");
        assert!(r.x + r.width <= W as f32 + 1e-3, "{r:?}");
        assert!(r.y + r.height <= H as f32 + 1e-3, "{r:?}");
        assert!(
            (r.width / r.height - W as f32 / H as f32).abs() < 1e-3,
            "{r:?}"
        );
    }

    #[test]
    fn jitter_inside_dead_zone_never_moves_the_window() {
        let mut t = tracker();
        // ±15 px of per-frame jitter around a fixed position.
        let (moves, last) = run(&mut t, 300, |i| {
            let j = [(-15.0, 10.0), (12.0, -8.0), (5.0, 15.0), (-10.0, -12.0)][(i % 4) as usize];
            Some(subject(400.0 + j.0, 250.0 + j.1))
        });
        assert_eq!(moves, 0);
        assert_inside(&last);
    }

    #[test]
    fn sustained_shift_moves_once_after_hold() {
        let mut t = tracker();
        run(&mut t, 30, |_| Some(subject(400.0, 250.0)));
        let start = t.update(Some(subject(400.0, 250.0)), false);
        // Person moves left by 120 px and stays.
        let mut first_move = None;
        let mut last = start;
        for i in 0..120 {
            let r = t.update(Some(subject(280.0, 250.0)), false);
            if first_move.is_none() && !r.approx_eq(&start, 1e-3) {
                first_move = Some(i);
            }
            last = r;
        }
        let first_move = first_move.expect("window must move");
        assert!(
            first_move >= 29,
            "moved after {first_move} frames, before hold"
        );
        assert!(!t.is_gliding());
        assert!((last.center_x() - 280.0).abs() < 5.0, "{last:?}");
        assert_inside(&last);
    }

    #[test]
    fn short_excursion_is_ignored() {
        let mut t = tracker();
        let (moves, _) = run(&mut t, 200, |i| {
            // 10 frames (< hold) away, then back.
            let cx = if (50..60).contains(&i) { 250.0 } else { 400.0 };
            Some(subject(cx, 250.0))
        });
        assert_eq!(moves, 0);
    }

    #[test]
    fn urgent_update_skips_the_hold() {
        let mut t = tracker();
        run(&mut t, 30, |_| Some(subject(400.0, 250.0)));
        let start = t.update(Some(subject(400.0, 250.0)), false);
        // Smoothing 0.5 needs a couple of frames to leave the dead zone;
        // after that the move must start at once, not after 30 frames.
        let moved_at = (0..10)
            .position(|_| {
                !t.update(Some(subject(250.0, 250.0)), true)
                    .approx_eq(&start, 1e-3)
            })
            .expect("urgent move must start within a few frames");
        assert!(moved_at < 5, "moved after {moved_at} frames");
    }

    /// A window that leaves a subject cut off must move even when the
    /// target is within the dead zone.  Otherwise a seated person cut off
    /// at the bottom keeps the window still: the extended box puts the
    /// target near the window, the true extent is never measured, and the
    /// window never zooms back in.
    #[test]
    fn cut_off_subject_moves_the_window_within_the_dead_zone() {
        let mut t = tracker();
        run(&mut t, 30, |_| Some(subject(400.0, 250.0)));
        let start = t.update(Some(subject(400.0, 250.0)), false);
        // 20 px off a 400 px window: 5 %, well inside the 12 % dead zone.
        let near = Some(subject(420.0, 250.0));
        assert!(
            (0..60).all(|_| t.update(near, false).approx_eq(&start, 1e-3)),
            "a plain shift inside the dead zone must not move the window"
        );
        let moved = (0..10).any(|_| !t.update(near, true).approx_eq(&start, 1e-3));
        assert!(moved, "a cut-off subject must move the window");
    }

    /// A window placed for a cut-off subject comes from a guess at its
    /// extent.  Once the subject is measured in full, the window settles
    /// on it even when the correction is inside the dead zone; otherwise
    /// it stays at whatever zoom the guess produced.
    #[test]
    fn window_placed_for_a_cut_off_subject_settles_within_the_dead_zone() {
        let mut t = tracker();
        run(&mut t, 30, |_| Some(subject(400.0, 250.0)));
        let exact = t.update(Some(subject(400.0, 250.0)), false);
        // The guess: taller than the subject, so the window is 440 px
        // wide instead of 400 — 10 % off, inside the 12 % dead zone.
        let guess = Some(SubjectBox {
            hh: 107.0,
            ..subject(400.0, 250.0)
        });
        let mut placed = exact;
        for _ in 0..60 {
            placed = t.update(guess, true);
        }
        assert!(!t.is_gliding());
        assert!((placed.width - 440.0).abs() < 5.0, "{placed:?}");
        // Measured in full again: the window must return to the exact fit.
        let mut last = placed;
        for _ in 0..120 {
            last = t.update(Some(subject(400.0, 250.0)), false);
        }
        assert!(last.approx_eq(&exact, 1.0), "{last:?} vs {exact:?}");
        // Settled: jitter inside the dead zone no longer moves it.
        let (moves, _) = run(&mut t, 120, |i| {
            Some(subject(
                400.0 + if i % 2 == 0 { 10.0 } else { -10.0 },
                250.0,
            ))
        });
        assert_eq!(moves, 0);
    }

    #[test]
    fn glide_is_monotonic_and_smooth() {
        let mut t = tracker();
        run(&mut t, 30, |_| Some(subject(400.0, 250.0)));
        let mut xs = Vec::new();
        for _ in 0..120 {
            xs.push(t.update(Some(subject(250.0, 250.0)), false).center_x());
        }
        assert!(xs.windows(2).all(|w| w[1] <= w[0] + 1e-3), "{xs:?}");
        let max_step = xs.windows(2).map(|w| w[0] - w[1]).fold(0.0, f32::max);
        // 150 px over 24 frames with smoothstep: peak ~1.5 × average.
        assert!(max_step < 12.0, "step {max_step}");
    }

    #[test]
    fn lost_subject_returns_to_full_view() {
        let mut t = tracker();
        run(&mut t, 30, |_| Some(subject(400.0, 250.0)));
        assert!(
            !t.update(Some(subject(400.0, 250.0)), false)
                .approx_eq(&t.full_view(), 1.0)
        );
        let (_, last) = run(&mut t, 60 + 24 + 5, |_| None);
        assert!(last.approx_eq(&t.full_view(), 1e-3), "{last:?}");
    }

    #[test]
    fn brief_dropout_holds_the_window() {
        let mut t = tracker();
        let (moves, _) = run(&mut t, 200, |i| {
            if (80..90).contains(&i) {
                None
            } else {
                Some(subject(400.0, 250.0))
            }
        });
        assert_eq!(moves, 0);
    }

    #[test]
    fn windows_stay_inside_frame_near_edges() {
        let mut t = tracker();
        let (_, last) = run(&mut t, 200, |_| Some(subject(5.0, 440.0)));
        assert_inside(&last);
    }

    #[test]
    fn zoom_max_bounds_the_window_size() {
        let mut t = tracker();
        let tiny = SubjectBox {
            cx: 400.0,
            cy: 224.0,
            hw: 5.0,
            hh: 5.0,
        };
        let r = t.update(Some(tiny), false);
        assert!((r.width - W as f32 / 2.0).abs() < 1e-3, "{r:?}");
    }

    #[test]
    fn full_view_respects_requested_aspect() {
        // 4:3 space framed as 16:9 → centred band.
        let t = SubjectTracker::new(params(), 1280, 960, 16.0 / 9.0);
        let full = t.full_view();
        assert!((full.height - 720.0).abs() < 1e-3);
        assert!((full.y - 120.0).abs() < 1e-3);
    }

    fn mask_with(w: usize, h: usize, rect: (usize, usize, usize, usize)) -> Vec<f32> {
        let mut m = vec![0.0f32; w * h];
        for y in rect.1..rect.3 {
            for x in rect.0..rect.2 {
                m[y * w + x] = 1.0;
            }
        }
        m
    }

    #[test]
    fn meter_finds_a_solid_block() {
        let mut meter = SubjectMeter::new();
        let m = mask_with(200, 100, (50, 20, 150, 100));
        let b = meter.measure(&m, 200, 100, 0.5).expect("subject");
        assert!((b.cx - 100.0).abs() < 4.0, "{b:?}");
        assert!((b.hw - 50.0).abs() < 4.0, "{b:?}");
        assert!(b.cy - b.hh <= 21.0, "top must survive: {b:?}");
    }

    #[test]
    fn meter_ignores_stray_pixels() {
        let mut meter = SubjectMeter::new();
        let mut m = mask_with(200, 100, (50, 20, 150, 100));
        // Isolated noise far left and an arm-thin line to the right.
        m[50 * 200 + 2] = 1.0;
        for x in 150..195 {
            m[60 * 200 + x] = 1.0;
        }
        let b = meter.measure(&m, 200, 100, 0.5).expect("subject");
        assert!(b.cx - b.hw > 45.0, "noise widened the box: {b:?}");
        assert!(b.cx + b.hw < 155.0, "arm widened the box: {b:?}");
    }

    #[test]
    fn meter_reports_none_for_noise_only() {
        let mut meter = SubjectMeter::new();
        let mut m = vec![0.0f32; 200 * 100];
        m[10] = 1.0;
        assert!(meter.measure(&m, 200, 100, 0.5).is_none());
        assert!(
            meter
                .measure(&vec![0.0; 200 * 100], 200, 100, 0.5)
                .is_none()
        );
    }
}
