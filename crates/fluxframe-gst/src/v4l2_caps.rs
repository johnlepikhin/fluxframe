//! V4L2 camera capability discovery — picks a native mode at startup so
//! the input pipeline never silently stretches content.
//!
//! Operators should not need to know their camera's native mode list
//! (different cameras advertise different mode tables, and asking for
//! something the camera does not natively provide forces `videoscale`
//! to anisotropically scale — content ends up visually distorted).
//!
//! Selection policy (highest priority first):
//! 1. **Prefer raw formats** (`YUYV`, `NV12`) over compressed (`MJPG`,
//!    `H264`) so we avoid the CPU cost of decode and the artefacts of
//!    re-encoding.
//! 2. **Prefer the requested fps** (or the highest fps that does not
//!    exceed the requested ceiling).
//! 3. **Prefer the largest resolution** within the format and fps
//!    constraints — operators almost always want the camera's full
//!    detail (downscaling for output is cheap; upscaling cannot
//!    invent detail).
//! 4. Cap at 1920×1080 by default — 4K capture pushes CPU budgets we
//!    haven't validated and is out of scope for the MVP.
//!
//! If no matching mode exists the function returns an error explaining
//! what the camera does advertise.

use std::path::Path;

use fluxframe_core::error::PipelineError;
use fluxframe_core::frame::PixelFormat;
use v4l::video::Capture;
use v4l::{Device, FourCC};

/// Upper bound on the resolution auto-detect will pick.  4K+ capture
/// is outside MVP CPU/memory budgets; operators who explicitly want
/// it can still force the mode via `[input] width/height` overrides.
pub const AUTODETECT_MAX_WIDTH: u32 = 1920;
/// See [`AUTODETECT_MAX_WIDTH`].
pub const AUTODETECT_MAX_HEIGHT: u32 = 1080;

/// Slack in frames-per-second granted when filtering out modes that
/// fall below `prefer_fps`.  A camera advertising 25fps survives a
/// request for 30fps; one advertising 5fps does not.
pub const FPS_TOLERANCE: u32 = 5;

/// A concrete v4l2 capture mode the input pipeline can pin via caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeMode {
    /// Frame width in pixels (as advertised by `VIDIOC_ENUM_FRAMESIZES`).
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format the camera natively delivers.
    pub format: PixelFormat,
    /// Frame rate to request (preferred fps clamped to what the mode
    /// supports).
    pub fps: u32,
}

/// Pick the best native mode for `device` honouring `prefer_fps` as
/// an upper bound.  See the module header for the selection policy.
///
/// # Errors
///
/// Returns [`PipelineError::InputDeviceUnavailable`] when the device
/// cannot be opened or when no mode is acceptable (no raw format,
/// or every mode exceeds the auto-detect cap).
pub fn detect_native_mode(
    device_path: &Path,
    prefer_fps: u32,
) -> Result<NativeMode, PipelineError> {
    let device = Device::with_path(device_path).map_err(|e| {
        PipelineError::InputDeviceUnavailable {
            device: device_path.display().to_string(),
            reason: format!("v4l::Device::with_path failed: {e}"),
            hint: "ensure the device exists and is readable".into(),
        }
    })?;
    let formats = Capture::enum_formats(&device).map_err(|e| {
        PipelineError::InputDeviceUnavailable {
            device: device_path.display().to_string(),
            reason: format!("VIDIOC_ENUM_FMT failed: {e}"),
            hint: "check the device is readable and not held open by another process \
                   (e.g. cheese, chrome)"
                .into(),
        }
    })?;

    let mut candidates: Vec<NativeMode> = Vec::new();
    for fmt in &formats {
        let Some(pixel_format) = fourcc_to_pixel_format(fmt.fourcc) else {
            // Unsupported / compressed format we don't model.
            continue;
        };
        let sizes = match Capture::enum_framesizes(&device, fmt.fourcc) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    fourcc = %fmt.fourcc,
                    error = %e,
                    "VIDIOC_ENUM_FRAMESIZES failed; format skipped",
                );
                continue;
            }
        };
        for size in sizes {
            for d in size.size.to_discrete() {
                let (w, h) = (d.width, d.height);
                if w > AUTODETECT_MAX_WIDTH || h > AUTODETECT_MAX_HEIGHT {
                    continue;
                }
                // Pick the highest fps the mode supports that does not
                // exceed `prefer_fps`.  An empty list means the driver
                // does not enumerate intervals — treat as "any fps OK"
                // and assume `prefer_fps`.
                let intervals = match Capture::enum_frameintervals(
                    &device, fmt.fourcc, w, h,
                ) {
                    Ok(i) => i,
                    Err(e) => {
                        tracing::warn!(
                            fourcc = %fmt.fourcc,
                            width = w,
                            height = h,
                            error = %e,
                            "VIDIOC_ENUM_FRAMEINTERVALS failed; \
                             assuming preferred fps",
                        );
                        Vec::new()
                    }
                };
                let fps = pick_fps(&intervals, prefer_fps);
                candidates.push(NativeMode {
                    width: w,
                    height: h,
                    format: pixel_format,
                    fps,
                });
            }
        }
    }

    if candidates.is_empty() {
        let advertised: Vec<String> = formats
            .iter()
            .map(|f| format!("{} ({})", f.fourcc, f.description))
            .collect();
        return Err(PipelineError::InputDeviceUnavailable {
            device: device_path.display().to_string(),
            reason: format!(
                "no usable raw-format mode under {AUTODETECT_MAX_WIDTH}x{AUTODETECT_MAX_HEIGHT}; \
                 advertised: [{}]",
                advertised.join(", ")
            ),
            hint: "set [input] format to one of YUY2/NV12/RGB/BGR/RGBA/GRAY8 if the device offers it, \
                   or set [input] width/height explicitly to override auto-detect"
                .into(),
        });
    }

    select_best(candidates, prefer_fps).ok_or_else(|| {
        PipelineError::InputDeviceUnavailable {
            device: device_path.display().to_string(),
            reason: "no acceptable mode after policy filter (this is a bug)".into(),
            hint: "report this — `select_best` should never return None on non-empty input"
                .into(),
        }
    })
}

/// Pick the best mode from a candidate list given a target fps.
/// Pure function — no I/O, fully unit-testable.
///
/// Selection policy:
/// 1. Filter out modes whose fps is more than [`FPS_TOLERANCE`] below
///    `prefer_fps` (a 5fps slack so 25fps survives a request for 30).
/// 2. If filtering left nothing, fall back to highest-fps candidate.
/// 3. Otherwise prefer largest area; tiebreak by closest fps to
///    `prefer_fps`; tiebreak again by raw-YUV > raw-RGB > Gray
///    (saves a `videoconvert` step on typical cameras).
fn select_best(mut candidates: Vec<NativeMode>, prefer_fps: u32) -> Option<NativeMode> {
    if candidates.is_empty() {
        return None;
    }
    let fps_floor = prefer_fps.saturating_sub(FPS_TOLERANCE);
    let mut filtered: Vec<NativeMode> = candidates
        .iter()
        .copied()
        .filter(|m| m.fps >= fps_floor)
        .collect();
    if filtered.is_empty() {
        // No mode meets the fps target — fall back to all candidates
        // sorted by fps-descending and pick the highest.  Operator
        // sees the chosen fps in the auto-detected log line and can
        // adjust if needed.
        candidates.sort_by(|a, b| b.fps.cmp(&a.fps));
        return Some(candidates[0]);
    }

    // Within fps-acceptable modes, prefer largest area, then closest
    // fps to `prefer_fps`, then raw-YUV > raw-RGB (saves a
    // videoconvert step for typical V4L2 cameras).
    filtered.sort_by(|a, b| {
        let area_a = u64::from(a.width) * u64::from(a.height);
        let area_b = u64::from(b.width) * u64::from(b.height);
        area_b
            .cmp(&area_a)
            .then_with(|| {
                let da = a.fps.abs_diff(prefer_fps);
                let db = b.fps.abs_diff(prefer_fps);
                da.cmp(&db)
            })
            .then_with(|| format_priority(a.format).cmp(&format_priority(b.format)))
    });

    Some(filtered[0])
}

/// Lower value = higher priority for tie-breaking.
fn format_priority(f: PixelFormat) -> u8 {
    match f {
        PixelFormat::Yuy2 | PixelFormat::Nv12 => 0,
        PixelFormat::Rgb | PixelFormat::Bgr | PixelFormat::Rgba => 1,
        PixelFormat::Gray8 => 2,
    }
}

fn pick_fps(intervals: &[v4l::frameinterval::FrameInterval], prefer_fps: u32) -> u32 {
    use v4l::frameinterval::FrameIntervalEnum;
    let mut best: Option<u32> = None;
    for iv in intervals {
        let candidates: Vec<u32> = match &iv.interval {
            FrameIntervalEnum::Discrete(d) => {
                if d.numerator == 0 {
                    Vec::new()
                } else {
                    vec![d.denominator / d.numerator]
                }
            }
            FrameIntervalEnum::Stepwise(_) => Vec::new(),
        };
        for fps in candidates {
            if fps == 0 {
                continue;
            }
            best = Some(match best {
                None => fps,
                Some(cur) => choose_better_fps(cur, fps, prefer_fps),
            });
        }
    }
    best.unwrap_or(prefer_fps)
}

fn choose_better_fps(a: u32, b: u32, prefer: u32) -> u32 {
    // Prefer the highest fps that does not exceed `prefer`; if both
    // exceed `prefer`, pick the smallest (least over-budget).
    match (a <= prefer, b <= prefer) {
        (true, true) => a.max(b),
        (false, false) => a.min(b),
        (true, false) => a,
        (false, true) => b,
    }
}

fn fourcc_to_pixel_format(fcc: FourCC) -> Option<PixelFormat> {
    match &fcc.repr {
        b"YUYV" => Some(PixelFormat::Yuy2),
        b"NV12" => Some(PixelFormat::Nv12),
        b"RGB3" => Some(PixelFormat::Rgb),
        b"BGR3" => Some(PixelFormat::Bgr),
        b"RGB4" => Some(PixelFormat::Rgba),
        b"GREY" => Some(PixelFormat::Gray8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourcc_round_trip_for_known_formats() {
        for (code, fmt) in [
            (*b"YUYV", PixelFormat::Yuy2),
            (*b"NV12", PixelFormat::Nv12),
            (*b"RGB3", PixelFormat::Rgb),
            (*b"GREY", PixelFormat::Gray8),
        ] {
            assert_eq!(fourcc_to_pixel_format(FourCC::new(&code)), Some(fmt));
        }
    }

    #[test]
    fn fourcc_returns_none_for_compressed() {
        assert_eq!(fourcc_to_pixel_format(FourCC::new(b"MJPG")), None);
        assert_eq!(fourcc_to_pixel_format(FourCC::new(b"H264")), None);
    }

    #[test]
    fn format_priority_prefers_yuv_over_rgb() {
        assert!(format_priority(PixelFormat::Yuy2) < format_priority(PixelFormat::Rgb));
        assert!(format_priority(PixelFormat::Rgb) < format_priority(PixelFormat::Gray8));
    }

    #[test]
    fn choose_better_fps_prefers_highest_within_budget() {
        assert_eq!(choose_better_fps(30, 60, 60), 60);
        assert_eq!(choose_better_fps(30, 15, 30), 30);
        // Both over budget → pick smallest.
        assert_eq!(choose_better_fps(60, 120, 30), 60);
    }

    fn mode(width: u32, height: u32, format: PixelFormat, fps: u32) -> NativeMode {
        NativeMode {
            width,
            height,
            format,
            fps,
        }
    }

    #[test]
    fn select_best_drops_low_fps_modes() {
        // 1080p@5 is below floor (25), 720p@30 and 480p@30 survive.
        // Among survivors 720p has larger area than 480p.
        let cands = vec![
            mode(1920, 1080, PixelFormat::Yuy2, 5),
            mode(1280, 720, PixelFormat::Yuy2, 30),
            mode(640, 480, PixelFormat::Yuy2, 30),
        ];
        let chosen = select_best(cands, 30).expect("non-empty input");
        assert_eq!(chosen.width, 1280);
        assert_eq!(chosen.height, 720);
        assert_eq!(chosen.fps, 30);
    }

    #[test]
    fn select_best_prefers_larger_area_among_acceptable_fps() {
        let cands = vec![
            mode(1280, 720, PixelFormat::Yuy2, 30),
            mode(1920, 1080, PixelFormat::Yuy2, 30),
        ];
        let chosen = select_best(cands, 30).expect("non-empty input");
        assert_eq!(chosen.width, 1920);
        assert_eq!(chosen.height, 1080);
    }

    #[test]
    fn select_best_tiebreaks_by_fps_distance() {
        let cands = vec![
            mode(1280, 720, PixelFormat::Yuy2, 60),
            mode(1280, 720, PixelFormat::Yuy2, 30),
        ];
        let chosen = select_best(cands, 30).expect("non-empty input");
        assert_eq!(chosen.fps, 30);
    }

    #[test]
    fn select_best_tiebreaks_format_prefers_yuv_over_rgb() {
        let cands = vec![
            mode(1280, 720, PixelFormat::Rgb, 30),
            mode(1280, 720, PixelFormat::Yuy2, 30),
        ];
        let chosen = select_best(cands, 30).expect("non-empty input");
        assert_eq!(chosen.format, PixelFormat::Yuy2);
    }

    #[test]
    fn select_best_falls_back_to_highest_fps_when_all_low() {
        // Both modes below floor (25): fallback picks highest fps.
        let cands = vec![
            mode(640, 480, PixelFormat::Yuy2, 5),
            mode(640, 480, PixelFormat::Yuy2, 10),
        ];
        let chosen = select_best(cands, 30).expect("non-empty input");
        assert_eq!(chosen.fps, 10);
    }

    #[test]
    fn select_best_returns_none_on_empty() {
        assert_eq!(select_best(Vec::new(), 30), None);
    }
}
