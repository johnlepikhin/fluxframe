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
//!
//! # Device enumeration helpers
//!
//! This module also exposes [`enumerate_capture_devices`] (with the
//! testable [`enumerate_capture_devices_in`] variant) and
//! [`pick_input_device`].  They classify devices via `VIDIOC_QUERYCAP`
//! ioctl — the authoritative kernel-side capability flags.
//!
//! ## Why two enumerators coexist (with [`crate::v4l2`])
//!
//! [`crate::v4l2::enumerate_devices`] reads `/sys/class/video4linux`
//! and classifies devices by `modalias` + name substring heuristics.
//! It is cheap (no `open` per device) and used where coarse
//! input/virtual/unknown classification is enough — `list` / `check`
//! UI scenarios where a camera lacking the word "camera" in its name
//! would land in `Unknown` and the operator picks anyway.
//!
//! The helpers here `open()` the device and ask the kernel via
//! `VIDIOC_QUERYCAP` — strictly more authoritative.  A driver that
//! does NOT contain "camera"/"webcam" in its name (some industrial
//! cams, IP cam shims) shows up as `Unknown` to the sysfs classifier
//! but is correctly detected as a capture device here.  Used in the
//! hot path where we MUST pick a real camera (auto-select in `run`).
//!
//! Drift mitigation: both treat `v4l2loopback` as non-input —
//! sysfs classifier via `modalias`, capability classifier via
//! `VIDEO_OUTPUT` flag (which loopback always advertises).

use std::path::{Path, PathBuf};

use fluxframe_core::error::PipelineError;
use fluxframe_core::frame::PixelFormat;
use v4l::capability::Flags as CapFlags;
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
    let device =
        Device::with_path(device_path).map_err(|e| PipelineError::InputDeviceUnavailable {
            device: device_path.display().to_string(),
            reason: format!("v4l::Device::with_path failed: {e}"),
            hint: "ensure the device exists and is readable".into(),
        })?;
    let formats =
        Capture::enum_formats(&device).map_err(|e| PipelineError::InputDeviceUnavailable {
            device: device_path.display().to_string(),
            reason: format!("VIDIOC_ENUM_FMT failed: {e}"),
            hint: "check the device is readable and not held open by another process \
                   (e.g. cheese, chrome)"
                .into(),
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
                let intervals = match Capture::enum_frameintervals(&device, fmt.fourcc, w, h) {
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

    select_best(candidates, prefer_fps).ok_or_else(|| PipelineError::InputDeviceUnavailable {
        device: device_path.display().to_string(),
        reason: "no acceptable mode after policy filter (this is a bug)".into(),
        hint: "report this — `select_best` should never return None on non-empty input".into(),
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

/// Enumerate `/dev/video*` nodes and return those that look like
/// real capture cameras (have `V4L2_CAP_VIDEO_CAPTURE`, do NOT have
/// `V4L2_CAP_VIDEO_OUTPUT`).
///
/// Skips:
/// * Non-existent paths (the directory is scanned only — no globbing).
/// * Devices that cannot be opened (permissions, busy) — they are
///   re-tried on the next call.
/// * Devices that advertise `VIDEO_OUTPUT` capability (the v4l2loopback
///   pseudo-camera we publish into ends up here — preventing a
///   feedback loop).
///
/// Order: lexicographic by path so `pick_input_device` is deterministic.
/// Best-effort and non-failing: a sysfs/devfs hiccup returns an empty
/// list rather than an error, matching the "never hard-fail in `auto`
/// mode" rule.
///
/// Each candidate is opened EXACTLY ONCE per enumeration tick.
/// [`pick_input_device`] does NOT re-probe — it filters the already-
/// vetted list.
#[must_use]
pub fn enumerate_capture_devices() -> Vec<PathBuf> {
    enumerate_capture_devices_in(Path::new("/dev"))
}

/// Testable form of [`enumerate_capture_devices`].
///
/// `dev_root` is the directory holding the `videoN` character-device
/// nodes (production: `/dev`).  Exposed so unit tests can drive a
/// synthetic layout (note: meaningful behaviour still requires real
/// V4L2 devices because the capability probe goes through the kernel —
/// the seam is here for the future when we mock the open path).
#[must_use]
pub fn enumerate_capture_devices_in(dev_root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dev_root) else {
        tracing::debug!(?dev_root, "dev root unreadable; returning empty list");
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name();
            let s = name.to_str()?;
            if !is_video_node_name(s) {
                return None;
            }
            Some(e.path())
        })
        .filter(|p| probe_capture_device(p))
        .collect();
    paths.sort();
    paths
}

/// Returns `true` when `name` matches `videoN` where N is one or more
/// ASCII digits.  Used to filter `/dev` entries that look like V4L2
/// device nodes (rejects `videoX-aux`, `video-codec`, `vide0`, etc.).
fn is_video_node_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("video") else {
        return false;
    };
    !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
}

/// Probe `path` and return `true` iff it is a real capture-capable
/// V4L2 device that is NOT also an output device.
///
/// Logs at `debug` level for every skipped candidate so the operator
/// can diagnose "auto-select found nothing" without strace.
fn probe_capture_device(path: &Path) -> bool {
    let device = match Device::with_path(path) {
        Ok(d) => d,
        Err(e) => {
            tracing::debug!(path = %path.display(), error = %e, "device probe failed");
            return false;
        }
    };
    let caps = match device.query_caps() {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!(path = %path.display(), error = %e, "device probe failed");
            return false;
        }
    };
    let flags = caps.capabilities;
    if !flags.contains(CapFlags::VIDEO_CAPTURE) {
        tracing::debug!(
            path = %path.display(),
            "missing VIDEO_CAPTURE capability",
        );
        return false;
    }
    if flags.contains(CapFlags::VIDEO_OUTPUT) {
        tracing::debug!(
            path = %path.display(),
            "has VIDEO_OUTPUT capability — likely v4l2loopback writer side",
        );
        return false;
    }
    true
}

/// Pick the first `candidate` that is not present in `exclude`.
///
/// Pure function — performs NO filesystem or device I/O.  The contract
/// is that `candidates` was already open-probed (via
/// [`enumerate_capture_devices`] or its `_in` variant) so the only
/// remaining task here is to filter out caller-supplied exclusions
/// (typically: the output `v4l2loopback` device).
///
/// Returns `None` when every candidate is excluded — the caller is
/// expected to retry later (a fresh enumeration may surface a new
/// camera).  Emits a `tracing::debug!` for each excluded candidate so
/// the operator can see "all candidates excluded, returned None".
#[must_use]
pub fn pick_input_device(candidates: &[PathBuf], exclude: &[PathBuf]) -> Option<PathBuf> {
    for candidate in candidates {
        if exclude.iter().any(|x| x == candidate) {
            tracing::debug!(
                path = %candidate.display(),
                "candidate excluded by caller",
            );
            continue;
        }
        return Some(candidate.clone());
    }
    None
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

    #[test]
    fn is_video_node_name_accepts_canonical_forms() {
        for (name, expected) in [
            ("video0", true),
            ("video10", true),
            ("video", false),   // no digits
            ("video0a", false), // trailing non-digit
            ("video-codec", false),
            ("videoX-aux", false),
            ("vide0", false), // typo
            ("foo", false),
        ] {
            assert_eq!(
                is_video_node_name(name),
                expected,
                "is_video_node_name({name:?}) wrong",
            );
        }
    }

    #[test]
    fn pick_returns_first_non_excluded() {
        let candidates = vec![
            PathBuf::from("/dev/video0"),
            PathBuf::from("/dev/video1"),
            PathBuf::from("/dev/video2"),
        ];
        let exclude = vec![PathBuf::from("/dev/video0")];
        assert_eq!(
            pick_input_device(&candidates, &exclude),
            Some(PathBuf::from("/dev/video1")),
        );
    }

    #[test]
    fn pick_returns_none_when_all_excluded() {
        let candidates = vec![PathBuf::from("/dev/video0"), PathBuf::from("/dev/video1")];
        let exclude = candidates.clone();
        assert_eq!(pick_input_device(&candidates, &exclude), None);
    }

    #[test]
    fn pick_returns_none_on_empty_candidates() {
        let candidates: Vec<PathBuf> = Vec::new();
        let exclude = vec![PathBuf::from("/dev/video0")];
        assert_eq!(pick_input_device(&candidates, &exclude), None);
    }

    #[test]
    fn pick_handles_exclude_not_in_candidates() {
        let candidates = vec![PathBuf::from("/dev/video0"), PathBuf::from("/dev/video1")];
        let exclude = vec![PathBuf::from("/dev/video99")];
        assert_eq!(
            pick_input_device(&candidates, &exclude),
            Some(PathBuf::from("/dev/video0")),
        );
    }
}
