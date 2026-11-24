//! V4L2 camera capability discovery — picks the native capture mode that
//! best serves the operator's `[input] width/height/fps` and pins it on
//! the source, so the camera never runs in a mode we did not choose.
//!
//! Operators should not need to know their camera's native mode list:
//! the requested geometry is a target, and the camera mode is picked to
//! serve it (the input pipeline's `videoscale` adapts whatever remains,
//! padding instead of stretching).
//!
//! Selection policy (highest priority first), see [`select_best`]:
//! 1. **Frame rate**: the mode must reach the requested fps (with
//!    [`FPS_TOLERANCE`] slack). When no mode does, only the fastest
//!    modes stay in the running.
//! 2. **Size**: exactly the requested size; otherwise the smallest mode
//!    that covers it (downscaling keeps detail, upscaling cannot invent
//!    it); otherwise the largest mode available. Within a size class a
//!    mode with the requested aspect ratio wins, so the output is not
//!    letterboxed.
//! 3. **Fps distance** to the requested fps.
//! 4. **Encoding**: raw formats (`YUYV`, `NV12`, …) over compressed
//!    ones, which cost a decode on the CPU.
//!
//! If the camera advertises no usable mode at all, the function returns
//! an error listing what it does advertise.
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

use crate::frame_conv::pixel_format_to_gst;
use crate::util::caps_i32;

/// Slack in frames-per-second granted when filtering out modes that
/// fall below `prefer_fps`.  A camera advertising 25fps survives a
/// request for 30fps; one advertising 5fps does not.
pub const FPS_TOLERANCE: u32 = 5;

/// Relative aspect-ratio difference under which a mode counts as having
/// the requested aspect ratio (800×448 vs 16:9 differs by ~0.5 %).
const ASPECT_TOLERANCE: f64 = 0.02;

/// How the camera encodes frames on the wire in a given mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureEncoding {
    /// Uncompressed frames in the given pixel format.
    Raw(PixelFormat),
    /// Motion-JPEG: every frame is a JPEG image, decoded on the CPU.
    /// USB 2.0 cameras typically reach 720p/1080p at full frame rate
    /// only in this encoding.
    Mjpeg,
}

impl std::fmt::Display for CaptureEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Raw(format) => write!(f, "raw {format:?}"),
            Self::Mjpeg => f.write_str("MJPEG"),
        }
    }
}

/// A concrete v4l2 capture mode the input pipeline pins via caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeMode {
    /// Frame width in pixels (as advertised by `VIDIOC_ENUM_FRAMESIZES`).
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Wire encoding the camera delivers in this mode.
    pub encoding: CaptureEncoding,
    /// Frame rate to request (preferred fps clamped to what the mode
    /// supports).
    pub fps: u32,
}

impl NativeMode {
    /// Caps that pin `v4l2src` to exactly this mode.
    ///
    /// The frame rate is a narrow range around [`Self::fps`] rather than
    /// an exact fraction: drivers report intervals such as
    /// `333333/10000000` s, which GStreamer may expose as a fraction that
    /// is not exactly `30/1`.
    pub(crate) fn source_caps(&self) -> Result<gstreamer::Caps, PipelineError> {
        let width = caps_i32("width", self.width)?;
        let height = caps_i32("height", self.height)?;
        let fps2 = caps_i32("fps", self.fps.saturating_mul(2))?;
        let framerate = gstreamer::FractionRange::new(
            gstreamer::Fraction::new(fps2.saturating_sub(1).max(1), 2),
            gstreamer::Fraction::new(fps2.saturating_add(1), 2),
        );
        let builder = match self.encoding {
            CaptureEncoding::Raw(format) => gstreamer::Caps::builder("video/x-raw")
                .field("format", pixel_format_to_gst(format).to_str()),
            CaptureEncoding::Mjpeg => gstreamer::Caps::builder("image/jpeg"),
        };
        Ok(builder
            .field("width", width)
            .field("height", height)
            .field("framerate", framerate)
            .build())
    }
}

/// Pick the native mode of `device` that best serves a requested
/// `width × height` at `prefer_fps`.  See the module header for the
/// selection policy.
///
/// # Errors
///
/// Returns [`PipelineError::InputDeviceUnavailable`] when the device
/// cannot be opened or advertises no mode in a supported encoding.
pub fn detect_native_mode(
    device_path: &Path,
    width: u32,
    height: u32,
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
        let Some(encoding) = fourcc_to_encoding(fmt.fourcc) else {
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
                    encoding,
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
                "no capture mode in a supported encoding; advertised: [{}]",
                advertised.join(", ")
            ),
            hint: "the camera must offer one of YUYV/NV12/RGB3/BGR3/RGB4/GREY/MJPG".into(),
        });
    }

    let request = ModeRequest {
        width,
        height,
        fps: prefer_fps,
    };
    select_best(candidates, request).ok_or_else(|| PipelineError::InputDeviceUnavailable {
        device: device_path.display().to_string(),
        reason: "no acceptable mode after policy filter (this is a bug)".into(),
        hint: "report this — `select_best` should never return None on non-empty input".into(),
    })
}

/// What the operator asked the capture to deliver (`[input]` section).
#[derive(Debug, Clone, Copy)]
struct ModeRequest {
    width: u32,
    height: u32,
    fps: u32,
}

/// How well a mode's frame size serves the request; lower is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SizeFit {
    /// Exactly the requested size — no scaling at all.
    Exact,
    /// Larger than requested in both axes — downscaled, detail kept.
    Covers,
    /// Smaller in at least one axis — upscaled, detail lost.
    Short,
}

/// Pick the best mode from a candidate list for `request`.
/// Pure function — no I/O, fully unit-testable.  The policy is spelled
/// out in the module header.
fn select_best(candidates: Vec<NativeMode>, request: ModeRequest) -> Option<NativeMode> {
    let max_fps = candidates.iter().map(|m| m.fps).max()?;
    // When no mode reaches the fps target, the fastest modes are the
    // only ones left in the running: the operator sees the chosen fps in
    // the log line and can lower the requested size.
    let fps_floor = request.fps.saturating_sub(FPS_TOLERANCE).min(max_fps);
    candidates
        .into_iter()
        .filter(|m| m.fps >= fps_floor)
        .min_by_key(|m| {
            let fit = size_fit(m, request);
            let area = i64::from(m.width) * i64::from(m.height);
            // Covering modes: the smallest one is enough (less to
            // decode and downscale). Short modes: the largest one loses
            // the least detail.
            let area_rank = match fit {
                SizeFit::Exact => 0,
                SizeFit::Covers => area,
                SizeFit::Short => -area,
            };
            (
                fit,
                !same_aspect(m, request),
                area_rank,
                m.fps.abs_diff(request.fps),
                encoding_priority(m.encoding),
            )
        })
}

fn size_fit(mode: &NativeMode, request: ModeRequest) -> SizeFit {
    if mode.width == request.width && mode.height == request.height {
        SizeFit::Exact
    } else if mode.width >= request.width && mode.height >= request.height {
        SizeFit::Covers
    } else {
        SizeFit::Short
    }
}

fn same_aspect(mode: &NativeMode, request: ModeRequest) -> bool {
    if mode.height == 0 || request.height == 0 {
        return false;
    }
    let mode_aspect = f64::from(mode.width) / f64::from(mode.height);
    let request_aspect = f64::from(request.width) / f64::from(request.height);
    ((mode_aspect - request_aspect) / request_aspect).abs() <= ASPECT_TOLERANCE
}

/// Lower value = higher priority for tie-breaking.
fn encoding_priority(encoding: CaptureEncoding) -> u8 {
    // Compressed modes lose to every known raw format: decoding costs
    // CPU and JPEG adds artefacts. They still win whenever they serve
    // the request better (rules 1–3), which is their whole point.
    const MJPEG_PRIORITY: u8 = 16;
    match encoding {
        CaptureEncoding::Raw(format) => format_priority(format),
        CaptureEncoding::Mjpeg => MJPEG_PRIORITY,
    }
}

/// Lower value = higher priority for tie-breaking.
fn format_priority(f: PixelFormat) -> u8 {
    match f {
        PixelFormat::Yuy2 | PixelFormat::Nv12 => 0,
        PixelFormat::Rgb | PixelFormat::Bgr | PixelFormat::Rgba => 1,
        PixelFormat::Gray8 => 2,
        // `PixelFormat` is `#[non_exhaustive]`; unknown future
        // variants get the lowest priority so they are never
        // accidentally picked over a real V4L2 format.
        _ => u8::MAX,
    }
}

fn pick_fps(intervals: &[v4l::frameinterval::FrameInterval], prefer_fps: u32) -> u32 {
    use v4l::frameinterval::FrameIntervalEnum;
    let mut best: Option<u32> = None;
    for iv in intervals {
        let candidates: Vec<u32> = match &iv.interval {
            FrameIntervalEnum::Discrete(d) => {
                d.denominator.checked_div(d.numerator).into_iter().collect()
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

fn fourcc_to_encoding(fcc: FourCC) -> Option<CaptureEncoding> {
    let raw = match &fcc.repr {
        b"MJPG" | b"JPEG" => return Some(CaptureEncoding::Mjpeg),
        b"YUYV" => PixelFormat::Yuy2,
        b"NV12" => PixelFormat::Nv12,
        b"RGB3" => PixelFormat::Rgb,
        b"BGR3" => PixelFormat::Bgr,
        b"RGB4" => PixelFormat::Rgba,
        b"GREY" => PixelFormat::Gray8,
        _ => return None,
    };
    Some(CaptureEncoding::Raw(raw))
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
            assert_eq!(
                fourcc_to_encoding(FourCC::new(&code)),
                Some(CaptureEncoding::Raw(fmt))
            );
        }
    }

    #[test]
    fn fourcc_maps_mjpeg_and_rejects_other_compressed() {
        assert_eq!(
            fourcc_to_encoding(FourCC::new(b"MJPG")),
            Some(CaptureEncoding::Mjpeg)
        );
        assert_eq!(fourcc_to_encoding(FourCC::new(b"H264")), None);
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
            encoding: CaptureEncoding::Raw(format),
            fps,
        }
    }

    fn req(width: u32, height: u32, fps: u32) -> ModeRequest {
        ModeRequest { width, height, fps }
    }

    fn size(m: NativeMode) -> (u32, u32) {
        (m.width, m.height)
    }

    /// Mode table of the UGREEN 2K camera's YUYV path (USB 2.0 caps the
    /// large modes' frame rate).
    fn ugreen_yuyv() -> Vec<NativeMode> {
        vec![
            mode(640, 480, PixelFormat::Yuy2, 30),
            mode(800, 448, PixelFormat::Yuy2, 30),
            mode(864, 480, PixelFormat::Yuy2, 25),
            mode(1280, 720, PixelFormat::Yuy2, 10),
            mode(1920, 1080, PixelFormat::Yuy2, 5),
        ]
    }

    #[test]
    fn select_best_prefers_exact_size_over_larger() {
        // The old policy picked the largest area (864×480@25) and ran the
        // camera at 25 fps although 800×448@30 was requested.
        let chosen = select_best(ugreen_yuyv(), req(800, 448, 30)).expect("non-empty");
        assert_eq!(size(chosen), (800, 448));
        assert_eq!(chosen.fps, 30);
    }

    #[test]
    fn select_best_drops_low_fps_modes() {
        // 1280×720 exists but only at 10 fps: below the floor (25).
        // Nothing covers 720p at full rate, so the largest fast mode wins.
        let chosen = select_best(ugreen_yuyv(), req(1280, 720, 30)).expect("non-empty");
        assert_eq!(size(chosen), (864, 480));
    }

    fn mjpeg(width: u32, height: u32, fps: u32) -> NativeMode {
        NativeMode {
            width,
            height,
            encoding: CaptureEncoding::Mjpeg,
            fps,
        }
    }

    #[test]
    fn select_best_takes_mjpeg_when_raw_is_too_slow() {
        let mut cands = ugreen_yuyv();
        cands.extend([mjpeg(1280, 720, 30), mjpeg(1920, 1080, 30)]);
        let chosen = select_best(cands, req(1280, 720, 30)).expect("non-empty");
        assert_eq!(size(chosen), (1280, 720));
        assert_eq!(chosen.encoding, CaptureEncoding::Mjpeg);
        assert_eq!(chosen.fps, 30);
    }

    #[test]
    fn select_best_prefers_raw_when_both_serve_the_request() {
        let mut cands = ugreen_yuyv();
        cands.extend([mjpeg(800, 448, 30), mjpeg(1280, 720, 30)]);
        let chosen = select_best(cands, req(800, 448, 30)).expect("non-empty");
        assert_eq!(size(chosen), (800, 448));
        assert_eq!(chosen.encoding, CaptureEncoding::Raw(PixelFormat::Yuy2));
    }

    #[test]
    fn select_best_uses_mjpeg_only_camera() {
        let cands = vec![mjpeg(1280, 720, 30), mjpeg(640, 480, 30)];
        let chosen = select_best(cands, req(640, 480, 30)).expect("non-empty");
        assert_eq!(size(chosen), (640, 480));
        assert_eq!(chosen.encoding, CaptureEncoding::Mjpeg);
    }

    #[test]
    fn source_caps_for_mjpeg_are_image_jpeg() {
        crate::init().expect("gstreamer init");
        let caps = mjpeg(1280, 720, 30).source_caps().expect("caps");
        let s = caps.structure(0).expect("one structure");
        assert_eq!(s.name(), "image/jpeg");
        assert!(!s.has_field("format"));
        assert_eq!(s.get::<i32>("width").expect("width"), 1280);
    }

    #[test]
    fn select_best_prefers_smallest_covering_mode() {
        let cands = vec![
            mode(1920, 1080, PixelFormat::Yuy2, 30),
            mode(1280, 720, PixelFormat::Yuy2, 30),
            mode(640, 360, PixelFormat::Yuy2, 30),
        ];
        let chosen = select_best(cands, req(960, 540, 30)).expect("non-empty");
        assert_eq!(size(chosen), (1280, 720));
    }

    #[test]
    fn select_best_covering_prefers_requested_aspect() {
        // 1280×960 is smaller, but 4:3 would be letterboxed into 16:9.
        let cands = vec![
            mode(1280, 960, PixelFormat::Yuy2, 30),
            mode(1920, 1080, PixelFormat::Yuy2, 30),
        ];
        let chosen = select_best(cands, req(1280, 720, 30)).expect("non-empty");
        assert_eq!(size(chosen), (1920, 1080));
    }

    #[test]
    fn select_best_short_prefers_largest_area() {
        let cands = vec![
            mode(640, 360, PixelFormat::Yuy2, 30),
            mode(1280, 720, PixelFormat::Yuy2, 30),
        ];
        let chosen = select_best(cands, req(1920, 1080, 30)).expect("non-empty");
        assert_eq!(size(chosen), (1280, 720));
    }

    #[test]
    fn select_best_tiebreaks_by_fps_distance() {
        let cands = vec![
            mode(1280, 720, PixelFormat::Yuy2, 60),
            mode(1280, 720, PixelFormat::Yuy2, 30),
        ];
        let chosen = select_best(cands, req(1280, 720, 30)).expect("non-empty");
        assert_eq!(chosen.fps, 30);
    }

    #[test]
    fn select_best_tiebreaks_format_prefers_yuv_over_rgb() {
        let cands = vec![
            mode(1280, 720, PixelFormat::Rgb, 30),
            mode(1280, 720, PixelFormat::Yuy2, 30),
        ];
        let chosen = select_best(cands, req(1280, 720, 30)).expect("non-empty");
        assert_eq!(chosen.encoding, CaptureEncoding::Raw(PixelFormat::Yuy2));
    }

    #[test]
    fn select_best_falls_back_to_highest_fps_when_all_low() {
        // All modes below the floor (25): only the fastest ones remain,
        // and among them the size policy still applies.
        let cands = vec![
            mode(640, 480, PixelFormat::Yuy2, 5),
            mode(640, 480, PixelFormat::Yuy2, 10),
            mode(320, 240, PixelFormat::Yuy2, 10),
        ];
        let chosen = select_best(cands, req(640, 480, 30)).expect("non-empty");
        assert_eq!(chosen.fps, 10);
        assert_eq!(size(chosen), (640, 480));
    }

    #[test]
    fn select_best_returns_none_on_empty() {
        assert_eq!(select_best(Vec::new(), req(640, 480, 30)), None);
    }

    #[test]
    fn source_caps_pin_mode_with_fps_range() {
        crate::init().expect("gstreamer init");
        let caps = mode(800, 448, PixelFormat::Yuy2, 30)
            .source_caps()
            .expect("caps");
        let s = caps.structure(0).expect("one structure");
        assert_eq!(s.name(), "video/x-raw");
        assert_eq!(s.get::<&str>("format").expect("format"), "YUY2");
        assert_eq!(s.get::<i32>("width").expect("width"), 800);
        assert_eq!(s.get::<i32>("height").expect("height"), 448);
        // A driver interval of 333333/10000000 s must fall in the range.
        let exact = gstreamer::Caps::builder("video/x-raw")
            .field("framerate", gstreamer::Fraction::new(10_000_000, 333_333))
            .build();
        let range_only = gstreamer::Caps::builder("video/x-raw")
            .field(
                "framerate",
                s.get::<gstreamer::FractionRange>("framerate")
                    .expect("range"),
            )
            .build();
        assert!(exact.can_intersect(&range_only));
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
