//! Internal helpers shared between the input and output pipelines.
//!
//! Some helpers (`make_element`, `build_caps`) stay `pub(crate)` so the
//! GStreamer types do not leak through the crate's public surface.  The
//! V4L2 access pre-checks ([`check_v4l2_input_access`],
//! [`check_v4l2_output_access`]) are `pub` so the CLI layer can re-use
//! the exact same path-validation logic instead of reimplementing it in
//! `check.rs`.

use std::fs::OpenOptions;
use std::io;
use std::os::unix::fs::{FileTypeExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use fluxframe_core::error::PipelineError;
use fluxframe_core::frame::PixelFormat;

use crate::frame_conv::pixel_format_to_gst;

/// Construct a GStreamer element by factory name, surfacing a structured
/// [`PipelineError::MissingElement`] when the plugin is not registered.
pub(crate) fn make_element(factory: &str, name: &str) -> Result<gstreamer::Element, PipelineError> {
    gstreamer::ElementFactory::make(factory)
        .name(name)
        .build()
        .map_err(|_| PipelineError::MissingElement {
            element: factory.into(),
            hint: format!("GStreamer plugin providing `{factory}` is not installed"),
        })
}

/// Build a `video/x-raw` caps description with the given geometry, framerate
/// and pixel format.
///
/// GStreamer requires `i32` for width/height/framerate; values that do not
/// fit are rejected with [`PipelineError::CapsNegotiationFailed`] instead of
/// being silently truncated to `i32::MAX` (which would have negotiated an
/// arbitrary smaller resolution downstream).
pub(crate) fn build_caps(
    width: u32,
    height: u32,
    fps: u32,
    format: PixelFormat,
) -> Result<gstreamer::Caps, PipelineError> {
    let width_i32 = i32::try_from(width).map_err(|_| PipelineError::CapsNegotiationFailed {
        reason: format!("value too large for GStreamer: width={width}"),
    })?;
    let height_i32 = i32::try_from(height).map_err(|_| PipelineError::CapsNegotiationFailed {
        reason: format!("value too large for GStreamer: height={height}"),
    })?;
    let fps_i32 = i32::try_from(fps).map_err(|_| PipelineError::CapsNegotiationFailed {
        reason: format!("value too large for GStreamer: fps={fps}"),
    })?;

    let gst_fmt = pixel_format_to_gst(format);
    Ok(gstreamer::Caps::builder("video/x-raw")
        .field("format", gst_fmt.to_str())
        .field("width", width_i32)
        .field("height", height_i32)
        .field("framerate", gstreamer::Fraction::new(fps_i32, 1))
        .build())
}

/// Whether the V4L2 device is being opened for capture or for output.
///
/// Used by [`validate_v4l2_device_path`] to attach the correct
/// [`PipelineError`] variant to precondition failures (path traversal,
/// missing device, wrong file type).  Both arms share the same checks —
/// the only difference is the error variant the caller wants to see.
#[derive(Debug, Clone, Copy)]
enum Access {
    /// Capture side: failures become
    /// [`PipelineError::InputDeviceUnavailable`].
    Input,
    /// Output side: failures become
    /// [`PipelineError::OutputDeviceUnavailable`].
    Output,
}

impl Access {
    /// Build the appropriate `PipelineError` variant for this access mode.
    fn err(self, device: &Path, reason: String, hint: &str) -> PipelineError {
        let device_disp = device.display().to_string();
        let hint = hint.to_string();
        match self {
            Self::Input => PipelineError::InputDeviceUnavailable {
                device: device_disp,
                reason,
                hint,
            },
            Self::Output => PipelineError::OutputDeviceUnavailable {
                device: device_disp,
                reason,
                hint,
            },
        }
    }
}

/// Resolve `device` to a canonical path and verify it is a character device.
///
/// Pre-check used by both [`check_v4l2_input_access`] and
/// [`check_v4l2_output_access`].  Rejects:
///
/// * paths that cannot be canonicalised (broken symlink, missing entry);
/// * non-character-device file types (regular file, directory, fifo).
///
/// Paths whose canonical form does not live under `/dev/` are accepted but
/// produce a `tracing::warn!` — some legitimate setups expose devices via
/// `/run/udev/...` or by-id symlinks that resolve outside `/dev/`, and the
/// `is_char_device` check still rejects regular files reached through
/// those paths.  The canonical path is **not** included in error reasons
/// (to avoid leaking host filesystem layout); it appears only in tracing
/// at `debug!` level.
///
/// Returning the canonical [`PathBuf`] is deliberate: callers (`v4l2src`,
/// `v4l2sink`) then receive the resolved path with no symlink indirection,
/// which is what we actually want to hand to the GStreamer element.
fn validate_v4l2_device_path(device: &Path, access: Access) -> Result<PathBuf, PipelineError> {
    let canon = std::fs::canonicalize(device).map_err(|e| {
        access.err(
            device,
            format!("cannot canonicalize path: {e}"),
            "verify the device path",
        )
    })?;
    if !canon.starts_with("/dev/") {
        tracing::warn!(
            canonical = %canon.display(),
            original = %device.display(),
            "device path resolves outside /dev/; continuing because target is a character device",
        );
    }
    let meta = std::fs::metadata(&canon).map_err(|e| {
        access.err(
            device,
            format!("metadata failed: {e}"),
            "check that the device exists",
        )
    })?;
    if !meta.file_type().is_char_device() {
        tracing::debug!(canonical = %canon.display(), original = %device.display(), "rejected: non-char-device target");
        return Err(access.err(
            device,
            "path resolves to non-character-device target".into(),
            "use a real /dev/video* character device",
        ));
    }
    tracing::debug!(canonical = %canon.display(), original = %device.display(), "device path validated");
    Ok(canon)
}

/// Map an `io::Error` from a v4l2 device open into a structured pipeline
/// hint.
#[must_use]
fn map_v4l2_open_error(err: &io::Error) -> &'static str {
    match err.kind() {
        io::ErrorKind::NotFound => {
            "check that the device exists; run 'fluxframe list' for the available devices"
        }
        io::ErrorKind::PermissionDenied => "add user to the 'video' group or check udev rules",
        io::ErrorKind::ResourceBusy => {
            "another application is holding the device; close it (e.g. browser tab, OBS)"
        }
        _ => "see dmesg for kernel-level diagnostics",
    }
}

/// Verify the calling process can open `device` for reading and return the
/// canonical, validated device path.
///
/// Surfaces a [`PipelineError::InputDeviceUnavailable`] with an actionable
/// hint *before* GStreamer's `v4l2src` tries to open the same path — the
/// raw GStreamer state-change error for `EACCES`/`EBUSY`/`ENOENT` is opaque
/// ("Internal data stream error" or similar), which defeats the §27
/// Error/Reason/Hint diagnostic contract without this pre-check.
///
/// The probe uses `O_NONBLOCK` so a device that would block on open
/// reports `EWOULDBLOCK`/`EAGAIN` immediately instead of hanging this
/// thread for an indefinite period.  The returned [`PathBuf`] is the
/// canonicalised path: symlinks have been resolved and the result is
/// guaranteed to be a character device.  Callers should feed this
/// canonical path to `v4l2src`/`v4l2sink` rather than the user-supplied
/// original so the pipeline targets exactly the inode the access check
/// authorised.
///
/// There is a small TOCTOU window between this check and the actual
/// `v4l2src` open; the Stage 2 plan documents that as an accepted
/// trade-off (still strictly better than no hint at all).
pub fn check_v4l2_input_access(device: &Path) -> Result<PathBuf, PipelineError> {
    let canon = validate_v4l2_device_path(device, Access::Input)?;
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&canon)
        .map_err(|e| PipelineError::InputDeviceUnavailable {
            device: device.display().to_string(),
            reason: e.to_string(),
            hint: map_v4l2_open_error(&e).into(),
        })?;
    Ok(canon)
}

/// Verify the calling process can open `device` for writing and return the
/// canonical, validated device path.
///
/// Surfaces a [`PipelineError::OutputDeviceUnavailable`] with an actionable
/// hint *before* GStreamer's `v4l2sink` tries to open the same path — the
/// raw GStreamer error for `EACCES`/`EBUSY`/`ENOENT` is opaque ("Device
/// '/dev/video10' cannot be opened for writing"), which makes Stage 2's
/// "diagnostic over silence" trade-off impossible without this pre-check.
///
/// The probe uses `O_NONBLOCK` so a device that would block on open (e.g.
/// `v4l2loopback` without a current consumer when configured with
/// `exclusive_caps`) reports `EWOULDBLOCK`/`EAGAIN` immediately instead of
/// hanging this thread for an indefinite period.  The returned [`PathBuf`]
/// is the canonical path (see [`check_v4l2_input_access`] for rationale).
///
/// There is a small TOCTOU window between this check and the actual sink
/// open; that's documented in the Stage 2 plan as an accepted trade-off.
pub fn check_v4l2_output_access(device: &Path) -> Result<PathBuf, PipelineError> {
    let canon = validate_v4l2_device_path(device, Access::Output)?;
    OpenOptions::new()
        .write(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&canon)
        .map_err(|e| PipelineError::OutputDeviceUnavailable {
            device: device.display().to_string(),
            reason: e.to_string(),
            hint: map_v4l2_open_error(&e).into(),
        })?;
    Ok(canon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn unique_tmp(label: &str) -> PathBuf {
        let nano = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tid = std::thread::current().id();
        std::env::temp_dir().join(format!("ff-util-{label}-{nano}-{tid:?}"))
    }

    #[test]
    fn check_input_access_rejects_missing() {
        let path = Path::new("/nonexistent/sysfs/video0");
        let err = check_v4l2_input_access(path).expect_err("missing path must error");
        assert!(
            matches!(err, PipelineError::InputDeviceUnavailable { .. }),
            "expected InputDeviceUnavailable, got {err:?}",
        );
    }

    #[test]
    fn check_input_access_rejects_regular_file() {
        let tmp = unique_tmp("regular");
        std::fs::File::create(&tmp)
            .unwrap()
            .write_all(b"x")
            .unwrap();
        let err = check_v4l2_input_access(&tmp).expect_err("regular file must error");
        // canonicalize ok; is_char_device fails (tmp is a regular file).
        assert!(
            matches!(err, PipelineError::InputDeviceUnavailable { .. }),
            "expected InputDeviceUnavailable, got {err:?}",
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn check_output_access_rejects_missing() {
        let path = Path::new("/nonexistent/sysfs/video10");
        let err = check_v4l2_output_access(path).expect_err("missing path must error");
        assert!(
            matches!(err, PipelineError::OutputDeviceUnavailable { .. }),
            "expected OutputDeviceUnavailable, got {err:?}",
        );
    }

    #[test]
    fn check_output_access_rejects_directory() {
        let tmp = unique_tmp("dir");
        std::fs::create_dir_all(&tmp).unwrap();
        let err = check_v4l2_output_access(&tmp).expect_err("directory must error");
        // is_char_device rejects this; maps to OutputDeviceUnavailable for
        // the output access path.
        assert!(
            matches!(err, PipelineError::OutputDeviceUnavailable { .. }),
            "expected OutputDeviceUnavailable, got {err:?}",
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn check_input_hint_actionable() {
        let path = Path::new("/nonexistent");
        let err = check_v4l2_input_access(path).expect_err("missing");
        if let PipelineError::InputDeviceUnavailable { hint, .. } = err {
            assert!(
                hint.contains("device") || hint.contains("check"),
                "hint should be actionable, got: {hint}"
            );
        } else {
            panic!("expected InputDeviceUnavailable");
        }
    }

    #[test]
    fn check_rejects_symlink_to_regular_file() {
        let nano = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tid = std::thread::current().id();
        let dir = std::env::temp_dir().join(format!("ff-symlink-test-{nano}-{tid:?}"));
        std::fs::create_dir_all(&dir).unwrap();
        let regular = dir.join("regular");
        std::fs::write(&regular, b"x").unwrap();
        let link = dir.join("link-to-regular");
        std::os::unix::fs::symlink(&regular, &link).unwrap();
        let err = check_v4l2_input_access(&link).expect_err("symlink-to-regular must fail");
        assert!(matches!(err, PipelineError::InputDeviceUnavailable { .. }));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
