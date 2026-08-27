//! V4L2 device enumeration and inspection.
//!
//! Stage 2 implementation reads `/sys/class/video4linux/video*` directly
//! — no `v4l2-ctl` or `ioctl` dependency.  Limitations: capture-vs-
//! output classification and "virtual" detection are heuristic.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Functional classification of a V4L2 device entry.
///
/// `#[non_exhaustive]` so a future variant (e.g. dedicated `Output` once a
/// kernel-level signal becomes reliable) is not a breaking change.  The
/// previous `Output` variant was removed in Stage 2 because [`classify`]
/// never returned it — the heuristic only ever produces `Input`/`Virtual`/
/// `Unknown`, so the dead variant only invited mis-handling in `match`
/// sites elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum V4l2DeviceKind {
    /// Capture-only (e.g. UVC webcam).
    Input,
    /// Bidirectional or loopback (v4l2loopback default).
    Virtual,
    /// Heuristic could not determine — caller should treat as unknown.
    Unknown,
}

/// A single device entry produced by [`enumerate_devices`].
#[derive(Debug, Clone)]
pub struct V4l2Device {
    /// Path under `/dev` (e.g. `/dev/video0`).
    pub path: PathBuf,
    /// Driver-reported name (from `/sys/class/video4linux/<name>/name`).
    pub name: String,
    /// Heuristic classification.
    pub kind: V4l2DeviceKind,
}

/// Outcome of an enumeration attempt — keeps the three "empty" cases
/// distinct so the CLI can render a precise hint rather than the catch-all
/// "check that the kernel exposes the V4L2 sysfs tree" message.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum EnumerationStatus {
    /// `/sys/class/video4linux` does not exist — kernel does not expose
    /// V4L2 (rare on a desktop kernel), or sysfs itself is not mounted
    /// (containers without `/sys` propagation).
    SysfsAbsent,
    /// `/sys/class/video4linux` exists but `read_dir` failed (typically
    /// EACCES from MAC/sandbox policies).  The `io::ErrorKind` is kept so
    /// callers can branch on permission vs other failures.
    SysfsUnreadable(io::ErrorKind),
    /// `/sys/class/video4linux` is readable but contains no `video*`
    /// entries.  This is the "no cameras attached and no v4l2loopback
    /// loaded" case — qualitatively different from `SysfsAbsent`.
    NoDevices,
    /// One or more devices were enumerated.
    Found(Vec<V4l2Device>),
}

/// Enumerate the V4L2 devices visible to the system.
///
/// Reads `/sys/class/video4linux` and joins basenames against `/dev`.
/// Errors during per-device reads (e.g. permission denied on a single
/// entry) are logged via `tracing::trace` and the affected entry is
/// skipped — the function returns whatever it could read successfully.
///
/// Returns an empty vector when `/sys/class/video4linux` is absent OR
/// readable-but-empty.  Use [`enumerate_devices_status`] if you need to
/// distinguish those two cases (the CLI does, to render different hints).
#[must_use]
pub fn enumerate_devices() -> Vec<V4l2Device> {
    match enumerate_devices_status() {
        EnumerationStatus::Found(v) => v,
        _ => Vec::new(),
    }
}

/// Same as [`enumerate_devices`] but reads from caller-supplied roots.
///
/// See [`enumerate_devices_status_in`] for the rationale on having an
/// `_in` variant.
#[must_use]
pub fn enumerate_devices_in(sys_root: &Path, dev_root: &Path) -> Vec<V4l2Device> {
    match enumerate_devices_status_in(sys_root, dev_root) {
        EnumerationStatus::Found(v) => v,
        _ => Vec::new(),
    }
}

/// Like [`enumerate_devices`] but returns an [`EnumerationStatus`] so the
/// caller can tell the three "empty" cases apart.
#[must_use]
pub fn enumerate_devices_status() -> EnumerationStatus {
    enumerate_devices_status_in(Path::new("/sys/class/video4linux"), Path::new("/dev"))
}

/// Testable form of [`enumerate_devices_status`] — `sys_root` is the
/// directory whose subdirectories name video device entries (production:
/// `/sys/class/video4linux`); `dev_root` is the directory that holds the
/// matching character-device nodes (production: `/dev`).  Both are exposed
/// so unit tests can drive a synthetic layout without bind-mounting.
#[must_use]
pub fn enumerate_devices_status_in(sys_root: &Path, dev_root: &Path) -> EnumerationStatus {
    let entries = match fs::read_dir(sys_root) {
        Ok(it) => it,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            tracing::debug!(?sys_root, "sysfs root absent");
            return EnumerationStatus::SysfsAbsent;
        }
        Err(e) => {
            tracing::debug!(?sys_root, error = %e, "sysfs root unreadable");
            return EnumerationStatus::SysfsUnreadable(e.kind());
        }
    };

    let mut out: Vec<V4l2Device> = entries
        .filter_map(std::result::Result::ok)
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("video"))
        })
        .filter_map(|e| inspect_entry(&e.path(), dev_root))
        .collect();
    if out.is_empty() {
        return EnumerationStatus::NoDevices;
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    EnumerationStatus::Found(out)
}

/// What v4l2loopback's `state` attribute says about the **producer**.
///
/// The attribute's wording is the driver's, and it is the opposite of
/// what it looks like: `attr_show_state` prints `"capture"` when the
/// OUTPUT stream token is taken (somebody is writing frames, so the node
/// can be captured from) and `"output"` when it is free (nobody is
/// writing). It says nothing about capture-side clients — assuming
/// otherwise is what made the Stage 15 presence detector pin itself to
/// "consumer present" forever, and that mistake must not be repeated
/// here.
///
/// `#[non_exhaustive]` for the same reason as [`EnumerationStatus`]: the
/// "could not tell" cases are load-bearing and may need to grow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LoopbackState {
    /// The producer holds the OUTPUT stream (attribute reads `capture`).
    ///
    /// Also what a node loaded with `keep_format=1` reports permanently,
    /// which is one reason idle mode is unsupported in that mode.
    ProducerStreaming,
    /// Nobody is writing to the node (attribute reads `output`).
    ///
    /// Every capture client's `VIDIOC_STREAMON` fails with `EIO` while
    /// this holds, so for a daemon that believes it is publishing frames
    /// this is an internal fault, not an observation about consumers.
    ProducerStopped,
    /// The attribute could not be read: no such node, not a
    /// v4l2loopback device, a device path that is not a `videoN` name,
    /// or a permission failure. Never an assertion about the stream.
    Unreadable(io::ErrorKind),
    /// The attribute was read but held something this driver version
    /// does not define.
    Unparseable,
}

/// Read the loopback `state` attribute for `device` (e.g. `/dev/video10`).
///
/// See [`LoopbackState`] for the driver's inverted wording. Returns an
/// outcome rather than a `Result` because "unreadable" is a routine,
/// non-fatal answer here — the sink may not be a loopback node at all.
#[must_use]
pub fn read_loopback_state(device: &Path) -> LoopbackState {
    read_loopback_state_in(Path::new("/sys/class/video4linux"), device)
}

/// Testable form of [`read_loopback_state`] — `sys_root` is the
/// directory holding per-device attribute directories (production:
/// `/sys/class/video4linux`).
///
/// `device` contributes only its file name, and only if that name looks
/// like `videoN`: the path comes from operator-supplied config, and
/// joining it unchecked would turn this into an arbitrary-file reader
/// under `/sys` for a path like `/dev/../class/net/eth0/address`.
#[must_use]
pub fn read_loopback_state_in(sys_root: &Path, device: &Path) -> LoopbackState {
    let Some(name) = device.file_name().and_then(|n| n.to_str()) else {
        return LoopbackState::Unreadable(io::ErrorKind::InvalidInput);
    };
    if !is_video_node_name(name) {
        return LoopbackState::Unreadable(io::ErrorKind::InvalidInput);
    }
    let attr = sys_root.join(name).join("state");
    match fs::read_to_string(&attr) {
        Ok(s) => match s.trim() {
            "capture" => LoopbackState::ProducerStreaming,
            "output" => LoopbackState::ProducerStopped,
            other => {
                tracing::debug!(
                    path = %attr.display(),
                    value = other,
                    "unrecognised v4l2loopback state attribute"
                );
                LoopbackState::Unparseable
            }
        },
        Err(e) => {
            tracing::trace!(path = %attr.display(), error = %e, "loopback state attribute unreadable");
            LoopbackState::Unreadable(e.kind())
        }
    }
}

/// `videoN` with at least one digit and nothing else — the shape of a
/// V4L2 node basename.
fn is_video_node_name(name: &str) -> bool {
    match name.strip_prefix("video") {
        Some(digits) => !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()),
        None => false,
    }
}

fn inspect_entry(entry: &Path, dev_root: &Path) -> Option<V4l2Device> {
    let basename = entry.file_name()?.to_str()?.to_string();
    let path = dev_root.join(&basename);
    let name_path = entry.join("name");
    // Per-entry read failures are noise at debug level on systems with
    // many V4L2 devices; surface only at `trace` for deep diagnostics.
    let name = match fs::read_to_string(&name_path) {
        Ok(s) => s.trim().to_string(),
        Err(e) => {
            tracing::trace!(path = %name_path.display(), error = %e, "name read failed; falling back to basename");
            basename.clone()
        }
    };
    let modalias_path = entry.join("device/modalias");
    let modalias = match fs::read_to_string(&modalias_path) {
        Ok(s) => Some(s.trim().to_string()),
        Err(e) => {
            tracing::trace!(path = %modalias_path.display(), error = %e, "modalias read failed; classification falls back to name");
            None
        }
    };

    let kind = classify(&name, modalias.as_deref());
    Some(V4l2Device { path, name, kind })
}

fn classify(name: &str, modalias: Option<&str>) -> V4l2DeviceKind {
    let name_lc = name.to_ascii_lowercase();
    if let Some(m) = modalias {
        if m.contains("v4l2loopback") {
            return V4l2DeviceKind::Virtual;
        }
    }
    if name_lc.contains("loopback") || name_lc.contains("fluxframe") {
        return V4l2DeviceKind::Virtual;
    }
    if name_lc.contains("camera") || name_lc.contains("webcam") {
        return V4l2DeviceKind::Input;
    }
    V4l2DeviceKind::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;

    /// RAII guard removing a temporary directory on drop.
    struct TempDirGuard(PathBuf);

    impl TempDirGuard {
        fn new(label: &str) -> Self {
            let nano = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            // Add thread id to avoid collisions when tests run in parallel.
            let tid = std::thread::current().id();
            let p = std::env::temp_dir().join(format!("ff-v4l2-test-{label}-{nano}-{tid:?}"));
            let _ = fs::remove_dir_all(&p);
            fs::create_dir_all(&p).unwrap();
            Self(p)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut f = File::create(path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    fn make_fake_sys(root: &Path) {
        write(&root.join("video0/name"), "USB 2.0 Camera\n");
        // No modalias for video0 — webcam without virtual marker.
        write(&root.join("video1/name"), "USB 2.0 Camera Metadata\n");
        write(&root.join("video10/name"), "FluxFrame Camera\n");
        write(
            &root.join("video10/device/modalias"),
            "platform:v4l2loopback\n",
        );
    }

    #[test]
    fn loopback_state_maps_the_drivers_two_words() {
        let sys = TempDirGuard::new("sys-state");
        write(&sys.path().join("video10/state"), "capture\n");
        write(&sys.path().join("video11/state"), "output\n");
        assert_eq!(
            read_loopback_state_in(sys.path(), Path::new("/dev/video10")),
            LoopbackState::ProducerStreaming
        );
        assert_eq!(
            read_loopback_state_in(sys.path(), Path::new("/dev/video11")),
            LoopbackState::ProducerStopped
        );
    }

    #[test]
    fn loopback_state_reports_unparseable_content_separately() {
        let sys = TempDirGuard::new("sys-state-junk");
        write(&sys.path().join("video10/state"), "banana\n");
        assert_eq!(
            read_loopback_state_in(sys.path(), Path::new("/dev/video10")),
            LoopbackState::Unparseable
        );
    }

    #[test]
    fn loopback_state_missing_attribute_is_unreadable_not_stopped() {
        // The distinction matters: `ProducerStopped` is an internal
        // fault worth a warning, "no such file" is a non-loopback sink.
        let sys = TempDirGuard::new("sys-state-missing");
        assert_eq!(
            read_loopback_state_in(sys.path(), Path::new("/dev/video10")),
            LoopbackState::Unreadable(io::ErrorKind::NotFound)
        );
    }

    #[test]
    fn loopback_state_rejects_paths_that_are_not_video_nodes() {
        let sys = TempDirGuard::new("sys-state-traversal");
        // Would resolve to `<sys>/../etc/passwd` if the name were joined
        // unchecked; also covers plain non-`videoN` sinks.
        write(&sys.path().join("video10/state"), "capture\n");
        for path in [
            Path::new("/dev/../etc/passwd"),
            Path::new("/dev/videoX"),
            Path::new("/dev/video"),
            Path::new("/dev/"),
        ] {
            assert_eq!(
                read_loopback_state_in(sys.path(), path),
                LoopbackState::Unreadable(io::ErrorKind::InvalidInput),
                "{}",
                path.display()
            );
        }
    }

    #[test]
    fn enumerate_returns_sorted_entries() {
        let sys = TempDirGuard::new("sys-sorted");
        let dev = TempDirGuard::new("dev-sorted");
        make_fake_sys(sys.path());
        let devices = enumerate_devices_in(sys.path(), dev.path());
        let paths: Vec<_> = devices.iter().map(|d| d.path.clone()).collect();
        assert_eq!(
            paths,
            vec![
                dev.path().join("video0"),
                dev.path().join("video1"),
                dev.path().join("video10"),
            ]
        );
    }

    #[test]
    fn classify_camera_by_name() {
        let sys = TempDirGuard::new("sys-cam");
        let dev = TempDirGuard::new("dev-cam");
        make_fake_sys(sys.path());
        let devices = enumerate_devices_in(sys.path(), dev.path());
        let video0 = devices.iter().find(|d| d.path.ends_with("video0")).unwrap();
        assert_eq!(video0.kind, V4l2DeviceKind::Input);
        assert_eq!(video0.name, "USB 2.0 Camera");
        assert_eq!(video0.path, dev.path().join("video0"));
    }

    #[test]
    fn classify_loopback_by_modalias() {
        let sys = TempDirGuard::new("sys-loop");
        let dev = TempDirGuard::new("dev-loop");
        make_fake_sys(sys.path());
        let devices = enumerate_devices_in(sys.path(), dev.path());
        let video10 = devices
            .iter()
            .find(|d| d.path.ends_with("video10"))
            .unwrap();
        assert_eq!(video10.kind, V4l2DeviceKind::Virtual);
    }

    #[test]
    fn missing_root_returns_empty() {
        let dev = TempDirGuard::new("dev-missing");
        let path = Path::new("/nonexistent/sysfs/path");
        assert!(enumerate_devices_in(path, dev.path()).is_empty());
    }

    #[test]
    fn status_reports_sysfs_absent() {
        let dev = TempDirGuard::new("dev-status-absent");
        let path = Path::new("/nonexistent/sysfs/path");
        assert!(matches!(
            enumerate_devices_status_in(path, dev.path()),
            EnumerationStatus::SysfsAbsent
        ));
    }

    #[test]
    fn status_reports_no_devices_for_empty_dir() {
        let sys = TempDirGuard::new("sys-empty");
        let dev = TempDirGuard::new("dev-empty");
        // sys directory exists (TempDirGuard::new created it) but is empty.
        assert!(matches!(
            enumerate_devices_status_in(sys.path(), dev.path()),
            EnumerationStatus::NoDevices
        ));
    }

    #[test]
    fn status_reports_found_for_populated_dir() {
        let sys = TempDirGuard::new("sys-populated");
        let dev = TempDirGuard::new("dev-populated");
        make_fake_sys(sys.path());
        let found = match enumerate_devices_status_in(sys.path(), dev.path()) {
            EnumerationStatus::Found(v) => v,
            other => panic!("expected Found, got {other:?}"),
        };
        assert_eq!(found.len(), 3);
    }
}
