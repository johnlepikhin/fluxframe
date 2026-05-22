//! V4L2 device enumeration and inspection.
//!
//! Stage 2 implementation reads `/sys/class/video4linux/video*` directly
//! — no `v4l2-ctl` or `ioctl` dependency.  Limitations: capture-vs-
//! output classification and "virtual" detection are heuristic.

use std::fs;
use std::path::{Path, PathBuf};

/// Functional classification of a V4L2 device entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V4l2DeviceKind {
    /// Capture-only (e.g. UVC webcam).
    Input,
    /// Output-only sink (e.g. v4l2loopback configured as output-only).
    Output,
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

/// Enumerate the V4L2 devices visible to the system.
///
/// Reads `/sys/class/video4linux`.  Errors during reads (e.g. permission
/// denied on a single device) are logged via `tracing::debug` and the
/// affected entry is skipped — the function returns whatever it could
/// read successfully.
///
/// Returns an empty vector when `/sys/class/video4linux` is absent
/// (e.g. inside some containers, on non-Linux platforms — though gst's
/// `compile_error!` already prevents the latter).
#[must_use]
pub fn enumerate_devices() -> Vec<V4l2Device> {
    enumerate_devices_in(Path::new("/sys/class/video4linux"))
}

/// Same as [`enumerate_devices`] but reads from a caller-supplied root.
/// Useful for unit tests with a synthetic /sys layout.
#[must_use]
pub fn enumerate_devices_in(root: &Path) -> Vec<V4l2Device> {
    let Ok(entries) = fs::read_dir(root) else {
        tracing::debug!(?root, "/sys/class/video4linux not present");
        return Vec::new();
    };

    let mut out: Vec<V4l2Device> = entries
        .filter_map(std::result::Result::ok)
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("video"))
        })
        .filter_map(|e| inspect_entry(&e.path()))
        .collect();
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

fn inspect_entry(entry: &Path) -> Option<V4l2Device> {
    let basename = entry.file_name()?.to_str()?.to_string();
    let path = PathBuf::from(format!("/dev/{basename}"));
    let name = fs::read_to_string(entry.join("name"))
        .ok()
        .map_or_else(|| basename.clone(), |s| s.trim().to_string());
    let modalias = fs::read_to_string(entry.join("device/modalias"))
        .ok()
        .map(|s| s.trim().to_string());

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
        fn new() -> Self {
            let nano = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            // Add thread id to avoid collisions when tests run in parallel.
            let tid = std::thread::current().id();
            let p = std::env::temp_dir().join(format!("ff-v4l2-test-{nano}-{tid:?}"));
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
    fn enumerate_returns_sorted_entries() {
        let guard = TempDirGuard::new();
        make_fake_sys(guard.path());
        let devices = enumerate_devices_in(guard.path());
        let paths: Vec<_> = devices.iter().map(|d| d.path.clone()).collect();
        assert_eq!(
            paths,
            vec![
                PathBuf::from("/dev/video0"),
                PathBuf::from("/dev/video1"),
                PathBuf::from("/dev/video10"),
            ]
        );
    }

    #[test]
    fn classify_camera_by_name() {
        let guard = TempDirGuard::new();
        make_fake_sys(guard.path());
        let devices = enumerate_devices_in(guard.path());
        let video0 = devices.iter().find(|d| d.path.ends_with("video0")).unwrap();
        assert_eq!(video0.kind, V4l2DeviceKind::Input);
        assert_eq!(video0.name, "USB 2.0 Camera");
    }

    #[test]
    fn classify_loopback_by_modalias() {
        let guard = TempDirGuard::new();
        make_fake_sys(guard.path());
        let devices = enumerate_devices_in(guard.path());
        let video10 = devices
            .iter()
            .find(|d| d.path.ends_with("video10"))
            .unwrap();
        assert_eq!(video10.kind, V4l2DeviceKind::Virtual);
    }

    #[test]
    fn missing_root_returns_empty() {
        let path = Path::new("/nonexistent/sysfs/path");
        assert!(enumerate_devices_in(path).is_empty());
    }
}
