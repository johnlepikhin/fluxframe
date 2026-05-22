//! V4L2 device enumeration and inspection.
//!
//! Stage 2 implementation reads `/sys/class/video4linux/video*` directly
//! — no `v4l2-ctl` or `ioctl` dependency.  Limitations: capture-vs-
//! output classification and "virtual" detection are heuristic.

use std::fs;
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

/// Enumerate the V4L2 devices visible to the system.
///
/// Reads `/sys/class/video4linux` and joins basenames against `/dev`.
/// Errors during reads (e.g. permission denied on a single device) are
/// logged via `tracing::debug` and the affected entry is skipped — the
/// function returns whatever it could read successfully.
///
/// Returns an empty vector when `/sys/class/video4linux` is absent
/// (e.g. inside some containers, on non-Linux platforms — though gst's
/// `compile_error!` already prevents the latter).
#[must_use]
pub fn enumerate_devices() -> Vec<V4l2Device> {
    enumerate_devices_in(Path::new("/sys/class/video4linux"), Path::new("/dev"))
}

/// Same as [`enumerate_devices`] but reads from caller-supplied roots.
///
/// `sys_root` is the directory whose subdirectories name video device
/// entries (production: `/sys/class/video4linux`); `dev_root` is the
/// directory that holds the matching character-device nodes (production:
/// `/dev`).  Both are exposed so unit tests can drive a synthetic layout
/// without having to mount or bind-mount anything — the test passes
/// `dev_root = tmp_dev` and inspects `device.path == tmp_dev/video0`.
#[must_use]
pub fn enumerate_devices_in(sys_root: &Path, dev_root: &Path) -> Vec<V4l2Device> {
    let Ok(entries) = fs::read_dir(sys_root) else {
        tracing::debug!(?sys_root, "sysfs root not present");
        return Vec::new();
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
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

fn inspect_entry(entry: &Path, dev_root: &Path) -> Option<V4l2Device> {
    let basename = entry.file_name()?.to_str()?.to_string();
    let path = dev_root.join(&basename);
    let name_path = entry.join("name");
    let name = match fs::read_to_string(&name_path) {
        Ok(s) => s.trim().to_string(),
        Err(e) => {
            tracing::debug!(path = %name_path.display(), error = %e, "name read failed; falling back to basename");
            basename.clone()
        }
    };
    let modalias_path = entry.join("device/modalias");
    let modalias = match fs::read_to_string(&modalias_path) {
        Ok(s) => Some(s.trim().to_string()),
        Err(e) => {
            tracing::debug!(path = %modalias_path.display(), error = %e, "modalias read failed; classification falls back to name");
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
                .unwrap()
                .as_nanos();
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
        let video0 = devices
            .iter()
            .find(|d| d.path.ends_with("video0"))
            .unwrap();
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
}
