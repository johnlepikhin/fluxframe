//! `fluxframe list` — enumerate input/output V4L2 devices.
//!
//! Stage 2 implementation: reads `/sys/class/video4linux/` via
//! [`fluxframe_gst::enumerate_devices`].  Capture/output/virtual
//! classification is heuristic — see [`fluxframe_gst::V4l2DeviceKind`].

use fluxframe_core::FluxError;
use fluxframe_gst::{EnumerationStatus, V4l2Device, V4l2DeviceKind, enumerate_devices_status};
use tracing::info;

use crate::cli::ListArgs;

/// Entry point for `fluxframe list`.
///
/// # Errors
///
/// Currently returns `Ok(())` — enumeration is best-effort and prints
/// whatever it managed to read from `/sys/class/video4linux/`.
#[expect(
    clippy::unnecessary_wraps,
    reason = "signature mirrors other command handlers; Stage 5 may add real failures"
)]
pub fn run(_args: ListArgs) -> Result<(), FluxError> {
    let devices = match enumerate_devices_status() {
        EnumerationStatus::SysfsAbsent => {
            println!("No V4L2 devices: /sys/class/video4linux is not present.");
            println!("If this is unexpected, check that:");
            println!("  - the kernel exposes V4L2 (CONFIG_VIDEO_DEV)");
            println!("  - sysfs is mounted (containers may need an explicit /sys mount)");
            return Ok(());
        }
        EnumerationStatus::SysfsUnreadable(kind) => {
            println!("No V4L2 devices: /sys/class/video4linux is not readable ({kind:?}).");
            println!("If this is unexpected, check that:");
            println!("  - your user can read /sys/class/video4linux entries");
            println!("  - LSM/sandbox policy (AppArmor, SELinux, seccomp) is not blocking it");
            return Ok(());
        }
        EnumerationStatus::NoDevices => {
            println!("No V4L2 devices: /sys/class/video4linux is empty.");
            println!("No cameras are attached and no v4l2loopback is loaded. To get devices:");
            println!("  - plug in a USB camera");
            println!(
                "  - load v4l2loopback for a virtual sink: `sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=\"FluxFrame Camera\" exclusive_caps=1`"
            );
            return Ok(());
        }
        EnumerationStatus::Found(v) => v,
        // `EnumerationStatus` is `#[non_exhaustive]`; future variants
        // surface here with a generic message until `list` learns them.
        _ => {
            println!("No V4L2 devices: enumeration returned an unsupported status.");
            return Ok(());
        }
    };

    let (inputs, outputs) = partition(&devices);

    println!("Input devices:");
    if inputs.is_empty() {
        println!("  (none detected — webcams should show up here)");
    } else {
        for d in &inputs {
            println!("  {}", format_row(d));
        }
    }
    println!();
    println!("Output candidates:");
    if outputs.is_empty() {
        println!(
            "  (none — load v4l2loopback to create one, e.g.\n   `sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=\"FluxFrame Camera\" exclusive_caps=1`)"
        );
    } else {
        for d in &outputs {
            println!("  {}", format_row(d));
        }
    }
    // `inputs.len() + outputs.len()` over-counts `Unknown` (intentionally
    // shown in both buckets); log `total = devices.len()` separately so
    // the operator sees the true device count instead of the inflated
    // sum.
    info!(
        devices = devices.len(),
        inputs = inputs.len(),
        outputs = outputs.len(),
        "device enumeration complete",
    );
    Ok(())
}

fn partition(devices: &[V4l2Device]) -> (Vec<&V4l2Device>, Vec<&V4l2Device>) {
    // Stage 2 update: `V4l2DeviceKind::Output` was removed (the heuristic
    // never produced it).  `V4l2DeviceKind` is also `#[non_exhaustive]`,
    // so we need a `_` arm to stay forward-compatible — future variants
    // surface in both buckets so the user at least sees the device.
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for d in devices {
        match d.kind {
            V4l2DeviceKind::Input => inputs.push(d),
            V4l2DeviceKind::Virtual => outputs.push(d),
            V4l2DeviceKind::Unknown => {
                // Show in both — caller can decide.
                inputs.push(d);
                outputs.push(d);
            }
            _ => {
                // Future variants — show in both buckets as a safe default.
                inputs.push(d);
                outputs.push(d);
            }
        }
    }
    (inputs, outputs)
}

fn format_row(d: &V4l2Device) -> String {
    let path = d.path.display().to_string();
    let virtual_tag = if matches!(d.kind, V4l2DeviceKind::Virtual) {
        "  virtual=true"
    } else {
        ""
    };
    format!("{path:<14} {:<28} backend=v4l2{virtual_tag}", d.name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn device(path: &str, name: &str, kind: V4l2DeviceKind) -> V4l2Device {
        V4l2Device {
            path: PathBuf::from(path),
            name: name.into(),
            kind,
        }
    }

    #[test]
    fn partition_sorts_inputs_and_outputs() {
        let devs = vec![
            device("/dev/video0", "USB Camera", V4l2DeviceKind::Input),
            device("/dev/video10", "FluxFrame Camera", V4l2DeviceKind::Virtual),
        ];
        let (i, o) = partition(&devs);
        assert_eq!(i.len(), 1);
        assert_eq!(o.len(), 1);
        assert_eq!(i[0].path, PathBuf::from("/dev/video0"));
        assert_eq!(o[0].path, PathBuf::from("/dev/video10"));
    }

    #[test]
    fn partition_routes_unknown_to_both() {
        let devs = vec![device("/dev/video5", "??", V4l2DeviceKind::Unknown)];
        let (i, o) = partition(&devs);
        assert_eq!(i.len(), 1);
        assert_eq!(o.len(), 1);
    }

    #[test]
    fn format_row_marks_virtual() {
        let d = device("/dev/video10", "FluxFrame Cam", V4l2DeviceKind::Virtual);
        let row = format_row(&d);
        assert!(row.contains("virtual=true"));
        assert!(row.contains("/dev/video10"));
        assert!(row.contains("FluxFrame Cam"));
    }
}
