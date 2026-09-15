# FluxFrame

FluxFrame is a realtime video processing daemon for Linux. It captures a camera, separates the person from the background with a segmentation model, runs configurable effect chains over the mask, the background and the foreground, and publishes the result as a virtual camera. Any application that can use a webcam (browser-based conferencing, Zoom, OBS, and so on) can use the FluxFrame camera without knowing anything about it.

A companion GTK application, `fluxframe-gui`, edits the running daemon live and saves the result back to the configuration file.

## Features

- Background blur, pixelation, solid colour or image replacement, and foreground corrections (exposure, sharpening, vignette).
- Automatic framing that follows the person, and horizontal mirroring.
- Named presets in a single TOML file, switchable at runtime.
- Live reconfiguration over a local UNIX socket, with write-back to the configuration file.
- Inference through OpenVINO (Intel NPU, GPU or CPU) with automatic fallback to ONNX Runtime on the CPU.
- Idle mode: when no application reads the virtual camera, the physical camera is released (its LED goes off) and the daemon drops to near-zero CPU.
- Automatic camera selection, hot-plug and recovery from a busy camera.
- Output to v4l2loopback, PipeWire or a local preview window.
- Built-in metrics: frame rate, per-stage latency and per-thread CPU usage.

On an Intel Core Ultra 7 155H laptop with inference on the NPU, a typical call preset uses about 30% of one CPU core at 800x448 and 25-30 fps. See [Performance](doc/user/performance.md).

## Requirements

- Linux with the `v4l2loopback` kernel module (0.13 or newer recommended).
- GStreamer 1.x with the base and good plugin sets.
- ONNX Runtime (shared library). OpenVINO is optional.
- A segmentation model: MediaPipe Selfie Segmentation in ONNX format.
- Rust 1.85 or newer to build from source.
- For the GUI: GTK 4.18+ and libadwaita 1.8+.

## Quick start

```bash
# 1. Build
cargo build --release

# 2. Create the virtual camera
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="FluxFrame Camera" exclusive_caps=1

# 3. Point FluxFrame at ONNX Runtime
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so

# 4. Install the model and an example configuration
#    (see doc/user/installation.md for the model download)
mkdir -p ~/.config/fluxframe
cp examples/fluxframe.toml ~/.config/fluxframe/fluxframe.toml

# 5. Verify devices, preset and model
/path/to/fluxframe/target/release/fluxframe check

# 6. Run
/path/to/fluxframe/target/release/fluxframe run --preset blur
```

Select "FluxFrame Camera" in the application that should see the processed video. To tune effects while the daemon is running, start `fluxframe-gui`.

## Documentation

- [Installation](doc/user/installation.md): dependencies, building, v4l2loopback, ONNX Runtime and OpenVINO, the model.
- [Running](doc/user/running.md): command-line reference, running as a user service, camera selection.
- [Configuration](doc/user/configuration.md): file location and the complete reference of global settings.
- [Presets and effects](doc/user/presets-and-effects.md): how a frame is processed, every effect and parameter, example presets.
- [Idle mode](doc/user/idle-mode.md): releasing the camera when nobody is watching.
- [GUI](doc/user/gui.md): the live editor.
- [Control socket](doc/user/control-socket.md): the JSON protocol for scripting.
- [Performance](doc/user/performance.md): reference measurements, tuning and reading the metrics.
- [Troubleshooting](doc/user/troubleshooting.md): common problems and their fixes.

A commented configuration with several ready-made presets is in [examples/fluxframe.toml](examples/fluxframe.toml).

## Development

The workspace consists of five crates: `fluxframe-core` (types, configuration, protocol), `fluxframe-gst` (GStreamer and V4L2), `fluxframe-effects` (effects and inference), `fluxframe-cli` (the `fluxframe` daemon) and `fluxframe-gui`.

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

On Guix, `guix shell -m manifest.scm` provides the complete toolchain. The repository has a `pre-commit` configuration that runs formatting, clippy and tests:

```bash
pip install --user pre-commit
pre-commit install
```

## License

Dual-licensed under MIT or Apache-2.0, at your option. The MediaPipe Selfie Segmentation model is distributed separately under Apache-2.0.
