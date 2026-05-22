# FluxFrame

Realtime video processing layer for Linux. Reads a video stream from a camera (or a test source), runs it through a configurable effect chain, and publishes the result as a virtual camera (`v4l2loopback`) so that Zoom, Meet, OBS, browsers and similar applications can consume it.

The first production effect is `background_blur`. The architecture is deliberately effect-agnostic — additional effects (background replacement, auto-cropping, overlays, …) plug into the same chain.

## Status

**Stage 0 — scaffolding only.** The workspace, type/trait surface and CLI structure are in place; the pipeline itself does not yet capture or process frames. Subcommands print a `not implemented yet (Stage N)` notice. See `doc/plan/000-overview.md` for the full roadmap.

## Build

This project targets Linux. The build needs `rustc`/`cargo`, `pkg-config`, GStreamer + plugins-base + plugins-good, GLib and (from Stage 3 onward) `onnxruntime`.

### Option A — Guix dev shell (no global install)

```bash
guix shell -m manifest.scm
cargo build --workspace
cargo test --workspace
cargo run -p fluxframe-cli -- --help
```

### Option B — install once into your Guix profile

```bash
guix install gstreamer gst-plugins-base gst-plugins-good glib
```

If you already use `guix home`, packages installed via `guix install` land in `~/.guix-profile`, which is **not** part of the default `PKG_CONFIG_PATH` set up by `guix home`. Extend it once in your shell init:

```bash
export PKG_CONFIG_PATH="$HOME/.guix-profile/lib/pkgconfig:$PKG_CONFIG_PATH"
```

After that, plain `cargo build --workspace` works without `guix shell`.

## Virtual camera (one-time host setup)

`v4l2loopback` is a kernel module and is **not** managed by `manifest.scm`. Install it via your distribution and load it before running FluxFrame:

```bash
sudo modprobe v4l2loopback \
  devices=1 \
  video_nr=10 \
  card_label="FluxFrame Camera" \
  exclusive_caps=1
```

Verify:

```bash
v4l2-ctl --list-devices
```

## Pre-commit

Install once:

```bash
pip install --user pre-commit
pre-commit install
```

Hooks run `cargo fmt --check`, `cargo clippy -D warnings` and `cargo test`. They assume you are inside the Guix dev shell when committing.

## Roadmap

| Stage | Status |
|---|---|
| 0. Core scaffolding | done |
| 1. Passthrough on testsrc | planned |
| 2. Real V4L2 I/O (`list`, `check`, v4l2loopback) | planned |
| 3. Inference layer (`InferenceEngine`, ONNX) | planned |
| 4. `background_blur` effect | planned |
| 5. Realtime hardening (latency, drop, fallback) | planned |
| 6. Documentation + release candidate | planned |

## License

Dual-licensed under MIT or Apache-2.0 (your choice).
