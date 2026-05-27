# FluxFrame

Realtime video processing layer for Linux. Reads a video stream from a camera (or a test source), runs it through a configurable effect chain, and publishes the result as a virtual camera (`v4l2loopback`) so that Zoom, Meet, OBS, browsers and similar applications can consume it.

The first production effect is `background_blur`. The architecture is deliberately effect-agnostic — additional effects (background replacement, auto-cropping, overlays, …) plug into the same chain.

## Status

**Stage 10 — named-preset config in place.** Pipeline runs synthetic
video (testsrc) or V4L2 capture → segmented composite (mask + background +
foreground sub-pipelines) → fakesink, v4l2loopback or autovideosink.
`fluxframe list` / `fluxframe check` verify devices and the selected
preset's pipeline. ONNX Runtime is wired through
`fluxframe-effects::ml::OnnxEngine`; `fluxframe benchmark --model <path>`
exercises the inference layer in isolation. Composite effects (mask
post-processing + per-plane filters such as `blur`, `color_fill`,
`pixelate`) are configured per preset.

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

## ONNX Runtime setup (Stage 3+)

`fluxframe-effects::ml::OnnxEngine` loads ONNX Runtime dynamically via
`ort`'s `load-dynamic` feature, so the path to `libonnxruntime.so` is
read from the `ORT_DYLIB_PATH` environment variable at startup.

If you installed `onnxruntime` through Guix (see "Build" above), point
the variable at the store path:

```bash
export ORT_DYLIB_PATH=$(find ~/.guix-profile/lib /run/current-system/profile/lib \
    -maxdepth 1 -name 'libonnxruntime.so*' 2>/dev/null | head -1)
```

For other distributions, install `libonnxruntime` (apt: `libonnxruntime-dev`,
homebrew: `onnxruntime`) and point the variable at the resulting
`libonnxruntime.so` (or `.dylib` on macOS).

Once set, `fluxframe check --config fluxframe.toml` reports model
load status (the model path comes from the selected preset's
`[presets.NAME.mask] model = "..."` field), and
`fluxframe benchmark --model <path> --duration 5` runs an
inference-only latency benchmark.

## Named presets

The composite pipeline is described under `[presets.NAME]` sections in
the TOML config. Each preset bundles up to three sub-pipelines (all
optional):

* `[presets.NAME.mask]` — segmentation model plus the chain of
  mask-effects (`smooth_temporal`, `threshold`, `dilate`, `feather`,
  `largest_blob`, `invert`, `passthrough`).
* `[presets.NAME.background]` — plane-effects applied to the
  background half before alpha-composite (`blur`, `color_fill`,
  `pixelate`, `passthrough`).
* `[presets.NAME.foreground]` — same registry as `background`,
  applied to the foreground half.

Select a preset at runtime with `--preset NAME`. When the flag is
omitted, the CLI looks up `presets.default` and exits with a
structured error if it is missing.

```toml
[presets.blur.mask]
model = "./models/selfie_segmentation.onnx"
chain = ["smooth_temporal", "threshold", "feather"]

[presets.blur.mask.threshold]
level = 0.5

[presets.blur.mask.feather]
radius = 4

[presets.blur.background]
chain = ["blur"]

[presets.blur.background.blur]
radius = 20

[presets.green-screen.mask]
model = "./models/selfie_segmentation.onnx"
chain = ["threshold", "dilate"]

[presets.green-screen.background]
chain = ["color_fill"]

[presets.green-screen.background.color_fill]
rgb = [0, 255, 0]

# Preset without any sub-section = plain passthrough (no composite).
[presets.raw]
```

```bash
# Run with the blur preset:
cargo run --release -- run --config fluxframe.toml --preset blur

# Default preset (implicit):
cargo run --release -- run --config fluxframe.toml
```

Note: `fluxframe benchmark --model PATH` runs inference-only against the
supplied model and does not consult presets.

## Pre-commit

Install once:

```bash
pip install --user pre-commit
pre-commit install
```

Hooks run `cargo fmt --check`, `cargo clippy -D warnings` and `cargo test`. They assume you are inside the Guix dev shell when committing.

## GPU acceleration (Stage 7)

FluxFrame builds with a `wgpu` (Vulkan compute) blur backend behind
the default-on `wgpu` Cargo feature.  On the current pipeline
(640×480 input, `blur_downscale=4`) the GPU implementation
**loses to the CPU box-blur** — Stage 7 measurement showed
`processing_p95` +50%, fps -20% and `inference_p95` +40% (iGPU
contention with ORT).  Therefore the factory's `Auto` branch
deliberately keeps the CPU backend; the GPU path stays reachable
for debugging or future workloads through:

```bash
FLUXFRAME_FORCE_BLUR_BACKEND=wgpu fluxframe run --config fluxframe.toml ...
# or, to lock in CPU explicitly:
FLUXFRAME_FORCE_BLUR_BACKEND=cpu  fluxframe run --config fluxframe.toml ...
```

When `Wgpu` is forced and no Vulkan adapter is available, FluxFrame
exits with a structured error instead of silently demoting — see
`doc/plan/stage-7-wgpu-blur.md` "Closure" for the full finding and
the upgrade path (`Rgba16Float` intermediate, async readback,
DMA-BUF zero-copy, …).  Slim builds without GPU support compile
with `cargo build -p fluxframe-effects --no-default-features` (or
add only the `ml` feature).

## Roadmap

| Stage | Status |
|---|---|
| 0. Core scaffolding | done |
| 1. Passthrough on testsrc | done |
| 2. Real V4L2 I/O (`list`, `check`, v4l2loopback) | done |
| 3. Inference layer (`InferenceEngine`, ONNX) | done |
| 4. `background_blur` effect | done |
| 5. Realtime hardening (latency, drop, fallback) | done |
| 6. Sticky inference fallback decorator | done |
| 7. GPU blur backend seam (`wgpu`) | done |
| 8. OpenVINO inference backend (CPU/NPU) | done |
| 9. Composite pipeline (mask + bg + fg sub-chains) | done |
| 10. Named presets in config | done |

## License

Dual-licensed under MIT or Apache-2.0 (your choice).
