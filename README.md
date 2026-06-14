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
  `pixelate`, `passthrough`, `sharpen`, `vignette`, `exposure_correct`,
  `image_fill`).
* `[presets.NAME.foreground]` — same registry as `background`,
  applied to the foreground half. Typical foreground use cases are
  `sharpen` (crisper face on cheap webcams) and `exposure_correct`
  (lift dark faces in backlit scenes).
* `[presets.NAME.post]` — mask-aware frame-level chain run after the
  alpha-composite step. Effects here see the already-blended frame
  plus a read-only view of the upscaled mask. Available post-effects:
  `passthrough`, `auto_frame` (smart-crop + recenter around the
  detected person). Requires `[mask]` to be set.

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

A "work-call" preset combining the typical fixes for cheap webcams —
exposure correction + light sharpening on the speaker, branded
background photo with a soft vignette behind:

```toml
[presets.work.mask]
model = "./models/selfie_segmentation.onnx"
chain = ["smooth_temporal", "threshold", "feather"]

[presets.work.background]
chain = ["image_fill", "vignette"]
[presets.work.background.image_fill]
path = "./assets/office-bg.jpg"
fit = "cover"
[presets.work.background.vignette]
strength = 0.3

[presets.work.foreground]
chain = ["exposure_correct", "sharpen"]
[presets.work.foreground.exposure_correct]
target_brightness = 0.55
[presets.work.foreground.sharpen]
amount = 0.3
```

```bash
# Run with the blur preset:
cargo run --release -- run --config fluxframe.toml --preset blur

# Default preset (implicit):
cargo run --release -- run --config fluxframe.toml
```

A preset using the post chain for auto-framing on top of the same
stack:

```toml
[presets.framed.mask]
model = "./models/selfie_segmentation.onnx"
chain = ["smooth_temporal", "threshold", "feather"]

[presets.framed.background]
chain = ["blur"]
[presets.framed.background.blur]
radius = 20

[presets.framed.post]
chain = ["auto_frame"]
[presets.framed.post.auto_frame]
threshold = 0.5
padding = 0.15
smoothing = 0.85    # EMA inertia — higher = less jitter, slower response
zoom_max = 1.6      # 1.0 disables, 1.6 = mild zoom for office scenes
```

Note: `fluxframe benchmark --model PATH` runs inference-only against the
supplied model and does not consult presets.

## Live reconfiguration via control socket (Stage 13)

A running `fluxframe run` daemon can expose a UNIX socket for
line-delimited JSON commands. Enable it in `fluxframe.toml`:

```toml
[control]
enabled = true
# socket_path = "/run/user/1000/fluxframe.sock"  # optional override
```

The default path is `$XDG_RUNTIME_DIR/fluxframe.sock` (mode `0600`).
Talk to the daemon via `socat`, `nc -U`, or any UNIX-socket client:

```bash
echo '{"cmd":"list_presets"}'                | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
echo '{"cmd":"current_preset"}'              | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
echo '{"cmd":"set_preset","name":"blur"}'    | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
echo '{"cmd":"set","path":"background.blur.radius","value":40}' \
     | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
echo '{"cmd":"set_chain","section":"background","chain":["blur","vignette"]}' \
     | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
echo '{"cmd":"get_config","path":"background.blur"}' \
     | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
echo '{"cmd":"reload"}'                      | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
```

Every response is one line of JSON shaped as
`{"ok":"true","data":...}` or `{"ok":"false","error":"…","hint":"…"}`.

Capabilities:

* **`set_preset`** — switch among presets defined in the TOML. Full
  composite rebuild; may take 100–500 ms when the segmentation model
  changes.
* **`set <path> <value>`** — tweak one field of one effect, where
  `<path>` is `<section>.<effect>.<field>` (e.g.
  `background.blur.radius`). Sub-millisecond — perfect for
  slider-style tuning.
* **`set_chain <section> [names...]`** — add or remove effects from a
  sub-chain (`mask`, `background`, `foreground`, `post`).
* **`reload`** — re-read the TOML file from disk and rebuild the
  active preset. CLI overrides given at startup stay in effect.
* **`get_config [path]`** — introspect the live preset, optionally at
  a dot-path subtree.

Constraints: no auth beyond fs perms; the `[input]`/`[output]`
sections cannot be reconfigured live (would require a GStreamer
pipeline restart). Runtime tweaks are ephemeral — they are NOT
written back to `fluxframe.toml`.

## Live tuning via fluxframe-gui (Stage 14)

A GTK4/libadwaita companion app provides a slider-style editor over
the same control socket. Start the daemon with `[control].enabled =
true` (above), then launch the GUI:

```bash
guix shell -m manifest.scm -- cargo run -p fluxframe-gui
# or with a non-default socket path:
guix shell -m manifest.scm -- cargo run -p fluxframe-gui -- \
    --socket /run/user/1000/fluxframe.sock
```

What you get:

* **Preset switcher** — a DropDown in the header bar lists every
  preset defined in the loaded TOML; selecting one dispatches
  `set_preset` and the chain editor refreshes.
* **Chain editor** — one page with four groups (mask /
  background / foreground / post). Each effect is an expandable row
  with `↑` / `↓` / `✕` suffix buttons; clicking the `+` MenuButton on
  a group header lists every effect the daemon's registry advertises
  and adds the chosen one to the chain.
* **Per-parameter widgets** — sliders, spin buttons, switches,
  colour pickers, file pickers, drop-downs (chosen automatically from
  the effect's `ParamKind` metadata). Changes are debounced (50 ms
  for continuous Float / Integer params, 100–200 ms for heavier ones)
  before hitting the daemon.
* **Rollback toasts** — if the daemon rejects a `set` (e.g. out of
  range), an `AdwToast` shows the error and the widget snaps back to
  the last-good value.
* **Open preview** — a button in the header bar spawns
  `gst-launch-1.0 v4l2src device=/dev/video10 ! videoconvert !
  autovideosink` so you can watch the daemon's output while tuning.
* **Keyboard shortcuts**:
  * `Ctrl+R` / `F5` — `reload` daemon TOML + refetch state.
  * `Ctrl+Q` — quit.
  * `Ctrl+1` … `Ctrl+9` — switch to preset slot N (1-indexed in the
    preset list).
* **Window geometry** is persisted to
  `$XDG_CONFIG_HOME/fluxframe/gui.json` (or `~/.config/fluxframe/`)
  between runs.

The GUI requires `gtk` (≥ 4.18, Guix gives 4.20), `libadwaita` (≥ 1.6,
Guix gives 1.8), `graphene`, `pango`, `cairo`, and `gdk-pixbuf` — all
already listed in `manifest.scm`.

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
| 11. Plane-effects suite (sharpen, vignette, exposure_correct, image_fill) | done |
| 12. Post-composite mask-aware chain + `auto_frame` | done |
| 13. Live reconfiguration via UNIX control socket | done |

## License

Dual-licensed under MIT or Apache-2.0 (your choice).
