# FluxFrame

Realtime video processing layer for Linux. Reads a video stream from a camera (or a test source), runs it through a configurable effect chain, and publishes the result as a virtual camera (`v4l2loopback`) so that Zoom, Meet, OBS, browsers and similar applications can consume it.

The first production effect is `background_blur`. The architecture is deliberately effect-agnostic — additional effects (background replacement, auto-cropping, overlays, …) plug into the same chain.

## Status

**Stage 15 + persistence — production-ready end-to-end loop.**
Pipeline runs synthetic video (testsrc) or V4L2 capture → segmented
composite (mask + background + foreground sub-pipelines) → fakesink,
v4l2loopback, autovideosink, or `pipewiresink`. `fluxframe list` /
`fluxframe check` verify devices and the selected preset's pipeline.
ONNX Runtime (via `fluxframe-effects::ml::OnnxEngine`) and a slim
OpenVINO backend are wired; `fluxframe benchmark --model <path>`
exercises inference in isolation. A UNIX control socket exposes
live reconfiguration (`set`, `set_chain`, `set_preset`, `reload`)
and now write-back persistence (`save_preset`, `save_preset_as`)
into the TOML config. A GTK4/libadwaita companion app
(`fluxframe-gui`) drives the same surface with sliders, an embedded
live preview pane, dirty-state indication and explicit Save / Save
as / Revert buttons. Idle mode drops the camera and publishes a
placeholder when no consumer is attached.

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

> **`exclusive_caps=1` обязателен для Chrome/браузеров.** Chrome и другие
> потребители, фильтрующие устройства по возможностям, показывают
> loopback в списке камер только когда модуль загружен с
> `exclusive_caps=1`. Без этого «FluxFrame Camera» просто не появляется в
> выборе устройства — самая частая причина «виртуальной камеры не видно».
> `fluxframe check` диагностирует это и подскажет команду перезагрузки
> модуля.

> **Камера занята на старте (Stage 16).** При `[idle] enabled = true`
> FluxFrame поднимает `/dev/video10` и стримит placeholder ещё до захвата
> камеры, а занятую `/dev/video0` ретраит с backoff — `/dev/video10`
> остаётся видимым в Chrome. Проверка после деплоя: занять камеру
> `cat /dev/video0 >/dev/null &`, запустить FluxFrame → `v4l2-ctl
> --list-devices` показывает «FluxFrame Camera» и Chrome её видит (серый
> placeholder); затем `kill %1` — в пределах одного backoff появляется
> живое видео. В логе строка `input device unavailable …` должна быть
> одна (без флуда).

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

## Config file location

`fluxframe run` and `fluxframe check` resolve the TOML config file in
this order (`fluxframe benchmark` does not consult any config — it
runs inference-only against `--model PATH`):

1. **`--config PATH`** — explicit override, wins over everything.
2. **`$XDG_CONFIG_HOME/fluxframe/fluxframe.toml`** (or
   `$HOME/.config/fluxframe/fluxframe.toml` when `XDG_CONFIG_HOME`
   is unset) — the default lookup path. Same path the GUI's Save
   button writes to.
3. **Built-in defaults** — when neither of the above resolves to an
   existing file, the daemon boots from compiled-in defaults
   silently.

Pass **`--no-default-config`** to skip step 2 entirely. Useful for
CI / scripted runs that must never touch the operator's home
directory. With both `--no-default-config` and no `--config`,
`save_preset` / the GUI Save button return a structured error
(there is no writable target).

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
  detected person), `mirror` (horizontal flip — counters the
  self-view mirroring conference apps apply so on-camera text reads
  right-way-round). Requires `[mask]` to be set.

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
echo '{"cmd":"config_path"}'                 | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
echo '{"cmd":"set_preset","name":"blur"}'    | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
echo '{"cmd":"set","path":"background.blur.radius","value":40}' \
     | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
echo '{"cmd":"set_chain","section":"background","chain":["blur","vignette"]}' \
     | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
echo '{"cmd":"get_config","path":"background.blur"}' \
     | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
echo '{"cmd":"save_preset"}'                 | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
echo '{"cmd":"save_preset_as","name":"experimental"}' \
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
  sub-chain (`mask`, `background`, `foreground`, `post`). Pruning is
  automatic: per-effect parameter tables for effects no longer in
  the chain are dropped so the next `save_preset` cannot leave an
  orphan `[<section>.<effect>]` block on disk.
* **`reload`** — re-read the TOML file from disk and rebuild the
  active preset. CLI overrides given at startup stay in effect.
* **`get_config [path]`** — introspect the live preset, optionally at
  a dot-path subtree.
* **`config_path`** — report the writable TOML path the daemon will
  Save into, or `null` when the daemon was started with
  `--no-default-config` and no `--config`. The GUI calls this once
  on handshake to grey out its Save button when no target is
  available.
* **`save_preset`** — persist the in-memory active preset back into
  the TOML config. Comments and unrelated tables are preserved;
  `[presets.NAME]` and any `[presets.NAME.*]` sub-tables are
  rewritten as one contiguous block. Atomic write
  (sibling `*.tmp.<pid>` + `rename`).
* **`save_preset_as { name }`** — same as `save_preset` but writes
  under a fresh preset name. Fails if `name` already exists; the
  daemon's in-memory preset map is updated so a follow-up
  `list_presets` reflects the new entry.

Constraints: no auth beyond fs perms; the `[input]`/`[output]`
sections cannot be reconfigured live (would require a GStreamer
pipeline restart). `save_preset` writes back to disk, but
`[input]`/`[output]` and other root tables remain operator-managed
and are never touched.

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
* **Save / Save as… / Revert** — a button group next to the
  switcher commits the in-memory preset back to the TOML
  (`save_preset`), forks the current state into a new named preset
  (`save_preset_as`), or discards local edits by re-pulling the
  daemon baseline (`get_config`). The window title gains a `●`
  prefix when there are unsaved edits; Save is greyed out when the
  daemon has no writable config target (e.g. started with
  `--no-default-config` and no `--config`).
* **Embedded live preview** — a `gtk::Paned` at the top of the
  window streams `/dev/video10` directly into a `gtk::Picture` so
  you can see the daemon's output while tuning, with the chain
  editor below the split. The header bar also keeps an **Open
  preview** button that spawns a detached
  `gst-launch-1.0 v4l2src device=/dev/video10 ! videoconvert !
  autovideosink` viewer in a separate window (handy when you want
  the preview to outlive the GUI).
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
  the last-good value. Toasts also surface Save / Save as / Revert
  failures (e.g. duplicate preset name on Save as).
* **Keyboard shortcuts**:
  * `Ctrl+R` / `F5` — `reload` daemon TOML + refetch state.
  * `Ctrl+Q` — quit.
  * `Ctrl+1` … `Ctrl+9` — switch to preset slot N (1-indexed in the
    preset list).
* **Window geometry + split position** is persisted to
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

## Idle mode (Stage 15)

When no application is reading from `/dev/video10`, FluxFrame can drop
the camera, stop running effects and publish a cheap placeholder frame
instead. Re-attaching a reader (Zoom, Meet, OBS, browser) resumes the
full pipeline automatically. On a laptop this saves a CPU core,
~150 MB of ONNX session RAM (deferred — see follow-ups below) and the
webcam LED.

Opt in via `[idle] enabled = true` in `fluxframe.toml`. The defaults
match the typical "step away from desk" use case:

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch. `false` runs the Stage 14 path verbatim. |
| `placeholder` | `"color"` | `"color"` or `"image"`. |
| `placeholder_rgb` | `[16, 16, 16]` | Dark grey fill for `placeholder = "color"`. |
| `placeholder_path` | — | Required when `placeholder = "image"`. PNG or JPEG, absolute path (relative-path resolution is a Stage 15 follow-up). |
| `fps` | `1` | Cosmetic placeholder frame rate. The effective idle cadence is `max(fps, min_visibility_fps)`. |
| `min_visibility_fps` | `10` | Visibility heartbeat. Keeps the loopback advertising CAPTURE caps so Chrome/WebRTC keep listing the device while idle. Independent of the camera/input fps (a static placeholder, not real frames). A bare 1 Hz placeholder is too sparse for Chrome to reliably enumerate the node. |
| `teardown_secs` | `5` | "No consumer" grace window before the camera drops. Absorbs Zoom / OBS reopen storms. |
| `deep_idle_secs` | `30` | **Deprecated / ignored** since Stage 16 (the `DeepIdle` state was removed). Still accepted in TOML for back-compat; has no effect. |
| `poll_interval_ms` | `250` | Polling-fallback walk cadence. The default path is event-driven via `inotify`, which ignores this knob; it only kicks in when `inotify` is unavailable (sandbox, watch-limit exhaustion). |

A consumer disconnect triggers this lifecycle:

```
                 t = 0 s        t = 5 s
consumer drops ─────► Cooldown ─────► Idle (terminal until a consumer reattaches)
                       (LED on)       (LED off, placeholder @ max(fps, min_visibility_fps))
```

Re-attaching fires `ResumeActive`: the supervisor pushes
one placeholder immediately to clear v4l2loopback's stale-frame
replay, then spawns a reload thread that brings the input pipeline
back to `Playing`. Steady-state cold-start budget on UVC cameras is
≤ ~300 ms; the placeholder keeps flowing for the entire warmup.

### How "no consumer" is detected

The supervisor's detector thread watches `/dev/videoN` via
`inotify` for `IN_OPEN` / `IN_CLOSE_NOWRITE` / `IN_CLOSE_WRITE`.
In steady-state idle the per-event cost is zero — the kernel only
wakes the thread when somebody opens or closes the device. The
shutdown-poll window costs one nonblocking `read(2)` returning
`EAGAIN` every ~50 ms (so the worker honours Ctrl-C within that
budget), which is sub-microsecond and orders of magnitude cheaper
than the prior `/proc` polling. On a real wake, the detector walks
`/proc/[0-9]+/fd/`, reading each symlink and checking whether any
external process (`pid != fluxframe`) holds a file descriptor
whose target matches `/dev/videoN`. One or more → `Present`. Zero →
`Absent`. Per-pid permission errors are skipped silently
(steady-state expected on a multi-user system).

The walk on event is necessary because `inotify` cannot tell us
*which* process opened the device — fluxframe itself opens and
closes the v4l2 sink during pipeline state transitions, and we
have to distinguish those self-events from external ones.

If `inotify::init` or `inotify::watches::add` fails (the device
node doesn't exist, the user is at
`/proc/sys/fs/inotify/max_user_watches`, or a sandbox blocks
`inotify_init`), the detector falls back to walking
`/proc/[0-9]+/fd/` every `idle.poll_interval_ms`. Behaviour is
identical; only the wake mechanism differs (and the steady-state
CPU cost climbs from ~0 % to ~5 % on a typical desktop).

**Limitation: root-owned consumers are invisible.** An unprivileged
fluxframe cannot read `/proc/<root-pid>/fd`, so a root-owned process
consuming `/dev/video10` looks identical to "no consumer". If you
run a root-owned consumer alongside fluxframe, idle mode may tear
down the input pipeline while a real consumer is reading. The
pragmatic workaround is to run fluxframe as the same user as the
consumer.

### Compatibility notes

- **Linux-only.** `/proc/[0-9]+/fd/` walking is a Linux-specific
  interface; the detector module is `#[cfg(target_os = "linux")]`-gated.
  On non-Linux hosts idle mode falls through to the Stage 14 path with
  a one-shot `warn!` line.
- **V4L2 loopback sinks only.** Idle mode requires
  `OutputSink::V4l2Loopback`; `fakesink`, `autovideosink` and
  `pipewiresink` fall through with a `warn!` line explaining the
  reason.
- **Webcam LED quirks.** Most UVC cameras cut the LED within ~1 s of
  the pipeline transitioning to `Null`. A few Logitech models (C920,
  C922, Brio) firmware-keep the LED lit for 1–3 s; this is
  upstream-known and not actionable from FluxFrame.

### Observability

Idle transitions are logged at `info!` with `target =
"fluxframe::idle"`. Counters surface in the teardown summary log:

- `idle_entered_total` — Active → Idle transitions.
- `deep_idle_entered_total` — retained for dashboard back-compat; always `0` since the `DeepIdle` state was removed in Stage 16.
- `idle_frames_pushed_total` — placeholder frames emitted.

Status changes (`Present ↔ Absent` from the detector) also log at
`info!` so a `RUST_LOG=info` operator sees consumer attach/detach
without enabling debug-level globally.

### Stage 15 follow-ups

- **DeepIdle removed (Stage 16).** The counter-only stub was deleted —
  it never actually unloaded ONNX and could leave the daemon wedged.
  The real RAM reclamation (~150 MB) is deferred to a separate work
  item: drop the ONNX session on a long idle and rebuild it on resume,
  needing `EffectChain` ↔ `ManagedComposite` integration
  (`SegmentationBase::take_engine` / `install_engine` accessors are
  already in place; the chain refactor is the missing piece).
- **Always-on loopback visibility (Stage 16, done).** With `idle.enabled`
  the output (loopback) producer is now built and started *before* the
  camera, so `/dev/video10` advertises CAPTURE caps and streams the
  placeholder even when `/dev/video0` is busy/absent at startup. A busy
  camera is retried with exponential backoff (`input.acquire_backoff_base_ms`
  → `…_max_ms`) instead of crash-looping. **Remaining (deferred):** when
  the camera is unplugged *mid-stream*, the output still blips briefly on
  the `run_auto` re-entry (the loopback `LatestFrameSlot` is owned by the
  `InputPipeline`, so a zero-blip rebuild needs slot/input decoupling).
- **`engine_ready` race tightening** — currently survivable (one
  unnecessary placeholder push per resume race) but worth moving
  `store(false)` into `spawn_reload_thread`'s body to close the
  window.
- **`double_spawn_guard` unit test** — extract a helper from the
  current inline guard in `handle_idle_edge::ResumeActive` for
  focused testing.
- **Relative `placeholder_path` resolution** against the config-file
  directory.
- **`resume_latency_ms` histogram.** The reload thread already
  returns `ResumeOutcome.elapsed`; a histogram on top is the
  natural next step.

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
| 14. GTK4 GUI client over the control socket | done |
| 15. Idle mode (consumer-aware lifecycle) | done (DeepIdle removed in Stage 16; ONNX-drop reclamation deferred) |
| 16. GUI-initiated TOML persistence (`save_preset` / Save as / Revert) | done |

## License

Dual-licensed under MIT or Apache-2.0 (your choice).
