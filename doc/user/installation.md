# Installation

This page covers everything needed before the first `fluxframe run`: system
libraries, building, the virtual camera device, the inference runtime and the
segmentation model.

## System dependencies

Required at build time and at run time:

 - GStreamer 1.x, `gst-plugins-base`, `gst-plugins-good`, GLib
 - `pkg-config`
 - ONNX Runtime as a shared library (`libonnxruntime.so`)

Optional:

 - OpenVINO runtime, for inference on Intel NPU or GPU. On Intel hardware also
   install the user-mode drivers: `intel-npu-driver` for the NPU,
   `intel-compute-runtime` for the GPU.
 - `gst-plugin-pipewire`, for publishing to PipeWire instead of v4l2loopback.
 - A Vulkan loader and driver, only if you intend to try the GPU blur backend
   (it is not used by default; see [Performance](performance.md)).
 - GTK 4.18 or newer, libadwaita 1.8 or newer, graphene, pango, cairo and
   gdk-pixbuf, for `fluxframe-gui`.

Rust 1.85 or newer is required to build.

### Guix

The repository ships a manifest with the complete toolchain:

```bash
guix shell -m manifest.scm
```

Alternatively, install the libraries into your profile once:

```bash
guix install gstreamer gst-plugins-base gst-plugins-good glib onnxruntime \
    gtk libadwaita graphene pango cairo gdk-pixbuf
```

If you also use `guix home`, packages installed with `guix install` live in
`~/.guix-profile`, which is not on the `PKG_CONFIG_PATH` that `guix home` sets
up. Extend it in your shell initialisation:

```bash
export PKG_CONFIG_PATH="$HOME/.guix-profile/lib/pkgconfig:$PKG_CONFIG_PATH"
```

OpenVINO is not part of upstream Guix and is not listed in the manifest. If you
have it from a separate channel (`openvino-full`, `intel-npu-driver`,
`intel-compute-runtime`), make sure its libraries are on the loader path of the
shell that runs FluxFrame.

## Building

```bash
cargo build --release
```

This produces two binaries in `target/release/`: `fluxframe` (the daemon and
command-line tool) and `fluxframe-gui`. To install them into `~/.cargo/bin`:

```bash
cargo install --path crates/fluxframe-cli
cargo install --path crates/fluxframe-gui
```

The daemon is built with all optional features by default:

| Feature      | Provides                                      |
| ------------ | --------------------------------------------- |
| `ml`         | Segmentation, the mask chain and post effects |
| `image-fill` | The `image_fill` background effect            |
| `openvino`   | The OpenVINO inference backend                |
| `wgpu`       | The Vulkan blur backend                       |

A reduced build, for example without GPU and OpenVINO support:

```bash
cargo build --release -p fluxframe-cli --no-default-features --features ml,image-fill
```

## Virtual camera

FluxFrame writes its output to a v4l2loopback device. Install the module from
your distribution and load it:

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label="FluxFrame Camera" exclusive_caps=1
```

 - `video_nr=10` creates `/dev/video10`, which is the default output device in
   FluxFrame.
 - `card_label` is the name applications show in their camera list.
 - `exclusive_caps=1` is required for Chrome, Chromium-based browsers and other
   WebRTC clients. Without it the device does not appear in their camera
   selection at all. `fluxframe check` reports this condition.

To load the module with these options at boot, create
`/etc/modules-load.d/v4l2loopback.conf`:

```
v4l2loopback
```

and `/etc/modprobe.d/v4l2loopback.conf`:

```
options v4l2loopback devices=1 video_nr=10 card_label="FluxFrame Camera" exclusive_caps=1
```

Distributions differ in how kernel modules are configured; on Guix System use
the `kernel-loadable-modules` and `kernel-arguments` fields of the operating
system declaration instead.

v4l2loopback 0.13 or newer is recommended: idle mode uses its client-usage
events to detect when an application starts and stops reading the camera. Older
versions work with a less precise fallback (see [Idle mode](idle-mode.md)).

Your user must be able to open the physical camera and the loopback device,
which on most systems means membership in the `video` group.

## ONNX Runtime

ONNX Runtime is loaded dynamically at startup. Set `ORT_DYLIB_PATH` to the full
path of the shared library:

```bash
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
```

On Guix the library is in the profile where `onnxruntime` is installed, for
example `~/.guix-profile/lib/libonnxruntime.so` or
`~/.guix-home/profile/lib/libonnxruntime.so`.

ONNX Runtime is always required, even when inference runs on OpenVINO: it is the
fallback backend and it is used by `fluxframe check` and `fluxframe benchmark`.

## OpenVINO (optional)

When the daemon starts, it tries OpenVINO first with the device string
`AUTO:NPU,GPU,CPU`, which picks the NPU if one is available. If the OpenVINO
runtime cannot be loaded, the daemon logs
`openvino probe failed; falling back to ORT CPU` and continues with ONNX
Runtime.

The OpenVINO C library (`libopenvino_c.so`) must be findable through
`OPENVINO_INSTALL_DIR` or the regular library search path (`LD_LIBRARY_PATH`).
Inference on the NPU is the single largest performance factor on laptops that
have one; see [Performance](performance.md).

The backend and device can be forced with environment variables:

```bash
FLUXFRAME_FORCE_INFERENCE_BACKEND=cpu fluxframe run        # ONNX Runtime only
FLUXFRAME_FORCE_INFERENCE_BACKEND=openvino fluxframe run   # fail if OpenVINO is unavailable
FLUXFRAME_OPENVINO_DEVICE=GPU fluxframe run                # any OpenVINO device string
```

## Segmentation model

FluxFrame uses MediaPipe Selfie Segmentation (general model, 256x256 input)
converted to ONNX. The model is not included in the repository. The converted
files are published by PINTO_model_zoo, entry 109, under the Apache-2.0 license.

Download and unpack the archive, then copy the float32 ONNX model:

```bash
mkdir -p ~/.config/fluxframe/models
cd "$(mktemp -d)"
curl -L -o resources.tar.gz \
    https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/109_Selfie_Segmentation/resources.tar.gz
tar -xzf resources.tar.gz
cp saved_model_openvino/model_float32.onnx ~/.config/fluxframe/models/selfie_segmentation.onnx
```

The download URL comes from the `download.sh` script in the
`109_Selfie_Segmentation` directory of the PINTO_model_zoo repository on GitHub.
Check it there if the link above stops working.

Every model needs a sidecar file with the same base name and the `.toml`
extension, next to the model. It describes the tensor layout. Create
`~/.config/fluxframe/models/selfie_segmentation.toml`:

```toml
# MediaPipe Selfie Segmentation (general, 256x256), Apache-2.0
# Source: PINTO_model_zoo #109, model_float32.onnx
name = "mediapipe-selfie-segmentation"

input_width = 256
input_height = 256
input_layout = "NHWC"
input_dtype = "f32"
input_color = "RGB"
input_scale = 0.00392156862745098  # 1/255
input_zero_point = 0.0

output_index = 0
output_layout = "NHWC"
output_type = "probabilities"
threshold = 0.5
```

If the sidecar is missing, the daemon logs a warning and the segmentation does
not work correctly. The sidecar format is described in [Presets and
effects](presets-and-effects.md#model-sidecar-files).

To confirm that the model loads and to see the inference speed on your CPU:

```bash
fluxframe benchmark --model ~/.config/fluxframe/models/selfie_segmentation.onnx --duration 5
```

## Verifying the installation

```bash
fluxframe list
```

lists capture devices and loopback outputs. Then, with a configuration in place
(start from [examples/fluxframe.toml](../../examples/fluxframe.toml)):

```bash
fluxframe check
```

`check` verifies GStreamer, the input and output devices, the `exclusive_caps`
setting, the preset and the model, and prints every problem it finds with a
hint. It exits with status 0 when everything is usable. Model paths in the
example configuration are relative to the configuration file; see
[Configuration](configuration.md#relative-paths).
