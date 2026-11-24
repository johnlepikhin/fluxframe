# Troubleshooting

Start with `fluxframe check`. It tests the devices, the preset and the model,
and prints a hint for every problem it finds. For problems that appear only
while running, increase the log level with `-v` and look at the metrics line
(see [Performance](performance.md#reading-the-metrics)).

## The virtual camera does not appear in the browser

Chrome, Chromium-based browsers and many WebRTC applications list a v4l2loopback
device only when the module is loaded with `exclusive_caps=1`. `fluxframe check`
warns about this: "v4l2loopback is NOT exposing exclusive_caps=1 for this
device". Reload the module:

```bash
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="FluxFrame Camera" exclusive_caps=1
```

The module cannot be removed while any application has the device open.

Even with `exclusive_caps=1`, browsers list the device only while frames are
being written to it. If the daemon is not running, or runs without idle mode and
cannot open the physical camera, the device stays empty and is not listed.
Enable idle mode: the placeholder is then written continuously, even while the
camera is unavailable.

After changing the module or restarting the daemon, reload the browser page or
restart the browser so that it enumerates devices again.

## The camera is busy

`input device unavailable — streaming placeholder, retrying with backoff`, or a
hint that another application is holding the device, means something else has
the physical camera open. This is often a browser tab or another video
application that uses the camera directly instead of FluxFrame Camera. The
daemon keeps retrying and takes the camera as soon as it is released.

To find the process holding it:

```bash
fuser -v /dev/video0
```

## Permission denied

The user running the daemon needs read access to the camera and write access to
the loopback device. On most distributions this means membership in the `video`
group:

```bash
sudo usermod -aG video "$USER"
```

Log out and back in for the change to take effect.

## The output device does not exist

The v4l2loopback module is not loaded, or it was loaded with a different
`video_nr`. `fluxframe list` shows the available loopback devices. Load the
module as described in [Installation](installation.md#virtual-camera), or set
`output.device` to the existing device.

If `check` warns that the output "does not look like a v4l2loopback",
`output.device` points at a real camera. Correct it before running; otherwise
the daemon would try to write into the camera device.

## The camera runs at a lower frame rate or resolution than requested

The startup log line `camera mode selected` shows the mode the daemon picked for
the capture size (`input.capture_width` × `input.capture_height`, or
`input.width` × `input.height` when those are not set) and `input.fps`. Compare
it with the modes the camera offers:

```bash
v4l2-ctl --list-formats-ext -d /dev/video0
```

Many USB 2.0 cameras deliver 1280x720 or more at a full frame rate only as
MJPEG. If the camera offers no mode that reaches the requested frame rate, the
daemon uses the fastest one. If no mode is large enough, the largest one is
scaled up and the picture is softer than the requested size suggests. Lower
the capture size to a mode the camera delivers at full rate.

## The auto-framed picture moves too often or too late

`auto_frame` waits until you have stayed off-centre by more than `deadband` for
`hold_secs` before it moves the frame. If it still moves while you sit still,
raise `deadband` (for example to 0.2) or `hold_secs`. If it follows too late,
lower `hold_secs`. A soft face when zoomed in means the crop has fewer pixels
than the output: set `input.capture_width` and `input.capture_height` larger
than `input.width` and `input.height`; see
[Presets and effects](presets-and-effects.md#auto_frame).

## ONNX Runtime cannot be loaded

An error with the hint "set ORT_DYLIB_PATH to the libonnxruntime.so path" means
the variable is missing or points to a wrong file. Set it to the full path of
the library, including the file name. When the daemon runs as a service, set it
in the service definition; variables from your interactive shell are not
inherited.

## Inference runs on the CPU although OpenVINO is installed

The startup log line `openvino probe failed; falling back to ORT CPU` includes
the reason. Common causes:

 - The OpenVINO libraries are not on the library search path of the daemon's
   environment. Set `OPENVINO_INSTALL_DIR` or `LD_LIBRARY_PATH`.
 - The NPU or GPU user-mode driver is missing, so only the CPU device is
   available. The OpenVINO CPU plugin also works, but it is much slower in CPU
   time than an accelerator.

To get an explicit error instead of the fallback, run with
`FLUXFRAME_FORCE_INFERENCE_BACKEND=openvino`.

## The preset is not found

"preset 'X' is not defined in the config" lists the available presets in the
hint. Without `--preset`, the daemon uses the preset named `default`, and fails
if the configuration has none. Check that the daemon reads the file you expect:
without `--config` it is `$XDG_CONFIG_HOME/fluxframe/fluxframe.toml`, and if
that file does not exist the daemon starts with built-in defaults, which contain
no presets.

"preset 'X' declares [background] but no [mask] section": background, foreground
and post chains need a mask to work with. Add a mask section with a model.

## The model is not found or fails to load

"model file not found" means the model is not where the preset says. A relative
model path is resolved against the directory of the configuration file, not the
directory the daemon was started from; see
[Configuration](configuration.md#relative-paths). Make sure the sidecar `.toml`
file is next to the model.

## The fallback preset runs instead of the selected one

The log shows `preset cannot run on this machine; switching to its fallback
preset` or `running the fallback preset`, and the GUI shows a warning naming
both presets. The error in the log line gives the reason:

 - "no allowed inference device can run this model: none of NPU can run the
   model": the model's sidecar restricts it to the NPU (`devices = ["NPU"]`),
   and OpenVINO cannot use the NPU. Check that the NPU driver is installed and
   that `/dev/accel/accel0` exists and is accessible to your user. The text
   after the colon is OpenVINO's own error.
 - "the OpenVINO runtime is unavailable": see [Inference runs on the CPU
   although OpenVINO is installed](#inference-runs-on-the-cpu-although-openvino-is-installed).
 - "the CPU backend is forced": `FLUXFRAME_FORCE_INFERENCE_BACKEND=cpu` is set,
   and the model cannot run on ONNX Runtime.
 - "inference error inside effect 'segmentation'" after the preset had already
   run: inference kept failing, for example because the NPU stopped responding.

A model file that is damaged also produces the first error, because OpenVINO
reads the file while trying each device. Run `tools/prepare-rvm.py` again if in
doubt.

After fixing the cause, select the preset again (`set_preset`, the GUI preset
list, or `reload`); the daemon does not switch back on its own.

## The GUI shows "Daemon Unreachable"

The reason is shown under the socket path:

 - "the daemon is not running or not listening on this socket": start the
   daemon, and check that `[control] enabled = true` is set.
 - "the socket file does not exist": the control socket is disabled, or the
   daemon and the GUI use different paths. Pass the daemon's path with
   `fluxframe-gui --socket`.
 - "permission denied": the daemon runs as a different user. The socket is
   accessible only to its owner.

The GUI keeps retrying at growing intervals; press Retry Now to connect
immediately after fixing the cause.

## Save is unavailable in the GUI

The daemon was started with `--no-default-config` and without `--config`, so it
has no file to write to. Restart it with a configuration file.

## Vertical colour bands in the output

Faint vertical colour bands, two to three pixels wide, come from the RGB to YUY2
conversion. Set `output.format = "RGB"`. It is also cheaper.

## The picture lags behind movement or the edge flickers

 - An edge that trails behind movement indicates too much temporal smoothing.
   Lower `smooth_temporal.factor`.
 - A flickering edge indicates too little. Raise the factor, or increase
   `feather.radius`.
 - Halos around the person usually disappear with a higher `threshold.level`,
   for example 0.8.
 - Stray patches of person-coloured background are removed by `largest_blob`.

With Robust Video Matting the video is delayed by one frame by design (see
[Performance](performance.md#robust-video-matting)). Do not add
`smooth_temporal` or `threshold` to its mask chain: the model is already stable
over time, and thresholding destroys its soft edge.

## Applications get an I/O error when opening the camera

If the metrics line shows `output_stream_up=0`, the daemon is not currently
feeding the virtual camera. The v4l2loopback driver rejects readers in that
state. The log contains
`loopback output torn down — camera unavailable to clients`. The condition
usually clears on its own, and the log then shows
`loopback OUTPUT stream is held again`. If it persists, restart the daemon.

## Idle mode does not engage, or engages during a call

 - The metrics line field `consumer_source` shows the detection method: 0 kernel
   events, 1 inotify, 2 polling, 3 disabled. Anything other than 0 means
   v4l2loopback is older than 0.13 or its events are unavailable, and the less
   reliable fallback is in use.
 - `consumer_status=2` means the state is unknown, which is treated as "reader
   present". This is the expected result with `presence_source = "kernel_event"`
   when kernel events are unavailable.
 - `consumer_status=0` together with `external_openers=1` means an application
   opened the device but is not streaming. Typically another application already
   occupies the device's single capture slot.
 - A growing `consumer_resync_corrections_total` means kernel events are being
   lost and the periodic re-read is correcting the state. This is harmless as
   long as `resync_interval_secs` is not 0.
 - Any application that reads the virtual camera keeps the physical camera on.
   This includes the GUI preview while its window is visible.

If idle mode interferes with a call, set `presence_source = "disabled"` as an
immediate workaround. The virtual camera stays visible, but the camera is no
longer released.

## Reporting a problem

Include:

 - the output of `fluxframe check` and `fluxframe list`;
 - the relevant part of the log with `-v`;
 - the configuration file;
 - the v4l2loopback version (`modinfo v4l2loopback`) and the application that
   consumes the camera.
