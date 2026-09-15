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

## The camera mode is not supported

The requested `input.width`, `input.height` and `input.fps` must match a mode
the camera offers. List the modes:

```bash
v4l2-ctl --list-formats-ext -d /dev/video0
```

Many USB 2.0 cameras deliver 1280x720 or more only at 5-10 fps in uncompressed
formats. Choose a smaller mode with the full frame rate.

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

"model file not found" usually means a relative model path is resolved against a
different working directory than intended. Relative paths are resolved against
the directory the daemon was started from, not the configuration file's
directory. Use an absolute path or set the working directory. Make sure the
sidecar `.toml` file is next to the model.

## The GUI shows "Daemon Unreachable"

The reason is shown under the socket path:

 - "the daemon is not running or not listening on this socket": start the
   daemon, and check that `[control] enabled = true` is set.
 - "the socket file does not exist": the control socket is disabled, or the
   daemon and the GUI use different paths. Pass the daemon's path with
   `fluxframe-gui --socket`.
 - "permission denied": the daemon runs as a different user. The socket is
   accessible only to its owner.

The GUI does not reconnect by itself. Press Retry after fixing the cause.

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
