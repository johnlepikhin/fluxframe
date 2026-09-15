# Configuration

FluxFrame is configured with a single TOML file. It contains global settings
(input, output, threading, logging, the control socket, idle mode) and any
number of named presets. This page describes the file itself and the global
settings. Presets and effects are described in [Presets and
effects](presets-and-effects.md), and the `[idle]` table in [Idle
mode](idle-mode.md).

A complete, commented example is
[examples/fluxframe.toml](../../examples/fluxframe.toml).

## File location

`fluxframe run` and `fluxframe check` look for the configuration in this order:

 1. The file given with `--config PATH`. A missing file is an error.
 2. `$XDG_CONFIG_HOME/fluxframe/fluxframe.toml`, or
    `~/.config/fluxframe/fluxframe.toml` when `XDG_CONFIG_HOME` is not set.
 3. Built-in defaults, if the file from step 2 does not exist.

`--no-default-config` skips step 2. This is useful for scripts and tests that
must not read the user's configuration. With `--no-default-config` and without
`--config`, the daemon has no file to save to, so saving presets from the GUI or
the control socket is unavailable.

When the default path is used, it is also the file that Save writes to, even if
the file did not exist at startup. The first save creates it together with its
parent directory.

The file must be a regular file no larger than 1 MiB. Unknown keys are rejected
with an error that names the key, so a misspelled setting never goes unnoticed.

## Relative paths

Relative paths in the configuration (a preset's `model`, `image_fill` images,
`idle.placeholder_path`) are resolved against the directory that contains the
configuration file, so a configuration and the files next to it can be moved
together and the daemon can be started from any directory. The file keeps the
paths as written: saving a preset from the GUI does not turn them into absolute
paths.

Without a configuration file (`--no-default-config` and no `--config`), relative
paths are resolved against the daemon's working directory. `control.socket_path`
and device paths are used as given.

## `[input]`

| Key                       | Type    | Default         | Description                                                                                                                        |
| ------------------------- | ------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `device`                  | string  | `"/dev/video0"` | Capture source: a device path, `"auto"` or `"testsrc"`.                                                                            |
| `width`                   | integer | `1280`          | Requested capture width, 1 to 16384.                                                                                               |
| `height`                  | integer | `720`           | Requested capture height, 1 to 16384.                                                                                              |
| `fps`                     | integer | `30`            | Requested capture frame rate, 1 to 240.                                                                                            |
| `format`                  | string  | `"RGB"`         | Pixel format negotiated with the source: `RGB`, `RGBA`, `BGR`, `YUY2`, `NV12` or `GRAY8`. Presets with segmentation require `RGB`. |
| `acquire_backoff_base_ms` | integer | `500`           | Initial delay before retrying a busy or missing camera. Must be greater than 0.                                                    |
| `acquire_backoff_max_ms`  | integer | `5000`          | Maximum retry delay. At least `acquire_backoff_base_ms`, at most 30000.                                                            |

`device` values:

 - A path such as `/dev/video0` uses that camera.
 - `"auto"` uses the first capture device that can be opened and keeps looking
   if there is none. The output device is never picked. See `[input.auto]`.
 - `"testsrc"` uses a synthetic test pattern instead of a camera. It is useful
   for checking the setup and for benchmarks.

The width, height and frame rate must be a mode the camera supports.
`v4l2-ctl --list-formats-ext -d /dev/video0` lists the modes. Many USB 2.0
cameras deliver high resolutions only at a low frame rate in uncompressed
formats. A smaller mode at a full frame rate usually looks better and costs less
CPU.

The retry delay starts at `acquire_backoff_base_ms` and doubles up to
`acquire_backoff_max_ms`. Only temporary conditions (device busy, device absent)
are retried. Permanent errors stop the daemon.

### `[input.auto]`

Used only when `input.device = "auto"`.

| Key                  | Type           | Default | Description                                                        |
| -------------------- | -------------- | ------- | ------------------------------------------------------------------ |
| `poll_interval_secs` | integer        | `2`     | How often to look for a camera while none is available, 1 to 3600. |
| `exclude_devices`    | array of paths | `[]`    | Devices never to pick, in addition to the output device.           |

## `[output]`

| Key      | Type   | Default          | Description                                                                          |
| -------- | ------ | ---------------- | ------------------------------------------------------------------------------------ |
| `device` | string | `"/dev/video10"` | Where to publish the processed video.                                                |
| `format` | string | `"RGB"`          | Pixel format written to the output: `RGB`, `RGBA`, `BGR`, `YUY2`, `NV12` or `GRAY8`. |
| `scale`  | float  | `1.0`            | Output size relative to the input, 0.05 to 1.0.                                      |

`device` values:

 - `/dev/videoN` writes to a v4l2loopback device. This is the normal mode, and
   the only one that supports idle mode.
 - `pipewire` or `pipewire:NAME` publishes a PipeWire video stream (requires
   `gst-plugin-pipewire`).
 - `auto` opens a local preview window.
 - `fakesink` discards the frames, for benchmarks and tests.

The default `RGB` is the format the effect chain works in, so the output needs
no colour conversion (lower CPU usage) and shows none of the faint vertical
colour banding that an RGB to YUY2 conversion introduces. Set `YUY2` only for a
consumer that does not accept RGB.

The output resolution is always the input resolution multiplied by `scale`,
rounded down to an even number of pixels. The output frame rate always equals
the input frame rate. Upscaling is not supported.

## `[realtime]`

| Key                     | Type    | Default | Description                                                                                      |
| ----------------------- | ------- | ------- | ------------------------------------------------------------------------------------------------ |
| `processing_threads`    | integer | `0`     | Worker threads for image processing. `0` selects automatically (currently 2); otherwise 1 to 64. |
| `metrics_interval_secs` | integer | `30`    | Interval of the periodic metrics log line. `0` disables it; otherwise up to 3600.                |

Latency and frame dropping are not configurable. The camera feeds a slot that
holds one frame: when processing falls behind, the waiting frame is replaced by
the newest one instead of queueing, so latency does not accumulate.

The environment variable `FLUXFRAME_RAYON_THREADS` overrides
`processing_threads`. The trade-off between thread count, latency and CPU usage
is discussed in [Performance](performance.md).

## `[logging]`

| Key     | Type   | Default  | Description                                                         |
| ------- | ------ | -------- | ------------------------------------------------------------------- |
| `level` | string | `"info"` | Log level of FluxFrame's own messages, or a full filter expression. |

A plain level (`error`, `warn`, `info`, `debug`, `trace`) applies to FluxFrame's
messages only. A value containing `=` is used as a complete filter, for example
`"fluxframe=info,fluxframe::metrics=warn"` to hide the periodic metrics line.

The level is chosen with the following precedence: the `RUST_LOG` environment
variable, then the `-v` command-line flags, then this setting. Logs go to
standard output, coloured when it is a terminal. `NO_COLOR` disables colours;
`CLICOLOR_FORCE` forces them.

## `[control]`

| Key           | Type    | Default   | Description                                           |
| ------------- | ------- | --------- | ----------------------------------------------------- |
| `enabled`     | boolean | `false`   | Open the control socket. Required by `fluxframe-gui`. |
| `socket_path` | path    | see below | Location of the socket.                               |

The default socket path is `$XDG_RUNTIME_DIR/fluxframe.sock`, or
`/tmp/fluxframe.sock` when `XDG_RUNTIME_DIR` is not set. The socket is created
with mode 0600, so only the user running the daemon can connect. On a multi-user
system without `XDG_RUNTIME_DIR`, set an explicit path. The protocol is
described in [Control socket](control-socket.md).

## `[idle]`

Idle mode releases the camera when no application reads the output. It is
disabled by default. All keys are described in [Idle mode](idle-mode.md).

## `[presets.NAME]`

Named processing pipelines. `fluxframe run` uses the preset `default` unless
`--preset` is given. See [Presets and effects](presets-and-effects.md).

## Environment variables

| Variable                            | Effect                                                                                      |
| ----------------------------------- | ------------------------------------------------------------------------------------------- |
| `ORT_DYLIB_PATH`                    | Path to `libonnxruntime.so`. Required.                                                      |
| `FLUXFRAME_FORCE_INFERENCE_BACKEND` | `auto` (default), `cpu` (ONNX Runtime) or `openvino` (no fallback).                         |
| `FLUXFRAME_OPENVINO_DEVICE`         | OpenVINO device string. Default `AUTO:NPU,GPU,CPU`.                                         |
| `FLUXFRAME_FORCE_BLUR_BACKEND`      | `auto` (default, CPU), `cpu` or `wgpu`. Forcing `wgpu` without a Vulkan device is an error. |
| `FLUXFRAME_RAYON_THREADS`           | Number of processing threads; overrides `realtime.processing_threads`.                      |
| `RUST_LOG`                          | Log filter; overrides `-v` and `logging.level`.                                             |
| `NO_COLOR`, `CLICOLOR_FORCE`        | Disable or force coloured log output.                                                       |

## Applying changes

The daemon reads the file at startup. While it runs, presets can be changed live
through the GUI or the control socket, and the `reload` command re-reads the
file and rebuilds the active preset. The global tables (`[input]`, `[output]`,
`[realtime]`, `[logging]`, `[control]`, `[idle]`) are only applied at startup.
Restart the daemon after changing them.
