# Running

The `fluxframe` binary is both the daemon and a set of diagnostic tools. This
page describes its commands, running it permanently as a user service, and how
it handles cameras.

## Command-line reference

```
fluxframe [-v...] <COMMAND> [OPTIONS]
```

`-v` raises the log level to debug, `-vv` to trace. The `RUST_LOG` environment
variable takes precedence over it.

Exit status: 0 on success, 1 when a command fails, 2 when logging cannot be
initialised. Errors are printed as an `Error:` line followed by a `Hint:` line
with the suggested fix.

### Common options

`check` and `run` accept:

| Option                  | Description                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------ |
| `--config PATH`         | Configuration file. Default: `$XDG_CONFIG_HOME/fluxframe/fluxframe.toml`.            |
| `--no-default-config`   | Do not read the default configuration file.                                          |
| `--input PATH_OR_NAME`  | Override `input.device`: a path, `auto` or `testsrc`.                                |
| `--output PATH_OR_NAME` | Override `output.device`: a path, `auto`, `fakesink`, `pipewire` or `pipewire:NAME`. |

### `fluxframe list`

Lists V4L2 capture devices and v4l2loopback devices that can be used as output,
with their names. Loopback devices are marked `virtual=true`. If none are found,
it prints how to load the v4l2loopback module.

### `fluxframe check`

```bash
fluxframe check [--preset NAME]
```

Checks, in order:

 - GStreamer initialises.
 - The input device can be opened.
 - The output device can be written, looks like a v4l2loopback device and has
   `exclusive_caps=1`.
 - The preset (default: `default`) exists, uses known effects and can be built.
 - The model file exists and loads.

All problems are reported, not only the first one. Run `check` after changing
the configuration or the system setup, before relying on the daemon.

### `fluxframe run`

```bash
fluxframe run [--preset NAME] [--width PIXELS] [--height PIXELS] [--fps RATE]
```

Starts the processing pipeline and runs until interrupted with Ctrl+C or
SIGTERM.

 - `--preset` selects the preset. Default: `default`. If the preset does not
   exist, the daemon exits and lists the available presets.
 - `--width`, `--height` and `--fps` override the input capture mode. The output
   size always follows the input size multiplied by `output.scale`, and the
   output frame rate always equals the input frame rate.

Command-line overrides remain in effect when the configuration is reloaded
through the control socket.

### `fluxframe benchmark`

```bash
fluxframe benchmark --model PATH [--duration SECONDS]
```

Measures inference speed alone. The model is fed a synthetic input for
`--duration` seconds (default 30), and the command prints the number of runs and
the 50th percentile, 95th percentile and maximum time per inference in
microseconds. It reads the model's sidecar file but no configuration or preset.

The benchmark always runs on ONNX Runtime on the CPU. It does not measure
capture, effects, output or OpenVINO. To measure the complete pipeline, use the
metrics of a running daemon; see
[Performance](performance.md#reading-the-metrics).

## Running as a user service

The daemon is designed to run permanently in the background. With idle mode
enabled, it releases the camera and uses almost no CPU while no application is
reading the virtual camera. The repository does not ship service files. The
following systemd user unit is an example to adapt:

```ini
# ~/.config/systemd/user/fluxframe.service
[Unit]
Description=FluxFrame virtual camera

[Service]
WorkingDirectory=%h/.config/fluxframe
Environment=ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
ExecStart=%h/.cargo/bin/fluxframe run
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now fluxframe
journalctl --user -u fluxframe -f
```

The working directory makes relative model paths in the configuration resolve
against `~/.config/fluxframe`. On systems with GNU Shepherd, define an
equivalent service in your home configuration with the same command, environment
and working directory.

A service manager does not load the v4l2loopback module. Load it at boot as
described in [Installation](installation.md#virtual-camera).

## Camera handling

### Fixed device

With `input.device = "/dev/video0"`, the daemon uses that device. If the camera
is busy or not present when it is needed, the daemon retries with exponential
backoff (`input.acquire_backoff_base_ms` to `input.acquire_backoff_max_ms`)
instead of exiting. The log shows
`input device unavailable — streaming placeholder, retrying with backoff` once,
not on every attempt.

With idle mode enabled, the virtual camera is created and shows the placeholder
before the physical camera is opened. Applications therefore see "FluxFrame
Camera" even while the physical camera is busy, and live video appears as soon
as the camera becomes available.

### Automatic selection

With `input.device = "auto"`, the daemon picks the first capture device that can
be opened, skipping the output device and anything in
`input.auto.exclude_devices`. If there is no camera, it checks again every
`input.auto.poll_interval_secs` seconds.

If the camera disappears while running (for example, a USB camera is unplugged),
the daemon recovers when the camera returns or, in automatic mode, when another
camera becomes available. With idle mode enabled, the virtual camera stays
available throughout.

### Synthetic input

`--input testsrc` replaces the camera with a test pattern. Combined with
`--output fakesink`, it runs the full pipeline without any devices, which is
useful for checking a configuration and for measuring performance:

```bash
fluxframe run --input testsrc --output fakesink --preset blur
```

The test pattern fills the whole frame, so the mask covers everything and some
effects (notably `auto_frame`) do less work than with a real camera.

## Switching presets

A running daemon can switch presets without restarting, through the GUI
(`Ctrl+1` to `Ctrl+9` or the preset list) or the control socket:

```bash
echo '{"cmd":"set_preset","name":"blur"}' | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
```

Switching rebuilds the pipeline. When the new preset loads a different model,
this takes up to about half a second. Both require `[control] enabled = true`.
