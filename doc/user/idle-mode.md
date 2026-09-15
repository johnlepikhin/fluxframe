# Idle mode

A virtual camera daemon runs all day, but the camera is used only during calls.
Idle mode detects when no application is reading the virtual camera and then
stops the physical camera and the processing. The camera LED goes off, CPU usage
drops below 1% of a core, and the virtual camera keeps showing a static
placeholder so applications continue to list it. When an application starts
reading again, live video resumes automatically.

Idle mode is disabled by default:

```toml
[idle]
enabled = true
```

It requires a v4l2loopback output (`output.device = "/dev/videoN"`) and Linux.
With other outputs, the daemon logs a warning and runs without idle mode.

## Behaviour

 - **Active.** An application is reading the virtual camera. The camera is open
   and every frame is processed.
 - **Cooldown.** The last reader stopped. The camera stays open for
   `teardown_secs` (default 5 seconds), so an application that closes and
   reopens the camera, as many do on startup, does not cause it to switch off
   and on.
 - **Idle.** The camera is closed and the processing stops. The virtual camera
   receives the placeholder frame at `max(fps, min_visibility_fps)` frames per
   second.

When a reader appears in the idle state, the daemon immediately writes a
placeholder frame and reopens the camera in the background. Live video typically
appears within about 300 ms; until then the reader sees the placeholder.

A few camera models keep their LED lit for one to three seconds after being
closed. This is controlled by the camera firmware.

## Placeholder

The placeholder is either a solid colour or a static image:

```toml
[idle]
enabled = true
placeholder = "image"
placeholder_path = "/home/user/Pictures/away.png"
```

The image is loaded once at startup and scaled to the output size. The path must
be absolute.

Chrome and other WebRTC clients list a v4l2loopback device only while it
receives frames. `min_visibility_fps` keeps the placeholder frequent enough for
the device to stay in their camera list. The default of 10 frames per second is
sufficient; lowering it may cause the camera to disappear from browsers while
idle.

## How readers are detected

The detection method is set by `presence_source`:

| Value          | Behaviour                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------ |
| `auto`         | Kernel events if available, otherwise the fallback method. Default.                                                |
| `kernel_event` | Kernel events only. If they are unavailable, idle mode never engages.                                              |
| `inotify`      | Always use the fallback method.                                                                                    |
| `disabled`     | Never consider the camera unused. Idle mode never engages, but the placeholder and device visibility keep working. |

With v4l2loopback 0.13 or newer, the driver notifies the daemon whenever an
application starts or stops streaming from the virtual camera. This is exact and
works for any application, including sandboxed browsers and processes of other
users. An application that only lists devices without streaming is not counted
as a reader. Until the first notification after startup, the state is treated as
"reader present", so idle mode never engages on a guess.

Every `resync_interval_secs` (default 30) the daemon asks the driver for the
current state again. If a notification was lost, the log shows
`consumer state drift` and the state is corrected. Changes to this interval take
effect after a restart.

With older v4l2loopback versions, the daemon watches the device node for open
and close events and looks for processes that hold it open. This cannot see
processes of other users, so it is less reliable. If the watch mechanism is
unavailable, it falls back to polling every `poll_interval_ms`.

If idle mode ever engages while a call is in progress, set
`presence_source = "disabled"`. Unlike `enabled = false`, this keeps the
placeholder that makes the camera visible to browsers at startup and while the
physical camera is unavailable.

## Settings

| Key                    | Type                | Default        | Description                                                                                                      |
| ---------------------- | ------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------- |
| `enabled`              | boolean             | `false`        | Enable idle mode.                                                                                                |
| `placeholder`          | string              | `"color"`      | `color` or `image`.                                                                                              |
| `placeholder_rgb`      | array of 3 integers | `[16, 16, 16]` | Placeholder colour.                                                                                              |
| `placeholder_path`     | path                | none           | Absolute path to a PNG or JPEG. Required when `placeholder = "image"`.                                           |
| `fps`                  | integer             | `1`            | Placeholder frame rate, 1 to 60.                                                                                 |
| `min_visibility_fps`   | integer             | `10`           | Minimum placeholder frame rate that keeps the device visible to browsers, 1 to 60.                               |
| `teardown_secs`        | integer             | `5`            | Delay between the last reader leaving and closing the camera, 1 to 3600.                                         |
| `presence_source`      | string              | `"auto"`       | Detection method, see above.                                                                                     |
| `resync_interval_secs` | integer             | `30`           | How often the reader state is re-read from the driver. 0 disables; otherwise 5 to 3600. Applied at startup only. |
| `poll_interval_ms`     | integer             | `250`          | Polling interval of the last-resort fallback, 100 to 5000.                                                       |

## Log messages

Idle transitions are logged with the target `fluxframe::idle`:

| Message                                                     | Meaning                                                                            |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `entering idle — tearing down input pipeline`               | No reader for `teardown_secs`; the camera is being closed.                         |
| `consumer reconnected — spawning reload thread`             | A reader appeared; the camera is being reopened.                                   |
| `resumed active processing`                                 | Live video is flowing again.                                                       |
| `consumer state drift`                                      | A periodic re-read disagreed with the tracked state; the state has been corrected. |
| `failed to build idle placeholder — idle mode disabled`     | The placeholder could not be created, usually because of a bad `placeholder_path`. |
| `loopback output torn down — camera unavailable to clients` | The daemon no longer feeds the virtual camera; applications will get errors.       |
| `loopback OUTPUT stream is held again`                      | The condition above has cleared.                                                   |

The periodic metrics line carries the detector state: `consumer_status` (0 no
reader, 1 reader present, 2 unknown), `consumer_source` (0 kernel events, 1
inotify, 2 polling, 3 disabled), `idle_entered_total`, and the camera recovery
counters. See [Troubleshooting](troubleshooting.md) for interpreting them.
