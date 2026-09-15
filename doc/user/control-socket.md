# Control socket

The control socket lets other programs inspect and change the running daemon:
switch presets, edit effect parameters and chains, reload the configuration and
save changes. The GUI uses it, and it is equally suitable for scripts, keyboard
shortcuts in a desktop environment, or status bar widgets.

Enable it in the configuration:

```toml
[control]
enabled = true
```

## Transport

 - UNIX domain socket at `$XDG_RUNTIME_DIR/fluxframe.sock` by default
   (`/tmp/fluxframe.sock` without `XDG_RUNTIME_DIR`), or at
   `control.socket_path`.
 - The socket has mode 0600. There is no other authentication: any process of
   the same user can control the daemon.
 - Each request is a single line of JSON. Each request receives a single line of
   JSON in response. Several requests can be sent over one connection.
 - A request line is limited to 64 KiB and a response to 256 KiB. An oversized
   request receives an error and the connection is closed.
 - Unknown commands and unknown fields are rejected.

Any UNIX socket client works, for example `socat`:

```bash
SOCK=$XDG_RUNTIME_DIR/fluxframe.sock
echo '{"cmd":"list_presets"}' | socat - UNIX-CONNECT:$SOCK
```

## Responses

Success:

```json
{"ok":"true","data":["blur","default","raw"]}
```

Failure:

```json
{"ok":"false","error":"preset 'blurr' is not defined in the config","hint":"available presets: blur, default, raw"}
```

The value of `ok` is the string `"true"` or `"false"`, not a JSON boolean.
`hint` is optional. `data` is `null` for commands that return nothing.

## Commands

### Inspecting

| Request                                         | Result                                                                                                              |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `{"cmd":"list_presets"}`                        | Array of preset names.                                                                                              |
| `{"cmd":"current_preset"}`                      | Name of the active preset.                                                                                          |
| `{"cmd":"get_config"}`                          | The active preset as JSON, including unsaved changes.                                                               |
| `{"cmd":"get_config","path":"background.blur"}` | A part of the active preset, addressed by a dot-separated path.                                                     |
| `{"cmd":"list_effects"}`                        | All available effects by group, with their parameters, types, defaults and ranges, and the daemon's build features. |
| `{"cmd":"config_path"}`                         | `{"path":"..."}`: the file that saves write to, or `null` if there is none.                                         |

### Changing the active preset

All changes apply immediately and are kept in memory until saved.

`set` changes one parameter. The path has the form `section.effect.parameter`,
where section is `mask`, `background`, `foreground` or `post`:

```json
{"cmd":"set","path":"background.blur.radius","value":40}
{"cmd":"set","path":"background.color_fill.rgb","value":[0,255,0]}
```

The value is validated against the parameter's type and range. Parameter changes
take effect in well under a millisecond, so `set` is suitable for continuous
adjustments.

`set_chain` replaces the chain of a section:

```json
{"cmd":"set_chain","section":"background","chain":["blur","vignette"]}
```

Newly added effects use default parameters. Parameter tables of effects that are
no longer in the chain are removed.

`set_enabled` enables or disables an effect without removing it from the chain:

```json
{"cmd":"set_enabled","section":"background","effect":"blur","enabled":false}
```

`enabled` cannot be changed with `set`.

### Presets and files

| Request                                     | Effect                                                                                                                                                                                              |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `{"cmd":"set_preset","name":"blur"}`        | Activates a preset. The pipeline is rebuilt; with a model change this takes up to about half a second. Unsaved changes to the previous preset are discarded.                                        |
| `{"cmd":"reload"}`                          | Re-reads the configuration file and rebuilds the active preset. Command-line overrides remain in effect. Fails if the daemon has no configuration file or the active preset no longer exists in it. |
| `{"cmd":"save_preset"}`                     | Writes the active preset, including unsaved changes, to the configuration file.                                                                                                                     |
| `{"cmd":"save_preset_as","name":"evening"}` | Writes the current state as a new preset. The name may contain letters, digits, `_` and `-` and must not exist yet. The active preset does not change.                                              |

Saving rewrites only the `[presets.NAME]` section and its subsections. Comments
and all other content of the file are preserved. The file is written to a
temporary file and renamed, so an interrupted save never leaves a partially
written configuration.

## Limitations

Only presets can be changed through the socket. The global tables (`[input]`,
`[output]`, `[realtime]`, `[logging]`, `[control]`, `[idle]`) are read at
startup and are never modified by a save. To change them, edit the file and
restart the daemon.

The socket stays available for the whole life of the daemon, including while it
waits for a camera that is busy or unplugged. During such a wait there is no
running pipeline: `list_presets`, `current_preset`, `get_config`,
`list_effects` and `config_path` are answered as usual, and every other command
fails with `pipeline is not running (waiting for the camera)`. Retry it once
the camera is back. Unsaved changes made before the camera was lost are kept and
applied to the pipeline when it restarts.

## Examples

Cycle through presets from a desktop keyboard shortcut:

```bash
#!/bin/sh
SOCK=${XDG_RUNTIME_DIR:-/tmp}/fluxframe.sock
q() { echo "$1" | socat - UNIX-CONNECT:"$SOCK"; }

current=$(q '{"cmd":"current_preset"}' | jq -r .data)
next=$(q '{"cmd":"list_presets"}' | jq -r --arg c "$current" \
    '.data as $p | $p[(($p | index($c)) + 1) % ($p | length)]')
q "{\"cmd\":\"set_preset\",\"name\":\"$next\"}" > /dev/null
```

Temporarily turn off the background effect:

```bash
echo '{"cmd":"set_enabled","section":"background","effect":"blur","enabled":false}' \
    | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/fluxframe.sock
```
