# GUI

`fluxframe-gui` is a live editor for the running daemon. It shows the processed
video, lets you switch presets, edit effect chains and parameters with immediate
effect, and save the result to the configuration file.

The GUI does not process video itself. It is a client of the daemon's control
socket, so the daemon must be running with the socket enabled:

```toml
[control]
enabled = true
```

## Starting

```bash
fluxframe-gui
```

The GUI connects to the same default socket path as the daemon
(`$XDG_RUNTIME_DIR/fluxframe.sock`). If the daemon uses a different path, pass
it explicitly:

```bash
fluxframe-gui --socket /path/to/fluxframe.sock
```

If the daemon is not reachable, the window shows "Daemon Unreachable" with the
socket path and the reason: the daemon is not running, the socket file does not
exist (usually `[control] enabled` is not set), or permission is denied (the
daemon runs as a different user). Start or fix the daemon and press Retry. The
GUI does not reconnect automatically.

## Window

The header bar shows the active preset. When there are unsaved edits, the
subtitle reads "Unsaved changes". Next to it are:

 - The preset list. Selecting a preset activates it in the daemon.
 - Save: writes the active preset to the configuration file. The tooltip shows
   the file path.
 - Save As: saves the current state under a new preset name. The active preset
   does not change.
 - Revert: discards unsaved edits and restores the preset as last saved. Hidden
   in narrow windows; also available in the main menu.

The main menu contains Discard Unsaved Changes, Reload Configuration, Keyboard
Shortcuts and About FluxFrame. The About dialog includes troubleshooting
details: socket path, connection state, active preset, configuration path and
the features the daemon was built with.

### Preview

The upper part of the window shows the output of the virtual camera. The divider
between the preview and the editor can be dragged. Without a signal, the preview
shows "No Preview Signal" and checks again every few seconds.

The preview runs only while the window is visible. A minimised window does not
count as a reader, so it does not keep the daemon out of idle mode. While the
window is visible, however, the preview is a reader like any other application,
and the camera stays on.

The preview always reads `/dev/video10`, regardless of the daemon's
`output.device`.

### Chain editor

The editor has four groups: Mask, Background, Foreground and Post. Each group
lists the effects of its chain in processing order. Each effect is an expandable
row with:

 - a switch that enables or disables the effect without removing it;
 - buttons to move the effect up or down, or remove it from the chain;
 - one control per parameter.

The + button in a group header adds an effect from the list of effects the
daemon supports.

Parameter controls match the parameter type: sliders for numeric ranges
(logarithmic where appropriate), spin buttons, switches, colour pickers, file
choosers and drop-down lists. Changes are sent to the daemon after a short delay
while you drag. If the daemon rejects a value, a notification explains why and
the control returns to the last accepted value.

## Unsaved changes

All edits apply to the running daemon immediately but are kept in memory until
saved. The configuration file changes only when you press Save or Save As.

Save writes only the active preset's section. Comments and the rest of the file
are preserved, and the file is replaced atomically. Save is unavailable when the
daemon has no configuration file to write to, which happens when it was started
with `--no-default-config` and without `--config`.

Actions that would lose unsaved edits ask for confirmation:

 - switching to another preset;
 - Revert;
 - reloading the configuration from disk.

Preset names for Save As may contain letters, digits, `_` and `-`, and must not
already exist.

## Keyboard shortcuts

| Shortcut             | Action                                           |
| -------------------- | ------------------------------------------------ |
| `Ctrl+S`             | Save                                             |
| `Ctrl+Shift+S`       | Save As                                          |
| `Ctrl+R`, `F5`       | Reload configuration                             |
| `Ctrl+1` to `Ctrl+9` | Activate the preset at that position in the list |
| `Ctrl+?`             | Show keyboard shortcuts                          |
| `Ctrl+Q`             | Quit                                             |

## Stored state

The window size, maximised state and preview divider position are stored in
`$XDG_CONFIG_HOME/fluxframe/gui.json` (`~/.config/fluxframe/gui.json` by
default).
