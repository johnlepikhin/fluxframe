# Presets and effects

A preset is a named description of how each frame is processed. The
configuration can hold any number of presets. One of them is active at a time,
and it can be switched while the daemon runs.

## How a frame is processed

A preset has up to four sections: `mask`, `background`, `foreground` and `post`.
When the `mask` section is present, every frame goes through these steps:

 1. The segmentation model produces a mask at the model's resolution: 256x256
    for the Selfie Segmentation model, the full frame size for Robust Video
    Matting. Each mask value is the confidence that the pixel belongs to a
    person.
 2. The mask chain refines the mask at model resolution: thresholding, removing
    stray regions, smoothing over time, softening the edge.
 3. The mask is scaled up to the frame size, unless it already has that size.
 4. The background chain runs on a copy of the frame. The copy still contains
    the person; effects that need to tell the person from the room read the
    mask.
 5. The foreground chain runs on the original frame.
 6. The two planes are blended using the mask: foreground where the mask is 1,
    background where it is 0, a mix in between.
 7. The post chain runs on the blended frame. Post effects can read the mask but
    not change it.

Each chain is an ordered list of effects. Background and foreground chains use
the same set of effects, so a `color_fill` on the foreground hides the person
and keeps the room, and a `sharpen` on the background sharpens the room.

A preset without any sections passes the camera image through unchanged.

## Preset structure

```toml
[presets.blur.mask]
model = "./models/selfie_segmentation.onnx"
chain = ["smooth_temporal", "threshold", "largest_blob", "feather"]

[presets.blur.mask.threshold]
level = 0.8

[presets.blur.mask.feather]
radius = 4

[presets.blur.background]
chain = ["blur"]

[presets.blur.background.blur]
radius = 20
passes = 1
downscale = 1
```

Each section has a `chain` with effect names in processing order, and one table
per effect with its parameters. An effect listed in `chain` without a table uses
default values for all parameters. A table for an effect that is not in the
chain is an error.

The `mask` section has additional keys:

| Key                  | Description                                                                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`              | Path to the ONNX model. Required. The sidecar file is expected at the same path with the `.toml` extension.                                     |
| `fallback_threshold` | How many consecutive inference failures to tolerate, default 3. While the count is below the threshold, failed frames pass through unprocessed. |

Rules checked when a preset is built:

 - `background`, `foreground` and `post` require a `mask` section.
 - Effect names must exist in the corresponding group.
 - Parameter values must be within the documented ranges.
 - Files referenced by the preset (the model, `image_fill` images) must exist.

`fluxframe check --preset NAME` performs all of these checks without starting
the pipeline.

### Disabling an effect

Every effect table accepts the key `enabled`. With `enabled = false` the effect
stays in the chain with all of its parameters, but it is skipped for every
frame:

```toml
[presets.blur.background.blur]
enabled = false
radius = 20
```

A disabled effect is still loaded, so enabling it again is instant. An
`image_fill` effect with a missing file fails the preset even when disabled.
Effects that keep state between frames (`smooth_temporal`, `auto_frame`) start
from a clean state when they are enabled again. If the same effect appears twice
in a chain, `enabled` applies to both.

### Fallback preset

A preset can name another preset to run when it cannot run on this machine:

```toml
[presets.default]
fallback_preset = "mediapipe"
```

The daemon switches to the fallback preset, with that preset's own chains, in
two cases:

 - The preset's model is restricted to accelerators the machine does not have
   (the sidecar key `devices`, see [Model sidecar files](#model-sidecar-files)).
   The switch happens when the preset is loaded.
 - Inference keeps failing while the preset runs, so the mask section's
   `fallback_threshold` is exceeded. Without `fallback_preset` the daemon stops
   in this case.

The daemon logs a warning with the reason, and `daemon_info` on the [control
socket](control-socket.md) reports both the selected and the running preset.
The fallback stays active until the next `set_preset`, `reload` or restart, each
of which tries the selected preset again. A fallback preset can have its own
fallback. The target must exist, and a chain of fallbacks must not lead back to
a preset already in it; `fluxframe check` reports both errors.

## Mask effects

Mask effects operate on the mask at model resolution. They are cheap: a full
mask chain costs well under a millisecond per frame on a 256x256 mask and about
1 ms on a frame-sized 800x448 mask.

### `threshold`

Converts the mask to strictly 0 or 1.

| Parameter | Type  | Default | Range  | Description                                      |
| --------- | ----- | ------- | ------ | ------------------------------------------------ |
| `level`   | float | 0.5     | 0 to 1 | Values at or above the level become 1, others 0. |

Higher levels keep only confident person pixels and remove halos, at the cost of
losing hair and thin edges.

### `largest_blob`

Keeps only the largest connected region of the mask. It removes stray detections
such as pictures of people on the wall or objects in the background.

| Parameter | Type  | Default | Range  | Description                                             |
| --------- | ----- | ------- | ------ | ------------------------------------------------------- |
| `level`   | float | 0.5     | 0 to 1 | Pixels at or above the level count as part of a region. |

### `smooth_temporal`

Averages the mask over time (exponential moving average) to suppress flicker at
the edges.

| Parameter | Type  | Default | Range  | Description                                                            |
| --------- | ----- | ------- | ------ | ---------------------------------------------------------------------- |
| `factor`  | float | 0.65    | 0 to 1 | Weight of the previous mask. 0 disables smoothing, 1 freezes the mask. |

Higher values give a calmer edge but make the mask lag behind fast movement.
Values around 0.8 to 0.85 work well for a person sitting at a desk.

### `feather`

Softens the mask edge so the transition between person and background is
gradual.

| Parameter | Type    | Default | Range   | Description                             |
| --------- | ------- | ------- | ------- | --------------------------------------- |
| `radius`  | integer | 7       | 0 to 64 | Blur radius in mask pixels. 0 disables. |

The radius is measured at model resolution. For a 256x256 mask and a 1280x720
frame, one mask pixel is about five frame pixels wide, so small values (2 to 4)
are usually enough. With a frame-sized model such as Robust Video Matting the
radius is in frame pixels, so the same value gives a much narrower transition;
2 suits its already soft edge.

### `dilate`

Expands the person region by one pixel in every direction per pass.

| Parameter    | Type    | Default | Range   | Description                       |
| ------------ | ------- | ------- | ------- | --------------------------------- |
| `iterations` | integer | 1       | 0 to 32 | Number of 3x3 passes. 0 disables. |

### `invert`

Replaces each mask value with 1 minus the value, swapping the roles of
foreground and background. No parameters.

### `passthrough`

Leaves the mask unchanged. No parameters.

## Background and foreground effects

These effects operate on full-resolution frames, so their cost grows with the
frame size. Every effect can read the mask, but most ignore it. Effects that
use it say so in their description.

### `blur`

Box blur with optional internal downscaling. Several passes approximate a
Gaussian blur.

| Parameter   | Type    | Default | Range    | Description                                                    |
| ----------- | ------- | ------- | -------- | -------------------------------------------------------------- |
| `radius`    | integer | 20      | 0 to 256 | Blur radius in frame pixels.                                   |
| `passes`    | integer | 2       | 1 to 16  | Number of box passes.                                          |
| `downscale` | integer | 4       | 1 to 8   | Blur at 1/N of the frame size, then scale back up. 1 disables. |

`downscale` greatly reduces the cost of a large blur. With strong blur the loss
of detail is invisible, but at `downscale = 1` the result is smoother.

### `color_fill`

Replaces the plane with a solid colour.

| Parameter | Type                | Default           | Description                           |
| --------- | ------------------- | ----------------- | ------------------------------------- |
| `rgb`     | array of 3 integers | `[128, 128, 128]` | Fill colour, each component 0 to 255. |

### `image_fill`

Replaces the plane with a static image. The image is loaded once and scaled to
the frame size.

| Parameter       | Type                | Default     | Description                                                                                                                                                                      |
| --------------- | ------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `path`          | path                | required    | PNG or JPEG file.                                                                                                                                                                |
| `fit`           | string              | `"cover"`   | `cover` scales and crops to fill the frame keeping the aspect ratio. `contain` fits the whole image and fills the rest with `letterbox_rgb`. `stretch` ignores the aspect ratio. |
| `letterbox_rgb` | array of 3 integers | `[0, 0, 0]` | Colour of the bars in `contain` mode.                                                                                                                                            |

### `pixelate`

Replaces the plane with a mosaic of averaged blocks.

| Parameter    | Type    | Default | Range    | Description                  |
| ------------ | ------- | ------- | -------- | ---------------------------- |
| `block_size` | integer | 16      | 2 to 256 | Block edge length in pixels. |

### `sharpen`

Unsharp-mask sharpening. Useful on the foreground with soft laptop cameras.

| Parameter | Type  | Default | Range  | Description                                             |
| --------- | ----- | ------- | ------ | ------------------------------------------------------- |
| `amount`  | float | 0.3     | 0 to 2 | Sharpening strength. 0 disables; 0.2 to 0.5 is typical. |

Sharpening also amplifies sensor noise, so in low light use small amounts.

### `vignette`

Darkens the plane smoothly towards the corners.

| Parameter      | Type  | Default | Range     | Description                                                 |
| -------------- | ----- | ------- | --------- | ----------------------------------------------------------- |
| `strength`     | float | 0.4     | 0 to 1    | Darkening at the corners. 0 disables.                       |
| `inner_radius` | float | 0.5     | 0 to 0.99 | Normalised distance from the centre where darkening starts. |

### `exposure_correct`

Stretches the brightness histogram and applies a gamma correction towards a
target brightness. It is intended for dark or backlit faces and is normally used
on the foreground.

| Parameter           | Type  | Default | Range      | Description                              |
| ------------------- | ----- | ------- | ---------- | ---------------------------------------- |
| `target_brightness` | float | 0.55    | 0.2 to 0.9 | Target mean brightness after correction. |
| `percentile_low`    | float | 0.05    | 0 to 0.2   | Histogram percentile mapped to black.    |
| `percentile_high`   | float | 0.95    | 0.8 to 1   | Histogram percentile mapped to white.    |

When the image already uses most of the brightness range, the effect leaves it
unchanged.

### `stained_glass`

Draws the plane as stained glass: fixed, irregular cells, each filled with the
averaged colour of the area it covers, separated by thin dark lines. The room's
own colours are kept but lose all detail. The cell pattern never changes and
each cell's colour changes slowly, so movement behind you becomes a slow shift
of tones rather than flicker.

| Parameter       | Type    | Default | Range     | Description                                                    |
| --------------- | ------- | ------- | --------- | -------------------------------------------------------------- |
| `cell_size`     | integer | 48      | 8 to 256  | Mean cell size in pixels.                                      |
| `irregularity`  | float   | 0.8     | 0 to 1    | 0 gives a regular grid of squares, 1 the most irregular cells. |
| `smoothing`     | float   | 0.9     | 0 to 0.99 | Per-frame inertia of the cell colours. 0 follows the frame.    |
| `line_width`    | integer | 2       | 0 to 8    | Width of the lines between cells in pixels. 0 draws none.      |
| `line_strength` | float   | 0.6     | 0 to 1    | How much darker the lines are: 0 invisible, 1 black.           |
| `ignore_person` | boolean | true    |           | Leave the person out of the cell colours. Uses the mask.       |

The pattern is the same on every run. It is rebuilt when `cell_size`,
`irregularity`, `line_width` or the frame size changes.

With `ignore_person`, the colour of a cell is averaged over the room only, so
your clothes do not tint the glass around and behind you, and the cells do not
change colour when you move. A cell you cover almost entirely keeps the last
colour of the room it saw; one you have covered since the start shows the mean
colour of the visible room until you move aside. Set `ignore_person = false` when the effect runs on
the foreground, where the plane is the person.

### `shadow`

Casts a soft shadow of the person on the plane, shifted by an offset, as if a
lamp in front of you threw it on the wall behind. Uses the mask. On the
background the shadow shows only beside the person; the person's own pixels
come from the foreground.

| Parameter  | Type    | Default | Range       | Description                                          |
| ---------- | ------- | ------- | ----------- | ---------------------------------------------------- |
| `offset_x` | integer | 14      | -128 to 128 | Horizontal shift in pixels; positive moves it right. |
| `offset_y` | integer | 8       | -128 to 128 | Vertical shift in pixels; positive moves it down.    |
| `blur`     | integer | 12      | 0 to 64     | Softness of the shadow's edge, in pixels.            |
| `strength` | float   | 0.35    | 0 to 1      | Darkening where the shadow is full: 0 none, 1 black. |

The mask is blurred at a quarter of the frame size, so even `blur = 0` leaves
an edge a few pixels soft. Placed before `stained_glass`, the shadow is
averaged into the glass cells; placed after it, the shadow lies over the glass.

### `halo`

Lights the plane around the person and dims it further away: a vignette
centred on you rather than on the frame. Uses the mask. The halo moves only
when you move and has no edges, so it does not draw attention.

| Parameter | Type    | Default | Range    | Description                                          |
| --------- | ------- | ------- | -------- | ---------------------------------------------------- |
| `radius`  | integer | 60      | 8 to 256 | How far the halo reaches from the person, in pixels. |
| `glow`    | float   | 0.15    | 0 to 1   | Brightening next to the person: 0 none, 1 double.    |
| `dim`     | float   | 0.3     | 0 to 1   | Darkening far from the person: 0 none, 1 black.      |

With `glow = 0` the effect only darkens the room away from you.

### `color_focus`

Keeps the room's colour next to you and fades it towards grey further away.
Uses the mask. Like `halo`, it is a vignette centred on you, but it removes
colour instead of light, so the room keeps its brightness.

| Parameter | Type    | Default | Range    | Description                                     |
| --------- | ------- | ------- | -------- | ----------------------------------------------- |
| `radius`  | integer | 80      | 8 to 256 | How far the colour reaches from you, in pixels. |
| `fade`    | float   | 0.8     | 0 to 1   | Colour lost far from you: 0 none, 1 grey.       |

### `passthrough`

Leaves the plane unchanged. No parameters.

## Post effects

Post effects run on the blended frame and require a `mask` section.

### `auto_frame`

Crops and zooms onto the person, keeping them in frame as they move. It
compensates for a badly placed camera without touching it.

| Parameter    | Type  | Default | Range     | Description                                                                                         |
| ------------ | ----- | ------- | --------- | --------------------------------------------------------------------------------------------------- |
| `threshold`  | float | 0.5     | 0 to 1    | Mask confidence used to measure the person.                                                         |
| `padding`    | float | 0.15    | 0 to 1    | Margin added around the person, as a fraction of their size.                                        |
| `smoothing`  | float | 0.85    | 0 to 0.99 | Per-frame inertia of the measured position. 0 uses each frame's measurement as is.                  |
| `zoom_max`   | float | 1.6     | 1 to 4    | Maximum zoom relative to the whole captured view. 1 disables cropping.                              |
| `deadband`   | float | 0.12    | 0 to 0.5  | How far the person may drift, as a fraction of the frame, before the frame follows.                 |
| `hold_secs`  | float | 1.0     | 0 to 10   | How long the person must stay outside the dead zone before the frame follows.                       |
| `glide_secs` | float | 0.8     | 0 to 5    | Duration of a reframing move. 0 jumps.                                                              |

The frame stays still while you fidget, gesture or lean slightly. It moves only
when your position has stayed off by more than `deadband` for `hold_secs`, and
then glides there smoothly. If you are cut off at an edge of the frame, it
follows at once. If nobody is in view for about two seconds, it returns to the
whole view. Stray mask pixels and hands held to the side do not count towards
your position.

`auto_frame` does not resample the processed frame. It chooses which part of
the captured frame is processed: the crop is cut out of the camera image before
segmentation and the other effects. Combined with
[`input.capture_width` and `input.capture_height`](configuration.md#input), a
large capture keeps a zoomed-in face sharp while the effects run at the smaller
`input.width` × `input.height`. Without the capture keys the crop is enlarged
from the processing-size frame and holds no more detail than the pixels it
covers. Real detail lasts up to a zoom of `capture_width / width` (1.6 for
1280x720 processed at 800x448).

The crop applies from the next frame. With a fixed camera this one-frame delay
is not visible.

### `mirror`

Flips the frame horizontally. Conferencing applications usually mirror your
self-view. If your own preview shows text the right way round, other
participants see it mirrored, and this effect corrects that. No parameters.

### `passthrough`

Leaves the frame unchanged. No parameters.

## Example presets

Soft blurred background:

```toml
[presets.blur.mask]
model = "./models/selfie_segmentation.onnx"
chain = ["smooth_temporal", "threshold", "largest_blob", "feather"]
[presets.blur.mask.smooth_temporal]
factor = 0.85
[presets.blur.mask.threshold]
level = 0.8
[presets.blur.mask.feather]
radius = 4

[presets.blur.background]
chain = ["blur"]
[presets.blur.background.blur]
radius = 20
```

Solid colour background, suitable for chroma keying in OBS:

```toml
[presets.green.mask]
model = "./models/selfie_segmentation.onnx"
chain = ["smooth_temporal", "threshold", "largest_blob"]
[presets.green.mask.threshold]
level = 0.8

[presets.green.background]
chain = ["color_fill"]
[presets.green.background.color_fill]
rgb = [0, 255, 0]
```

Image background with a corrected, sharpened speaker:

```toml
[presets.office.mask]
model = "./models/selfie_segmentation.onnx"
chain = ["smooth_temporal", "threshold", "largest_blob", "feather"]
[presets.office.mask.threshold]
level = 0.8
[presets.office.mask.feather]
radius = 3

[presets.office.background]
chain = ["image_fill", "vignette"]
[presets.office.background.image_fill]
path = "/home/user/Pictures/office.jpg"
fit = "cover"
[presets.office.background.vignette]
strength = 0.3

[presets.office.foreground]
chain = ["exposure_correct", "sharpen"]
[presets.office.foreground.sharpen]
amount = 0.3
```

Automatic framing over a blurred background, mirrored for the audience:

```toml
[presets.framed.mask]
model = "./models/selfie_segmentation.onnx"
chain = ["smooth_temporal", "threshold", "largest_blob", "feather"]
[presets.framed.mask.threshold]
level = 0.8
[presets.framed.mask.feather]
radius = 3

[presets.framed.background]
chain = ["blur"]

[presets.framed.post]
chain = ["auto_frame", "mirror"]
[presets.framed.post.auto_frame]
smoothing = 0.95
zoom_max = 1.3
```

Stained-glass background that keeps the colours of the room:

```toml
[presets.glass.mask]
model = "./models/selfie_segmentation.onnx"
chain = ["smooth_temporal", "threshold", "largest_blob", "feather"]
[presets.glass.mask.threshold]
level = 0.8
[presets.glass.mask.feather]
radius = 3

[presets.glass.background]
chain = ["stained_glass"]
[presets.glass.background.stained_glass]
cell_size = 48
line_strength = 0.6
```

The room stays sharp but steps back: colour and light fade away from you and
you cast a soft shadow on it:

```toml
[presets.spotlight.mask]
model = "./models/selfie_segmentation.onnx"
chain = ["smooth_temporal", "threshold", "largest_blob", "feather"]
[presets.spotlight.mask.threshold]
level = 0.8
[presets.spotlight.mask.feather]
radius = 3

[presets.spotlight.background]
chain = ["color_focus", "halo", "shadow"]
[presets.spotlight.background.color_focus]
fade = 0.7
```

Robust Video Matting on the NPU over a blurred background, with MediaPipe as
the fallback on machines without an NPU (see
[Installation](installation.md#robust-video-matting) for the model):

```toml
[presets.default]
fallback_preset = "mediapipe"
[presets.default.mask]
model = "./models/rvm_mobilenetv3_d0.4.onnx"
chain = ["largest_blob", "feather"]
[presets.default.mask.feather]
radius = 2
[presets.default.background]
chain = ["blur"]

[presets.mediapipe.mask]
model = "./models/selfie_segmentation.onnx"
chain = ["smooth_temporal", "threshold", "largest_blob", "feather"]
[presets.mediapipe.mask.threshold]
level = 0.8
[presets.mediapipe.mask.feather]
radius = 3
[presets.mediapipe.background]
chain = ["blur"]
```

Robust Video Matting returns a soft alpha matte that is already stable over
time, so its mask chain differs from the MediaPipe one:

 - No `threshold`: it would turn the soft edge, including hair, into a hard
   one.
 - No `smooth_temporal`: the model carries state between frames and does not
   flicker; extra smoothing only adds lag.
 - `largest_blob` stays: in recorded test clips it removed detached fragments,
   such as parts of a chair, in 6-8% of frames. It also clears faint
   translucent areas away from the person.

Camera image without processing:

```toml
[presets.raw]
```

## Model sidecar files

Every model is described by a sidecar file next to it: `model.onnx` needs
`model.toml`. The sidecar tells FluxFrame how to prepare the input tensor and
how to interpret the output, which makes it possible to use other single-person
segmentation models with a compatible structure.

| Key                           | Default   | Description                                                                                                                             |
| ----------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `name`                        | required  | Model name shown in logs.                                                                                                               |
| `input_size`                  | `"fixed"` | `fixed`: the model takes `input_width` x `input_height`. `frame`: the model takes the processed frame at its own size, without scaling. |
| `input_width`, `input_height` | required  | Input tensor size in pixels. Required with `input_size = "fixed"`, not allowed with `"frame"`.                                          |
| `input_layout`                | `"NHWC"`  | `NHWC` or `NCHW`.                                                                                                                       |
| `input_dtype`                 | `"f32"`   | Only `f32` is supported.                                                                                                                |
| `input_color`                 | `"RGB"`   | Channel order of the input.                                                                                                             |
| `input_scale`                 | 1/255     | Each pixel value is multiplied by this factor.                                                                                          |
| `input_zero_point`            | 0.0       | Subtracted from each pixel value before scaling.                                                                                        |
| `output_index`                | 0         | Which model output holds the mask.                                                                                                      |
| `output_layout`               | `"HW"`    | `HW`, `NHWC` or `NCHW`.                                                                                                                 |
| `output_type`                 | `"mask"`  | `mask`, `probabilities`, `logits` or `category_mask`.                                                                                   |
| `person_class_index`          | none      | Class id of a person; required for `category_mask`.                                                                                     |
| `threshold`                   | none      | Mask threshold, 0 to 1.                                                                                                                 |
| `state`                       | none      | Recurrent state: a list of `{ output = "...", input = "..." }` pairs. Each output is fed back into the paired input on the next frame.  |
| `pipeline`                    | `false`   | Run inference one frame ahead of the rest of the processing.                                                                            |
| `devices`                     | none      | OpenVINO devices the model may run on, in order of preference.                                                                          |

Unknown keys are rejected.

`state`, `pipeline` and `devices` exist for models such as Robust Video Matting
that are too slow to run within the frame time and must run on an accelerator:

 - **`state`.** The model's state starts at zero and is reset whenever the
   sequence of frames breaks: on a preset switch, on entering idle mode, after
   an inference failure, and when `auto_frame` jumps to a distant area of the
   capture. Models with state require OpenVINO; ONNX Runtime rejects them.
 - **`pipeline`.** While the model processes frame N, the rest of the chain
   processes frame N-1 with its own mask, so the mask always matches its frame.
   The video is delayed by one frame (33 ms at 30 fps). The first frame after
   the start or a reset is not published.
 - **`devices`.** When set, the model runs only on the listed devices, tried in
   order. It overrides the global device choice and
   `FLUXFRAME_OPENVINO_DEVICE`. If none of them is available, or OpenVINO
   cannot be used at all, the preset cannot run and its `fallback_preset`
   takes over. The sidecar for MediaPipe Selfie Segmentation is
shown in [Installation](installation.md#segmentation-model).
