# Presets and effects

A preset is a named description of how each frame is processed. The
configuration can hold any number of presets. One of them is active at a time,
and it can be switched while the daemon runs.

## How a frame is processed

A preset has up to four sections: `mask`, `background`, `foreground` and `post`.
When the `mask` section is present, every frame goes through these steps:

 1. The segmentation model produces a mask at the model's resolution (256x256
    for the Selfie Segmentation model). Each mask value is the confidence that
    the pixel belongs to a person.
 2. The mask chain refines the mask at model resolution: thresholding, removing
    stray regions, smoothing over time, softening the edge.
 3. The mask is scaled up to the frame size.
 4. The background chain runs on a copy of the frame.
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

## Mask effects

Mask effects operate on the mask at model resolution. They are cheap: a full
mask chain costs well under a millisecond per frame.

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
are usually enough.

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
frame size.

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

### `passthrough`

Leaves the plane unchanged. No parameters.

## Post effects

Post effects run on the blended frame and require a `mask` section.

### `auto_frame`

Crops and zooms onto the person, keeping them centred as they move. It
compensates for a badly placed camera without touching it.

| Parameter   | Type  | Default | Range     | Description                                                                                              |
| ----------- | ----- | ------- | --------- | -------------------------------------------------------------------------------------------------------- |
| `threshold` | float | 0.5     | 0 to 1    | Mask confidence used to find the person's bounding box.                                                  |
| `padding`   | float | 0.15    | 0 to 1    | Margin added around the bounding box, as a fraction of its size.                                         |
| `smoothing` | float | 0.85    | 0 to 0.99 | Inertia of the crop window. 0 follows instantly; higher values move the frame slowly and without jitter. |
| `zoom_max`  | float | 1.6     | 1 to 4    | Maximum zoom. 1 disables cropping.                                                                       |

The crop is scaled back to the full output size on every frame, which makes
`auto_frame` one of the more expensive effects. Its cost does not depend on the
zoom level.

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

Camera image without processing:

```toml
[presets.raw]
```

## Model sidecar files

Every model is described by a sidecar file next to it: `model.onnx` needs
`model.toml`. The sidecar tells FluxFrame how to prepare the input tensor and
how to interpret the output, which makes it possible to use other single-person
segmentation models with a compatible structure.

| Key                           | Default  | Description                                           |
| ----------------------------- | -------- | ----------------------------------------------------- |
| `name`                        | required | Model name shown in logs.                             |
| `input_width`, `input_height` | required | Input tensor size in pixels.                          |
| `input_layout`                | `"NHWC"` | `NHWC` or `NCHW`.                                     |
| `input_dtype`                 | `"f32"`  | `f32`, `u8` or `i8`.                                  |
| `input_color`                 | `"RGB"`  | Channel order of the input.                           |
| `input_scale`                 | 1/255    | Each pixel value is multiplied by this factor.        |
| `input_zero_point`            | 0.0      | Subtracted from each pixel value before scaling.      |
| `output_index`                | 0        | Which model output holds the mask.                    |
| `output_layout`               | `"HW"`   | `HW`, `NHWC` or `NCHW`.                               |
| `output_type`                 | `"mask"` | `mask`, `probabilities`, `logits` or `category_mask`. |
| `person_class_index`          | none     | Class id of a person; required for `category_mask`.   |
| `threshold`                   | none     | Mask threshold, 0 to 1.                               |

Unknown keys are rejected. The sidecar for MediaPipe Selfie Segmentation is
shown in [Installation](installation.md#segmentation-model).
