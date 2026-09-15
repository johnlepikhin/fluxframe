# Performance

A virtual camera runs during every call, often on battery, next to a browser
that is itself encoding video. FluxFrame is therefore tuned for low CPU time per
frame rather than for the lowest possible latency. This page shows where the
time goes, reference measurements, the settings that matter, and how to measure
your own setup.

## Where the time goes

For each frame, the work is split between:

 - **Capture and output conversion.** GStreamer converts the camera format
   (usually YUYV) to RGB and, depending on `output.format`, back to the output
   format.
 - **Inference.** The segmentation model runs at 256x256, independent of the
   camera resolution. On an NPU this takes a few milliseconds of wall time but
   very little CPU: the processing thread mostly waits for the accelerator.
 - **Image processing.** Scaling the mask to frame size, the background and
   foreground effects, blending, and post effects. All of these touch every
   pixel of the frame. They run in parallel on a small pool of worker threads,
   and their cost grows linearly with the frame area.

With inference on an accelerator, image processing is the largest part of CPU
usage. The image kernels are limited by memory throughput rather than by
computation, so adding threads reduces latency only slightly while increasing
CPU time noticeably. For that reason the default pool has two threads.

## Reference measurements

Test system:

 - Intel Core Ultra 7 155H laptop with the NPU 3720.
 - USB camera delivering YUYV at 800x448.
 - MediaPipe Selfie Segmentation 256x256 on OpenVINO with the device
   `AUTO:NPU,GPU,CPU`, which selected the NPU.

CPU usage is given as a percentage of one core. Measurements on a laptop vary by
15-25% between runs, so small differences are not significant.

### During a real call

Configuration:

 - Frame rate about 25 fps, as delivered by the camera during the call.
 - Preset: temporal smoothing, threshold, largest region and feather on the
   mask; blur and vignette on the background; sharpening on the foreground;
   pixelation; automatic framing and mirroring.

| Configuration                                  | Process CPU | Processing time per frame (median) |
| ---------------------------------------------- | ----------- | ---------------------------------- |
| Inference on the CPU (OpenVINO CPU plugin)     | about 130%  |                                    |
| Inference on the NPU, before optimisation      | 70-75%      | 13.8 ms                            |
| Inference on the NPU, first optimisation pass  | about 53%   | 11.7 ms                            |
| Inference on the NPU, second optimisation pass | 31-32%      | 9.3 ms                             |

In the last configuration the median time from capture to output was 9.4 ms. CPU
time was distributed as follows: about 21% in the image-processing threads, 4%
in capture conversion, 3% in output conversion and 3.4% in the main processing
thread, which includes waiting for the NPU.

The optimisation passes did not change the output image. They introduced:

 - Integer fixed-point arithmetic for blending, vignetting and scaling.
 - Precomputed interpolation weights for mask scaling.
 - A sliding-window implementation of the feather blur.
 - Run-length connected-component labelling for `largest_blob`.
 - A cheaper sharpening kernel.
 - Row-parallel mirroring.
 - A faster bounding-box search for automatic framing.

### Thread count

Synthetic source at 800x448 with the output discarded
(`--input testsrc --output fakesink`), same preset:

| Processing threads | Process CPU | Image-processing threads | Processing time (median) |
| ------------------ | ----------- | ------------------------ | ------------------------ |
| 4                  | 32.2%       | 21.2%                    | 8.8 ms                   |
| 3                  | 28.8%       | 17.8%                    | 9.1 ms                   |
| 2                  | 27.6%       | 17.2%                    | 9.7 ms                   |

Two threads save about 15% of the CPU time of four threads at a cost of under
one millisecond per frame. That is well within the 33-40 ms available per frame
at 25-30 fps.

### Output format

Same synthetic setup, two processing threads:

| `output.format` | Process CPU | Output thread |
| --------------- | ----------- | ------------- |
| `YUY2`          | 27.4%       | 4.2%          |
| `RGB`           | 25.6%       | 0.8%          |

Besides being cheaper, RGB output avoids faint vertical colour bands that the
RGB to YUY2 conversion introduces.

### Idle

With idle mode enabled and no application reading the virtual camera, the daemon
uses less than 1% of a core.

### GPU blur

The optional Vulkan blur backend was measured on the same class of integrated
GPU at 640x480 with `downscale = 4`. The median processing time rose from 10 ms
to 30-42 ms and the frame rate dropped by 20%, because moving frames to and from
the GPU costs more than the blur itself and the GPU is shared with inference.
The CPU backend is therefore the default.

## Tuning

In order of impact:

 1. **Run inference on an accelerator.** On Intel laptops with an NPU, install
    OpenVINO and the NPU driver. On the test system this alone took CPU usage
    from about 130% to 70%. Check the startup log to see which backend was
    selected. `FLUXFRAME_OPENVINO_DEVICE` selects a specific device.
 2. **Choose the capture resolution deliberately.** The cost of image processing
    is proportional to the number of pixels. The model works at 256x256
    regardless of the camera, so the mask does not improve with a larger frame.
    800x448 or 960x540 is enough for most calls, and many conferencing services
    downscale further anyway. `output.scale` reduces only the output size, after
    processing. To reduce processing cost, lower `input.width` and
    `input.height` instead.
 3. **Use `output.format = "RGB"`.**
 4. **Keep the default thread count.** Set `realtime.processing_threads` (or
    `FLUXFRAME_RAYON_THREADS`) higher only when latency matters more than CPU
    time, for example on a desktop machine.
 5. **Mind the expensive effects.**
    
     - `auto_frame` scales the whole frame on every frame and was the most
       expensive single effect in the call measurement, at about 1.1 ms.
     - `blur` with `downscale = 1` and a large radius and `sharpen` are the next
       most expensive.
     - `color_fill`, `image_fill` and all mask effects are cheap.
     - Disabled effects (`enabled = false`) cost nothing.
 6. **Enable idle mode** if the daemon runs permanently.

## Reading the metrics

The daemon logs a metrics line every `realtime.metrics_interval_secs` seconds
(default 30), and a final summary when it stops. Lines that would repeat the
previous one exactly are suppressed while nothing is happening. For
measurements, set the interval to 5 seconds.

The `metrics tick` line reports the frame rate and latency percentiles in
microseconds:

 - `fps`, `frames_out_total`, `frames_dropped_total`.
 - `capture_p50_us`, `inference_p50_us`, `processing_p50_us`, `output_p50_us`,
   `end_to_end_p50_us`, and the corresponding `_p95_us` values.
 - `processing_threads`: the size of the processing pool.

The `stage timings` line breaks processing down by step, as the median and 95th
percentile in microseconds. For example:

```
stages="seg/preprocess=488/610 mask/feather=128/170 background/blur=903/1200 composite/compose=424/560 post/auto_frame=1100/1400"
```

Keys have the form `section/effect`. `seg/*` keys are segmentation steps and
`composite/*` keys are mask scaling and blending. Disabled effects are not
listed.

The `cpu shares` line reports CPU usage by thread role since the previous line,
as a percentage of one core:

| Field             | Threads                                          |
| ----------------- | ------------------------------------------------ |
| `cpu_process_pct` | The whole process.                               |
| `cpu_worker_pct`  | The main processing thread, including inference. |
| `cpu_rayon_pct`   | The image-processing pool.                       |
| `cpu_input_pct`   | Capture and input conversion.                    |
| `cpu_output_pct`  | Output conversion and writing.                   |
| `cpu_other_pct`   | Everything else.                                 |

`cpu_process_pct` is the number that determines battery impact. Latency
percentiles alone can be misleading, because a faster frame can cost more CPU
time when more threads are involved.

### Comparing configurations

Run-to-run variation on laptops is large: 15-25% between identical runs is
normal, because of thermal state and frequency scaling. To compare two
configurations reliably:

 1. Use the synthetic source and discard the output, so the camera and consumers
    do not influence the result:
    `fluxframe run --input testsrc --output fakesink --no-default-config --config bench.toml`,
    with `[control] enabled = false`, `[idle] enabled = false` and
    `metrics_interval_secs = 5`.
 2. Alternate the two configurations several times (for example, three rounds of
    15-20 seconds each) instead of running each once.
 3. Compare the medians of `cpu_process_pct` and of the relevant `stage timings`
    keys.

Effects that depend on the image content, such as `auto_frame`, behave
differently on the synthetic pattern and should be verified with a real camera.
