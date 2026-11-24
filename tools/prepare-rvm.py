#!/usr/bin/env python3
"""Prepare Robust Video Matting (RVM) for FluxFrame.

Robust Video Matting (https://github.com/PeterL1n/RobustVideoMatting) is a
recurrent matting model: it carries state from frame to frame and outputs a
soft alpha matte at the full frame size. FluxFrame runs it on an Intel NPU
through OpenVINO. The weights are licensed under GPL-3.0 and are not shipped
with FluxFrame; this script downloads them (or takes a local copy) and turns
the release file into a model FluxFrame can run:

* checks the file against the SHA-256 of the v1.0.0 release;
* freezes the `downsample_ratio` input into a constant: the NPU compiles the
  model only when that value is fixed;
* removes the `fgr` (foreground colour) output, which FluxFrame does not use;
  the NPU then skips computing it;
* writes the model and its `.toml` sidecar.

Requirements: Python 3.9+ and the `onnx` package (`pip install onnx`).

Usage:
    tools/prepare-rvm.py [--source FILE] [--ratio 0.4] [--output-dir DIR]

The result is `DIR/rvm_mobilenetv3_d<ratio>.onnx` plus its sidecar. Point a
preset's `[mask] model` at it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

RELEASE_URL = (
    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/"
    "rvm_mobilenetv3_fp32.onnx"
)
RELEASE_SHA256 = "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828"
DEFAULT_RATIO = 0.4
STATE = [(f"r{k}o", f"r{k}i") for k in range(1, 5)]

SIDECAR = """\
# Robust Video Matting, MobileNetV3, downsample_ratio = {ratio}.
# Prepared by tools/prepare-rvm.py from {source_name}
# (sha256 {sha256}). Weights: GPL-3.0, https://github.com/PeterL1n/RobustVideoMatting
name = "rvm_mobilenetv3_d{ratio}"
# The model takes the processed frame as is and returns the alpha matte at
# the same size.
input_size = "frame"
input_layout = "NCHW"
output_layout = "NCHW"
output_type = "mask"
# Output 0 is `pha`, the alpha matte.
output_index = 0
# Inference runs one frame ahead of the rest of the pipeline; the video is
# delayed by one frame.
pipeline = true
# NPU only: on the CPU the model is too slow for 30 fps. Without an NPU the
# preset's `fallback_preset` runs instead.
devices = {devices}
# Recurrent state carried from one frame to the next.
state = [
{state}
]
"""


def default_output_dir() -> Path:
    config = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(config) / "fluxframe" / "models"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dst: Path) -> None:
    print(f"downloading {url}", file=sys.stderr)
    request = urllib.request.Request(url, headers={"User-Agent": "fluxframe-prepare-rvm"})
    with urllib.request.urlopen(request, timeout=60) as response, dst.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)


def prepare(source: Path, ratio: float, dst: Path):
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(source))
    graph = model.graph

    ratio_input = next((i for i in graph.input if i.name == "downsample_ratio"), None)
    if ratio_input is None:
        raise SystemExit("the model has no `downsample_ratio` input; is it an RVM release file?")
    graph.input.remove(ratio_input)
    graph.initializer.append(
        numpy_helper.from_array(np.array([ratio], dtype=np.float32), "downsample_ratio")
    )

    keep = [o for o in graph.output if o.name != "fgr"]
    if [o.name for o in keep][:1] != ["pha"]:
        raise SystemExit(f"unexpected outputs {[o.name for o in graph.output]}; expected pha first")
    del graph.output[:]
    graph.output.extend(keep)

    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source",
        type=Path,
        help="local rvm_mobilenetv3_fp32.onnx from the v1.0.0 release (default: download it)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help="downsample_ratio: the share of the frame size RVM's backbone works at. "
        "0.4 suits 800x448; aim for 256-512 px on the long side "
        f"(default {DEFAULT_RATIO})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="where to write the model and sidecar (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        action="append",
        dest="devices",
        help="OpenVINO device the model may run on; repeat for several (default: NPU)",
    )
    args = parser.parse_args()
    if not 0.0 < args.ratio <= 1.0:
        parser.error("--ratio must be in (0, 1]")
    devices = args.devices or ["NPU"]

    with tempfile.TemporaryDirectory() as tmp:
        source = args.source
        if source is None:
            source = Path(tmp) / "rvm_mobilenetv3_fp32.onnx"
            download(RELEASE_URL, source)
        actual = sha256_of(source)
        if actual != RELEASE_SHA256:
            print(
                f"error: {source} has sha256 {actual}, expected {RELEASE_SHA256} "
                "(rvm_mobilenetv3_fp32.onnx from the v1.0.0 release)",
                file=sys.stderr,
            )
            return 1

        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"rvm_mobilenetv3_d{args.ratio:g}"
        model_path = args.output_dir / f"{stem}.onnx"
        prepare(source, args.ratio, model_path)

    sidecar = SIDECAR.format(
        ratio=f"{args.ratio:g}",
        source_name=RELEASE_URL.rsplit("/", 1)[-1],
        sha256=RELEASE_SHA256,
        devices="[" + ", ".join(f'"{d}"' for d in devices) + "]",
        state="\n".join(f'  {{ output = "{o}", input = "{i}" }},' for o, i in STATE),
    )
    sidecar_path = model_path.with_suffix(".toml")
    sidecar_path.write_text(sidecar)
    print(f"wrote {model_path}\nwrote {sidecar_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
