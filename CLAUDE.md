# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Stages 0–15 complete, plus GUI-initiated TOML persistence on top.
End-to-end: V4L2 capture (or `videotestsrc`) → composite (mask + bg
+ fg sub-chains) → optional `[post]` chain → v4l2loopback /
autovideosink / fakesink / pipewiresink. ONNX Runtime + OpenVINO
inference backends, `wgpu` blur seam, named presets, UNIX control
socket with live `set` / `set_chain` / `set_preset` / `reload` and
write-back `save_preset` / `save_preset_as` / `config_path`.
`fluxframe-gui` (GTK4 / libadwaita) is the slider-style editor with
embedded preview pane, dirty marker and explicit Save / Save as /
Revert. Idle mode drops the camera + publishes a placeholder when
no consumer is reading from `/dev/video10`.

Per-stage detail: `doc/plan/000-overview.md` and `doc/plan/stage-*.md`.
Original spec: `doc/ideas/001-mvp.md`. README.md is the operator-
facing summary.

## Workspace layout

Cargo workspace with five crates under `crates/`:

- `fluxframe-core` — `VideoFrame`, traits (`VideoEffect`, `InferenceEngine`, `VideoSource`, `VideoSink`), error model, `FluxConfig`, control-socket wire types (`protocol::Command` / `Response`). Zero GStreamer/ONNX deps; `#![forbid(unsafe_code)]`.
- `fluxframe-gst` — GStreamer init plus `bus`/`input`/`output`/`slot`/`frame_conv`/`util`/`v4l2_caps` modules. Bus events arrive as a typed `BusEvent` enum via `BusListener::spawn`. Owns the V4L2 capture + v4l2loopback / pipewiresink output paths.
- `fluxframe-effects` — effect registry, chain, composite (mask + plane + post sub-chains), built-in effects (ML segmentation, blur, color_fill, image_fill, pixelate, sharpen, vignette, exposure_correct, auto_frame, mirror, …). ML, GPU-blur and image-processing helpers live as modules here.
- `fluxframe-cli` — `fluxframe` binary: clap subcommands (`list`, `check`, `run`, `benchmark`), tracing init, config loader (`config_merge`: XDG default lookup + `--no-default-config`), supervisor (`runtime`), control listener (`control`), TOML write-back (`persist`), idle-mode supervisor (`idle`).
- `fluxframe-gui` — `fluxframe-gui` binary: relm4 + GTK4 / libadwaita client over the UNIX control socket. Embedded live preview, chain editor, dirty tracking, Save / Save as / Revert.

Spec §7 prescribes 9 crates; the 5-crate split is a deliberate scope decision documented in `doc/plan/000-overview.md` — config / metrics live inside `fluxframe-cli`, v4l2 helpers inside `fluxframe-gst`, ML / processing inside `fluxframe-effects`.

## Dev environment (Guix)

Two supported setups (see `README.md` for the full instructions):

1. **Dev shell:** `guix shell -m manifest.scm` then `cargo …`.
2. **Persistent install:** `guix install gstreamer gst-plugins-base gst-plugins-good glib`, then extend `PKG_CONFIG_PATH` so it covers `~/.guix-profile/lib/pkgconfig` in addition to whatever `guix home` already sets.

`pre-commit` (Python) and the `v4l2loopback` kernel module are NOT managed by `manifest.scm`. `onnxruntime` lives in Guix (`onnxruntime` package); the Rust `ort` crate links against it dynamically (Stage 3+).

## Priorities

Infrastructure work takes hard priority over feature work in this project. "Infrastructure" means: the core/kernel of the system, DRY, and clean-code hygiene (naming, decomposition, removing duplication, fixing leaky abstractions).

Apply this as follows:
- When planning, start with infrastructure tasks and only then move to features.
- If, while working on a feature, you notice infrastructure debt (DRY violation, mess in the core, abstraction smell), stop, surface it to the user, and switch to fixing it before continuing the feature.
- Do not bury infra fixes inside a feature commit along the way — call them out explicitly so the user can decide on scope.

## Commands

- Build: `cargo build` (release: `cargo build --release`)
- Run: `cargo run`
- Test: `cargo test` (single test: `cargo test <name>`; show output: `cargo test -- --nocapture`)
- Lint: `cargo clippy --all-targets -- -D warnings`
- Format: `cargo fmt` (check only: `cargo fmt --check`)
- Check without building: `cargo check`

## Notes

- Rust edition is `2024` in `Cargo.toml` — keep this in mind when suggesting syntax/idioms; some patterns differ from 2021.
