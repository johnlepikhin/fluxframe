# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Stage 0 (core scaffolding) complete: Cargo workspace, core types and traits, CLI skeleton with stubbed subcommands. Pipeline does not yet process frames — that lands in Stage 1+. Implementation plan: `doc/plan/000-overview.md` (overview) and `doc/plan/stage-*-*.md` (per-stage detail). Original spec: `doc/ideas/001-mvp.md`.

## Workspace layout

Cargo workspace with four crates under `crates/`:

- `fluxframe-core` — `VideoFrame`, traits (`VideoEffect`, `InferenceEngine`, `VideoSource`, `VideoSink`), error model, `FluxConfig`. Zero GStreamer/ONNX deps; `#![forbid(unsafe_code)]`.
- `fluxframe-gst` — GStreamer init + (Stage 1+) pipeline construction, V4L2 enumeration.
- `fluxframe-effects` — effect registry, chain, built-in effects. ML and image-processing helpers live as modules here (no separate crates until a second consumer appears).
- `fluxframe-cli` — `fluxframe` binary: clap subcommands, tracing init, config loader, CLI/file merge.

Spec §7 prescribes 9 crates; the 4-crate split is a deliberate scope decision documented in `doc/plan/000-overview.md`.

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
