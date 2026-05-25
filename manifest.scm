;;; FluxFrame development environment manifest.
;;;
;;; Usage:
;;;   guix shell -m manifest.scm
;;;
;;; Provides the system-level toolchain and libraries required to build
;;; FluxFrame (Rust + GStreamer + ONNX Runtime).  Models, the v4l2loopback
;;; kernel module, and the pre-commit framework are intentionally NOT
;;; managed here — see README.md for the corresponding setup steps.
;;;
;;; The `openvino` and `onnxruntime` here come from the user's personal
;;; `johnlepikhin` channel (see ~/.config/guix/channels.scm), not from
;;; upstream Guix.  Picking them up just requires having run `guix pull`
;;; with the channel enabled — `specifications->manifest` resolves
;;; package names against the current Guix tree, which includes
;;; channels.

(specifications->manifest
 '("rust"
   "rust:cargo"
   "pkg-config"
   "gstreamer"
   "gst-plugins-base"
   "gst-plugins-good"
   "glib"
   ;; ONNX Runtime 1.26 (with the mp11/pybind11 fixes from the
   ;; johnlepikhin channel).  Use "onnxruntime-openvino" instead if
   ;; you want the OpenVINO execution provider compiled in — that
   ;; variant has a heavier build but lets `ort` dispatch to the
   ;; OpenVINO runtime via `load-dynamic` without an extra crate.
   "onnxruntime"
   ;; OpenVINO + Intel GPU/NPU user-mode drivers come from the
   ;; user's Guix Home (johnlepikhin channel) — packages
   ;; `openvino-full`, `intel-compute-runtime`, `intel-npu-driver`
   ;; live there.  They are NOT listed here because the dev `guix
   ;; shell` runs against the system Guix tree (which lacks those
   ;; packages until the operator runs `guix pull`).  Home Manager
   ;; exports `OPENVINO_INSTALL_DIR`, `OCL_ICD_VENDORS` and the
   ;; library load paths via `~/.profile`, so any bash login shell
   ;; — including a `guix shell` child — picks them up.  The build
   ;; binds against ONNX Runtime above; OpenVINO is loaded at
   ;; runtime via openvino-rs `runtime-linking` feature.
   ;; Vulkan runtime stack (Stage 7).  Mesa supplies the Intel ICD on
   ;; the host; in `guix shell` we also need the loader and headers
   ;; so `wgpu`'s runtime can locate the ICD via `VK_ICD_FILENAMES`
   ;; / `/etc/vulkan/icd.d/`.  Without these, the wgpu adapter probe
   ;; returns Err and the blur factory cleanly falls back to CPU —
   ;; but inside guix shell we want the GPU path to actually work.
   "vulkan-loader"
   "vulkan-headers"
   ;; GCC C++ runtime: previously needed because pip-installed
   ;; OpenVINO shipped manylinux2014 .so's depending on libstdc++.
   ;; Keep around because GStreamer plugins built from source still
   ;; want libstdc++ available.
   "gcc-toolchain"
   "nss-certs"
   "coreutils"))
