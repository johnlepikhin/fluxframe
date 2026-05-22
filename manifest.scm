;;; FluxFrame development environment manifest.
;;;
;;; Usage:
;;;   guix shell -m manifest.scm
;;;
;;; Provides the system-level toolchain and libraries required to build
;;; FluxFrame (Rust + GStreamer + ONNX Runtime).  Models, the v4l2loopback
;;; kernel module, and the pre-commit framework are intentionally NOT
;;; managed here — see README.md for the corresponding setup steps.

(specifications->manifest
 '("rust"
   "rust:cargo"
   "pkg-config"
   "gstreamer"
   "gst-plugins-base"
   "gst-plugins-good"
   "glib"
   "onnxruntime"
   "nss-certs"
   "coreutils"))
