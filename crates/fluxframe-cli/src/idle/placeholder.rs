//! Idle-mode placeholder frame source.
//!
//! When the supervisor is in `Idle` or `DeepIdle` it stops pulling
//! real frames from the input pipeline and instead pushes a cheap
//! pre-rendered placeholder to the output sink at `idle.fps`. Step 4
//! caches the converted bytes as `Arc<[u8]>` so the per-tick cost
//! collapses to one `Arc` clone plus an enqueue into the writer slot.
//!
//! Two concrete kinds are supported in Stage 15:
//!
//! * [`ColorPlaceholder`] — solid RGB fill at the configured
//!   `placeholder_rgb`. Always available.
//! * [`ImagePlaceholder`] — static image loaded from a file path
//!   on disk. Decoded via the `image` crate (PNG, JPEG) at build
//!   time, then resize-stretched into the supervisor's output
//!   dimensions.
//!
//! Both impls cache an RGB byte buffer matching the supervisor's
//! output `(width, height)`. `Placeholder::render` performs the
//! per-call format conversion only. The supervisor (Step 4) wraps
//! the resulting bytes in `FrameBuffer::Shared(Arc<[u8]>)` so steady-
//! state idle pushes are zero-copy.
//!
//! Format coverage matches the small set that shows up as
//! `input.format` (the appsrc format) in practice: RGB, RGBA, BGR,
//! GRAY8, YUY2. NV12 is intentionally not supported — planar
//! two-plane formats need a `Stride::Planar` `VideoFrame`, which the
//! placeholder cache would have to construct differently. NV12 is
//! never the appsrc format in current configs (it is only a sink
//! format, where videoconvert handles it). If a future config needs
//! it, extend [`Placeholder::render`] then.

use fluxframe_core::frame::{FrameBuffer, FrameMeta, PixelFormat, VideoFrame};
use fluxframe_core::{FluxError, IdleConfig, IdlePlaceholderKind};
#[cfg(feature = "image-fill")]
use fluxframe_effects::processing::image_loader::{ImageLoadError, decode_rgb_bounded};

// ---------------------------------------------------------------------------
// BT.601 colour-space constants (Q8 fixed point).
//
// Source: ITU-R BT.601-7 §2.5.1 + JFIF/JPEG full-range Y'CbCr matrix.
// All coefficients scaled by 256 (Q8), so `(x * K + ROUND) >> 8`
// computes round-to-nearest division by 256.
// ---------------------------------------------------------------------------

/// Luma Kr = 0.299 scaled by 256.
const BT601_KR_Q8: i32 = 77;
/// Luma Kg = 0.587 scaled by 256.
const BT601_KG_Q8: i32 = 150;
/// Luma Kb = 0.114 scaled by 256.
const BT601_KB_Q8: i32 = 29;
/// Cb coefficient for R (-0.169 × 256 ≈ -43).
const BT601_CB_R_Q8: i32 = -43;
/// Cb coefficient for G (-0.331 × 256 ≈ -85).
const BT601_CB_G_Q8: i32 = -85;
/// Cb coefficient for B (0.500 × 256 = 128).
const BT601_CB_B_Q8: i32 = 128;
/// Cr coefficient for R (0.500 × 256 = 128).
const BT601_CR_R_Q8: i32 = 128;
/// Cr coefficient for G (-0.419 × 256 ≈ -107).
const BT601_CR_G_Q8: i32 = -107;
/// Cr coefficient for B (-0.081 × 256 ≈ -21).
const BT601_CR_B_Q8: i32 = -21;
/// Round-to-nearest bias for the Q8 reduction: `(x + 128) >> 8`.
const BT601_ROUND_Q8: i32 = 128;
/// Chroma 0-centre offset: signed `(-128..=127)` → unsigned
/// `(0..=255)`.
const BT601_CHROMA_BIAS: i32 = 128;

// ---------------------------------------------------------------------------
// Trait + builder
// ---------------------------------------------------------------------------

/// One static placeholder frame produced on demand. Impls are
/// `Send + Sync` so the supervisor can share the trait object
/// behind an `Arc` across the worker thread and any future
/// reload-coordinator thread without locking.
pub(crate) trait Placeholder: Send + Sync {
    /// Produce a [`VideoFrame`] in `format` at the dimensions the
    /// placeholder was built for. Each call performs a format
    /// conversion from the cached RGB buffer; the supervisor (Step 4)
    /// caches the resulting bytes as `FrameBuffer::Shared(Arc<[u8]>)`
    /// so steady-state idle pushes are zero-copy.
    ///
    /// # Single-instance contract
    ///
    /// One [`Placeholder`] instance is built per `(width, height)`
    /// pair. Calling [`render`] with different dimensions returns
    /// [`FluxError::Config`]: the supervisor must rebuild the
    /// placeholder via [`build`] when output dimensions change
    /// (e.g. on `reload`).
    ///
    /// # Errors
    ///
    /// Returns [`FluxError::Config`] when the placeholder cannot
    /// satisfy the request: unsupported target format, dimension
    /// mismatch against the built cache, or zero-sized output.
    ///
    /// [`render`]: Self::render
    fn render(&self, width: u32, height: u32, format: PixelFormat)
    -> Result<VideoFrame, FluxError>;
}

/// Build the placeholder described by `cfg`, sized for the
/// supervisor's output `(width, height)`. For
/// [`IdlePlaceholderKind::Image`] this is the eager decode + resize
/// — failure here surfaces a real [`FluxError::Config`] with the
/// full source chain (`io::Error` from `ENOENT`, decoder error from
/// a corrupt PNG, …), rather than the generic "failed to load"
/// surfaced by a deferred `OnceLock` path on first render.
///
/// # Errors
///
/// Returns [`FluxError::Config`] when:
/// * `placeholder = "image"` is set without a `placeholder_path`
///   (validation also catches this; defensive against bypass);
/// * the image file cannot be opened, parsed, or decoded;
/// * `width` or `height` is zero.
pub(crate) fn build(
    cfg: &IdleConfig,
    width: u32,
    height: u32,
) -> Result<Box<dyn Placeholder>, FluxError> {
    if width == 0 || height == 0 {
        return Err(FluxError::Config {
            reason: format!("placeholder asked for zero-sized frame {width}x{height}"),
            hint: None,
        });
    }
    match cfg.placeholder {
        IdlePlaceholderKind::Color => Ok(Box::new(ColorPlaceholder::new(
            cfg.placeholder_rgb,
            width,
            height,
        ))),
        IdlePlaceholderKind::Image => {
            // The image branch requires the `image-fill` feature to
            // pull in the PNG/JPEG decoder. Slim builds
            // (`--no-default-features`) report a clean config error
            // pointing the operator at the build flag rather than
            // silently swapping in the colour placeholder.
            #[cfg(feature = "image-fill")]
            {
                let path = cfg
                    .placeholder_path
                    .as_ref()
                    .ok_or_else(|| FluxError::Config {
                        reason: "idle.placeholder = \"image\" requires idle.placeholder_path"
                            .into(),
                        hint: Some("set idle.placeholder_path to a PNG/JPEG file".into()),
                    })?;
                ImagePlaceholder::load(path.as_path(), width, height)
                    .map(|p| Box::new(p) as Box<dyn Placeholder>)
            }
            #[cfg(not(feature = "image-fill"))]
            {
                Err(FluxError::Config {
                    reason: "image placeholders require the `image-fill` cargo feature".into(),
                    hint: Some(
                        "rebuild with --features image-fill or switch idle.placeholder to \"color\""
                            .into(),
                    ),
                })
            }
        }
        // `IdlePlaceholderKind` is `#[non_exhaustive]`; this arm
        // future-proofs the builder against new variants added in
        // later stages. Surface the missing handler loudly instead
        // of silently defaulting to colour fill.
        _ => Err(FluxError::Config {
            reason: format!("unsupported idle placeholder kind {:?}", cfg.placeholder),
            hint: Some("update fluxframe-cli to handle the new IdlePlaceholderKind variant".into()),
        }),
    }
}

// ---------------------------------------------------------------------------
// ColorPlaceholder — solid RGB fill, cached once.
// ---------------------------------------------------------------------------

/// Solid-fill placeholder. The cached RGB buffer is built once at
/// construction and reused on every `render` call.
pub(crate) struct ColorPlaceholder {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
}

impl ColorPlaceholder {
    fn new(rgb: [u8; 3], width: u32, height: u32) -> Self {
        // Callers (via `build`) validated non-zero dimensions, so the
        // `(width * height * 3)` allocation is well-defined: u32 fits
        // in usize on every target FluxFrame builds on, and the
        // resulting size is bounded by the upstream config-level cap
        // on output dims.
        let pixels = (width as usize) * (height as usize);
        let mut buf = vec![0u8; pixels * 3];
        for chunk in buf.chunks_exact_mut(3) {
            chunk.copy_from_slice(&rgb);
        }
        Self {
            width,
            height,
            rgb: buf,
        }
    }
}

impl Placeholder for ColorPlaceholder {
    fn render(
        &self,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<VideoFrame, FluxError> {
        ensure_dims_match(self.width, self.height, width, height)?;
        let bytes = convert_rgb_to(&self.rgb, width, height, format)?;
        wrap_into_frame(bytes, width, height, format)
    }
}

// ---------------------------------------------------------------------------
// ImagePlaceholder — eager decode + resize, cached RGB once.
// ---------------------------------------------------------------------------

/// Static-image placeholder. The image file is decoded and resized
/// to `(width, height)` at construction time via
/// [`decode_rgb_bounded`]; subsequent `render` calls do format
/// conversion only.
///
/// Gated behind the `image-fill` cargo feature so slim builds
/// (`--no-default-features`) do not link the PNG/JPEG decoder. The
/// `build()` constructor surfaces a clean config error pointing the
/// operator at the feature flag when the slim binary is asked for an
/// image placeholder.
#[cfg(feature = "image-fill")]
pub(crate) struct ImagePlaceholder {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
}

#[cfg(feature = "image-fill")]
impl ImagePlaceholder {
    fn load(path: &std::path::Path, width: u32, height: u32) -> Result<Self, FluxError> {
        let decoded = decode_rgb_bounded(path).map_err(|e: ImageLoadError| FluxError::Config {
            reason: e.to_string(),
            hint: Some("ensure the path exists and contains a valid PNG/JPEG".into()),
        })?;
        // `decode_rgb_bounded`'s contract guarantees
        // `data.len() == width * height * 3` and non-zero dims, so
        // `RgbImage::from_raw` cannot return `None`. The `expect`
        // catches a future regression in the decoder contract — not
        // user input.
        let src = image::RgbImage::from_raw(decoded.width, decoded.height, decoded.data).expect(
            "decode_rgb_bounded guarantees data.len() == width * height * 3 and non-zero dims",
        );
        let resized =
            image::imageops::resize(&src, width, height, image::imageops::FilterType::Triangle);
        Ok(Self {
            width,
            height,
            rgb: resized.into_raw(),
        })
    }
}

#[cfg(feature = "image-fill")]
impl Placeholder for ImagePlaceholder {
    fn render(
        &self,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<VideoFrame, FluxError> {
        ensure_dims_match(self.width, self.height, width, height)?;
        let bytes = convert_rgb_to(&self.rgb, width, height, format)?;
        wrap_into_frame(bytes, width, height, format)
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn ensure_dims_match(
    cached_w: u32,
    cached_h: u32,
    asked_w: u32,
    asked_h: u32,
) -> Result<(), FluxError> {
    if cached_w == asked_w && cached_h == asked_h {
        Ok(())
    } else {
        Err(FluxError::Config {
            reason: format!(
                "placeholder built for {cached_w}x{cached_h}, asked for {asked_w}x{asked_h}"
            ),
            hint: Some("supervisor must rebuild the placeholder when output dims change".into()),
        })
    }
}

/// Wrap a packed byte buffer in a [`VideoFrame`] with a synthetic
/// [`FrameMeta`]. The supervisor's writer thread overwrites
/// timestamps anyway, so meta values are nominal.
fn wrap_into_frame(
    bytes: Vec<u8>,
    width: u32,
    height: u32,
    format: PixelFormat,
) -> Result<VideoFrame, FluxError> {
    VideoFrame::new_packed(
        FrameBuffer::Owned(bytes),
        width,
        height,
        format,
        FrameMeta::default(),
    )
    .ok_or_else(|| FluxError::Config {
        reason: format!("placeholder frame construction failed for {format:?} {width}x{height}"),
        hint: None,
    })
}

/// Convert a packed RGB byte buffer to one of the supported target
/// formats. The supervisor's appsrc only emits a handful of formats
/// in practice; anything else is rejected loudly rather than
/// silently mangled.
fn convert_rgb_to(
    rgb: &[u8],
    width: u32,
    height: u32,
    target: PixelFormat,
) -> Result<Vec<u8>, FluxError> {
    let pixels = (width as usize) * (height as usize);
    debug_assert_eq!(rgb.len(), pixels * 3, "RGB source has wrong length");
    match target {
        PixelFormat::Rgb => Ok(rgb.to_vec()),
        PixelFormat::Bgr => Ok(rgb_to_bgr(rgb, pixels)),
        PixelFormat::Rgba => Ok(rgb_to_rgba(rgb, pixels)),
        PixelFormat::Gray8 => Ok(rgb_to_gray8(rgb, pixels)),
        PixelFormat::Yuy2 => rgb_to_yuy2(rgb, width, height),
        PixelFormat::Nv12 => Err(FluxError::Config {
            reason: "idle placeholder does not support NV12 appsrc format".into(),
            hint: Some(
                "set input.format to RGB / RGBA / BGR / YUY2 / GRAY8 — \
                 videoconvert handles the NV12 sink leg automatically"
                    .into(),
            ),
        }),
        // `PixelFormat` is `#[non_exhaustive]`. New variants surface
        // as a config error rather than silently producing a
        // mis-coloured frame.
        _ => Err(FluxError::Config {
            reason: format!("idle placeholder does not support pixel format {target:?}"),
            hint: Some("add a converter or change input.format".into()),
        }),
    }
}

fn rgb_to_bgr(rgb: &[u8], pixels: usize) -> Vec<u8> {
    let mut out = vec![0u8; pixels * 3];
    for (src, dst) in rgb.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
    }
    out
}

fn rgb_to_rgba(rgb: &[u8], pixels: usize) -> Vec<u8> {
    let mut out = vec![0u8; pixels * 4];
    for (src, dst) in rgb.chunks_exact(3).zip(out.chunks_exact_mut(4)) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = 0xFF;
    }
    out
}

fn rgb_to_gray8(rgb: &[u8], pixels: usize) -> Vec<u8> {
    let mut out = vec![0u8; pixels];
    for (src, dst) in rgb.chunks_exact(3).zip(out.iter_mut()) {
        let r = i32::from(src[0]);
        let g = i32::from(src[1]);
        let b = i32::from(src[2]);
        let y = (BT601_KR_Q8 * r + BT601_KG_Q8 * g + BT601_KB_Q8 * b + BT601_ROUND_Q8) >> 8;
        *dst = clamp_u8(y);
    }
    out
}

/// RGB → YUYV / YUY2 packed conversion using BT.601 with full-range
/// JPEG coefficients (see ITU-R BT.601-7 §2.5.1 + JFIF Y'CbCr
/// matrix). Two RGB pixels collapse to one 4-byte YUY2 pair:
/// `[Y0 U Y1 V]`. Odd widths are rejected because YUY2 is
/// fundamentally pair-packed.
fn rgb_to_yuy2(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, FluxError> {
    if width % 2 != 0 {
        return Err(FluxError::Config {
            reason: format!("YUY2 placeholder requires even width, got {width}"),
            hint: None,
        });
    }
    let pixels = (width as usize) * (height as usize);
    let mut out = vec![0u8; pixels * 2];
    let row_rgb = (width as usize) * 3;
    let row_yuy2 = (width as usize) * 2;
    // Iterate row-by-row so we can pair-step through the RGB source
    // (6 bytes = 2 pixels) and the YUY2 destination (4 bytes = 1
    // packed word) with the `chunks_exact` idiom used by the other
    // converters in this module. Width-evenness checked above
    // guarantees `chunks_exact` consumes every byte without a
    // remainder.
    for (rgb_row, yuy2_row) in rgb
        .chunks_exact(row_rgb)
        .zip(out.chunks_exact_mut(row_yuy2))
    {
        for (rgb_pair, yuy2_word) in rgb_row.chunks_exact(6).zip(yuy2_row.chunks_exact_mut(4)) {
            let (y0, u0, v0) = rgb_to_ycbcr(rgb_pair[0], rgb_pair[1], rgb_pair[2]);
            let (y1, u1, v1) = rgb_to_ycbcr(rgb_pair[3], rgb_pair[4], rgb_pair[5]);
            // Sub-sample chroma to one pair per two luma. The +1
            // rounding bias keeps the integer division unbiased.
            let u = ((u32::from(u0) + u32::from(u1) + 1) >> 1) as u8;
            let v = ((u32::from(v0) + u32::from(v1) + 1) >> 1) as u8;
            yuy2_word[0] = y0;
            yuy2_word[1] = u;
            yuy2_word[2] = y1;
            yuy2_word[3] = v;
        }
    }
    Ok(out)
}

/// BT.601 JPEG-range RGB → YCbCr, Q8 fixed point. Result components
/// are clamped to `0..=255` via [`clamp_u8`]; the un-clamped
/// intermediate stays inside `i32` so the BT.601 chroma offsets
/// (signed, `-128..=127`) round-trip without overflow.
fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = i32::from(r);
    let g = i32::from(g);
    let b = i32::from(b);
    let y = (BT601_KR_Q8 * r + BT601_KG_Q8 * g + BT601_KB_Q8 * b + BT601_ROUND_Q8) >> 8;
    let cb = ((BT601_CB_R_Q8 * r + BT601_CB_G_Q8 * g + BT601_CB_B_Q8 * b + BT601_ROUND_Q8) >> 8)
        + BT601_CHROMA_BIAS;
    let cr = ((BT601_CR_R_Q8 * r + BT601_CR_G_Q8 * g + BT601_CR_B_Q8 * b + BT601_ROUND_Q8) >> 8)
        + BT601_CHROMA_BIAS;
    (clamp_u8(y), clamp_u8(cb), clamp_u8(cr))
}

#[inline]
fn clamp_u8(v: i32) -> u8 {
    // `as u8` only well-defined after the clamp; the BT.601 math
    // above can in principle round past 255 by one — clamp keeps
    // the helper total without relying on hand-proofs.
    v.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn color_cfg(rgb: [u8; 3]) -> IdleConfig {
        IdleConfig {
            enabled: true,
            placeholder_rgb: rgb,
            ..IdleConfig::default()
        }
    }

    #[test]
    fn color_renders_rgb() {
        let p = build(&color_cfg([10, 20, 30]), 4, 2).expect("build");
        let frame = p.render(4, 2, PixelFormat::Rgb).expect("render rgb");
        assert_eq!(frame.width, 4);
        assert_eq!(frame.height, 2);
        assert_eq!(frame.format, PixelFormat::Rgb);
        let data = frame.data.as_slice();
        assert_eq!(data.len(), 4 * 2 * 3);
        for chunk in data.chunks_exact(3) {
            assert_eq!(chunk, [10, 20, 30]);
        }
    }

    #[test]
    fn color_converts_to_bgr() {
        let p = build(&color_cfg([10, 20, 30]), 2, 1).expect("build");
        let frame = p.render(2, 1, PixelFormat::Bgr).expect("render bgr");
        let data = frame.data.as_slice();
        for chunk in data.chunks_exact(3) {
            assert_eq!(chunk, [30, 20, 10]);
        }
    }

    #[test]
    fn color_converts_to_rgba_with_full_alpha() {
        let p = build(&color_cfg([10, 20, 30]), 2, 1).expect("build");
        let frame = p.render(2, 1, PixelFormat::Rgba).expect("render rgba");
        let data = frame.data.as_slice();
        for chunk in data.chunks_exact(4) {
            assert_eq!(chunk, [10, 20, 30, 0xFF]);
        }
    }

    #[test]
    fn color_converts_to_yuy2() {
        let p = build(&color_cfg([255, 0, 0]), 2, 1).expect("build");
        let frame = p.render(2, 1, PixelFormat::Yuy2).expect("render yuy2");
        let data = frame.data.as_slice();
        assert_eq!(data.len(), 4);
        assert_eq!(data[0], data[2], "Y plane is uniform for a solid fill");
    }

    #[test]
    fn color_rejects_yuy2_with_odd_width() {
        let p = build(&color_cfg([255, 0, 0]), 3, 1).expect("build");
        match p.render(3, 1, PixelFormat::Yuy2) {
            Ok(_) => panic!("odd width must be rejected"),
            Err(e) => assert!(format!("{e}").contains("even width"), "{e}"),
        }
    }

    #[test]
    fn color_converts_to_gray8() {
        let p = build(&color_cfg([255, 255, 255]), 2, 2).expect("build");
        let frame = p.render(2, 2, PixelFormat::Gray8).expect("render gray");
        let data = frame.data.as_slice();
        assert_eq!(data.len(), 4);
        for v in data {
            assert!(*v > 250, "luma {v} should be near 255");
        }
    }

    #[test]
    fn color_rejects_nv12() {
        let p = build(&color_cfg([10, 20, 30]), 4, 2).expect("build");
        match p.render(4, 2, PixelFormat::Nv12) {
            Ok(_) => panic!("nv12 should be rejected"),
            Err(e) => assert!(format!("{e}").contains("NV12"), "{e}"),
        }
    }

    #[test]
    fn build_rejects_zero_dimensions() {
        match build(&color_cfg([0, 0, 0]), 0, 10) {
            Ok(_) => panic!("zero width must be rejected"),
            Err(e) => assert!(format!("{e}").contains("zero-sized")),
        }
    }

    #[test]
    fn build_color_placeholder_from_config() {
        let p = build(&color_cfg([1, 2, 3]), 2, 1).expect("build color");
        let frame = p.render(2, 1, PixelFormat::Rgb).expect("render");
        assert_eq!(frame.data.as_slice(), [1, 2, 3, 1, 2, 3]);
    }

    #[cfg(feature = "image-fill")]
    #[test]
    fn build_image_placeholder_requires_path() {
        let cfg = IdleConfig {
            enabled: true,
            placeholder: IdlePlaceholderKind::Image,
            placeholder_path: None,
            ..IdleConfig::default()
        };
        // `Box<dyn Placeholder>` does not implement `Debug`, so we
        // match on the Result instead of using `expect_err`.
        match build(&cfg, 4, 2) {
            Ok(_) => panic!("image without path must fail"),
            Err(e) => assert!(format!("{e}").contains("placeholder_path")),
        }
    }

    #[cfg(not(feature = "image-fill"))]
    #[test]
    fn build_image_placeholder_rejects_without_image_fill_feature() {
        // Slim build: the image branch is configured out. The error
        // must point the operator at the `image-fill` feature flag
        // rather than silently falling back to the colour placeholder.
        let cfg = IdleConfig {
            enabled: true,
            placeholder: IdlePlaceholderKind::Image,
            placeholder_path: Some(std::path::PathBuf::from("/tmp/bg.png")),
            ..IdleConfig::default()
        };
        match build(&cfg, 4, 2) {
            Ok(_) => panic!("image kind must fail without image-fill feature"),
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("image-fill"),
                    "error must mention the feature flag, got: {msg}"
                );
            }
        }
    }

    #[test]
    fn render_rejects_dim_mismatch() {
        let p = build(&color_cfg([1, 2, 3]), 2, 1).expect("build");
        match p.render(4, 1, PixelFormat::Rgb) {
            Ok(_) => panic!("dim mismatch must be rejected"),
            Err(e) => {
                let msg = format!("{e}");
                // `FluxError::Config` renders only `reason`; the
                // "rebuild" hint is on the separate `hint` field.
                assert!(
                    msg.contains("built for 2x1") && msg.contains("asked for 4x1"),
                    "expected dim-mismatch reason, got: {msg}"
                );
            }
        }
    }
}
