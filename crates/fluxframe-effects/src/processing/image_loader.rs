//! Bounded image decoding shared between background plane effects and
//! the CLI's idle-mode placeholder.
//!
//! The image-decode boilerplate (open → `with_guessed_format` → decode
//! → `into_rgb8`) plus the decompression-bomb limits are identical
//! across callers; consolidating them here avoids the constants and
//! the deep-error-chain walker drifting out of sync.
//!
//! Gated behind the `image-fill` cargo feature so slim builds without
//! the `image` crate stay link-clean.

use std::path::{Path, PathBuf};

use fluxframe_core::error::full_error_chain;

/// Hard upper bound on the source image dimensions accepted by
/// [`image::Limits`]. Picked to cover typical UHD/4K background stills
/// (8192 wide) without letting a malicious or accidental gigapixel
/// input run the decoder out of memory.
pub(crate) const MAX_IMAGE_DIM: u32 = 8192;

/// Hard upper bound on the in-flight allocation the decoder is allowed
/// to make while decoding. 256 MiB is enough for an 8192×8192 RGBA
/// frame (~256 MB) but stops the typical "decompression bomb" PNG
/// from claiming gigabytes of memory.
pub(crate) const MAX_IMAGE_ALLOC_BYTES: u64 = 256 * 1024 * 1024;

/// Failure stages exposed in [`ImageLoadError`]. Carried as a static
/// string so callers can convert into their domain error variant
/// (e.g. `EffectError::PrepareFailed` vs `FluxError::Config`) without
/// matching on a closed enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImageLoadStage {
    /// `image::ImageReader::open` failed (typically `io::Error`).
    Open,
    /// `ImageReader::with_guessed_format` failed (corrupt header).
    GuessFormat,
    /// Decoder failed (format-specific error chain).
    Decode,
    /// The decoded image has a zero-sized dimension.
    ZeroDimension,
}

impl ImageLoadStage {
    /// Human-readable label used by the `Display` impl on
    /// [`ImageLoadError`] for the non-zero-dimension stages.
    ///
    /// Returns `None` for [`ImageLoadStage::ZeroDimension`] because the
    /// `Display` impl handles that variant with a dedicated message
    /// ("image … has zero-sized dimension") rather than the generic
    /// "failed to {label} {path}" template. Surfacing `None` here is a
    /// trap for refactors: any future code that tries to use the label
    /// for `ZeroDimension` is forced to acknowledge the special case
    /// instead of silently emitting a misleading "use" string.
    #[must_use]
    pub fn label(self) -> Option<&'static str> {
        match self {
            ImageLoadStage::Open => Some("open"),
            ImageLoadStage::GuessFormat => Some("guess format for"),
            ImageLoadStage::Decode => Some("decode"),
            ImageLoadStage::ZeroDimension => None,
        }
    }
}

/// Bounded-decode failure carrying both the originating path and the
/// fully-walked `Error::source()` chain. Callers render it through
/// `Display`; the deep chain is what surfaces the underlying
/// `io::Error` (ENOENT, permission denied, …) that the top-level
/// `image::ImageError::Display` would otherwise hide.
#[derive(Debug)]
pub struct ImageLoadError {
    /// Filesystem path the failed operation was targeting.
    pub path: PathBuf,
    /// Which stage of the decode pipeline produced the error.
    pub stage: ImageLoadStage,
    /// Concatenation of every `Error::source()` link, separated by
    /// `": "`. Empty for [`ImageLoadStage::ZeroDimension`].
    pub chain: String,
}

impl std::fmt::Display for ImageLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.stage.label() {
            Some(label) => write!(
                f,
                "failed to {} {}: {}",
                label,
                self.path.display(),
                self.chain
            ),
            None => write!(f, "image {} has zero-sized dimension", self.path.display()),
        }
    }
}

impl std::error::Error for ImageLoadError {}

/// Decoded RGB8 image as raw bytes. Consumers reconstruct
/// `image::RgbImage` locally with `ImageBuffer::from_raw` when they
/// need it; keeping the returned struct neutral means callers do not
/// have to pull the `image` crate into their public API.
///
/// The contract is: `data.len() == width as usize * height as usize *
/// 3` and `width > 0 && height > 0`. [`decode_rgb_bounded`] is the
/// only constructor and it enforces both invariants before returning.
#[derive(Debug, Clone)]
pub struct DecodedRgb {
    /// Tightly packed `width * height * 3` RGB bytes.
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// Open `path`, apply the decompression-bomb limits and return the
/// fully-decoded RGB8 image. Caller decides what to do with the
/// dimensions (stretch, cover-crop, letterbox).
///
/// # Errors
///
/// Returns [`ImageLoadError`] tagged with the failing stage and the
/// full source chain. Zero-sized images are also rejected here so
/// downstream resize calls never face a degenerate input.
pub fn decode_rgb_bounded(path: &Path) -> Result<DecodedRgb, ImageLoadError> {
    let mut reader = image::ImageReader::open(path).map_err(|e| ImageLoadError {
        path: path.to_path_buf(),
        stage: ImageLoadStage::Open,
        chain: full_error_chain(&e),
    })?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIM);
    limits.max_image_height = Some(MAX_IMAGE_DIM);
    limits.max_alloc = Some(MAX_IMAGE_ALLOC_BYTES);
    reader.limits(limits);
    let reader = reader.with_guessed_format().map_err(|e| ImageLoadError {
        path: path.to_path_buf(),
        stage: ImageLoadStage::GuessFormat,
        chain: full_error_chain(&e),
    })?;
    let img = reader.decode().map_err(|e| ImageLoadError {
        path: path.to_path_buf(),
        stage: ImageLoadStage::Decode,
        chain: full_error_chain(&e),
    })?;
    let rgb = img.into_rgb8();
    let (src_w, src_h) = rgb.dimensions();
    if src_w == 0 || src_h == 0 {
        return Err(ImageLoadError {
            path: path.to_path_buf(),
            stage: ImageLoadStage::ZeroDimension,
            chain: String::new(),
        });
    }
    Ok(DecodedRgb {
        data: rgb.into_raw(),
        width: src_w,
        height: src_h,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn decode_rgb_bounded_happy_path() {
        // Synthesise a 2×2 RGB PNG and round-trip it through the
        // bounded decoder. Asserts both dimensions and tight packing
        // so the contract documented on `DecodedRgb` is exercised.
        let img = image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30]));
        let file = NamedTempFile::with_suffix(".png").expect("tempfile");
        img.save(file.path()).expect("save png");

        let decoded = decode_rgb_bounded(file.path()).expect("decode 2x2 rgb");
        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.data.len(), 12, "2 * 2 * 3 = 12");
        // Every pixel must match the source colour.
        for chunk in decoded.data.chunks_exact(3) {
            assert_eq!(chunk, [10, 20, 30]);
        }
    }

    #[test]
    fn decode_rgb_bounded_enoent_returns_open_stage() {
        // Non-existent path: the decoder cannot even open the file.
        // The error's chain must contain something resembling "no such
        // file" so the operator sees the IO root cause and the stage
        // must be `Open` so callers can branch on it.
        let err = decode_rgb_bounded(Path::new("/nonexistent/definitely-not-here.png"))
            .expect_err("missing file must fail");
        assert_eq!(err.stage, ImageLoadStage::Open);
        let chain = err.chain.to_lowercase();
        assert!(
            chain.contains("no such file") || chain.contains("not found"),
            "chain must surface the IO error, got: {chain}"
        );
    }

    #[test]
    fn decode_rgb_bounded_corrupt_data_returns_decode_or_guess_stage() {
        // 32 bytes of garbage written to a `.png`: either
        // `with_guessed_format` rejects the header or the decoder
        // chokes on the malformed body. Both are acceptable failures;
        // the test stays flexible because the exact behaviour depends
        // on the `image` crate's internal heuristics, which can shift
        // between versions.
        let mut file = NamedTempFile::with_suffix(".png").expect("tempfile");
        file.write_all(&[0u8; 32]).expect("write garbage");
        file.flush().expect("flush");
        let err = decode_rgb_bounded(file.path()).expect_err("garbage data must fail");
        assert!(
            matches!(
                err.stage,
                ImageLoadStage::Decode | ImageLoadStage::GuessFormat
            ),
            "garbage PNG must fail at Decode or GuessFormat, got: {:?}",
            err.stage
        );
    }

    #[test]
    fn full_error_chain_walks_nested_sources() {
        // Three-layer synthetic error: outer → middle → inner. The
        // `full_error_chain` helper must concatenate every layer's
        // `Display` separated by `": "`.
        #[derive(Debug)]
        struct Layered {
            msg: &'static str,
            source: Option<Box<Layered>>,
        }
        impl std::fmt::Display for Layered {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.msg)
            }
        }
        impl std::error::Error for Layered {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                self.source
                    .as_deref()
                    .map(|b| b as &(dyn std::error::Error + 'static))
            }
        }

        let inner = Layered {
            msg: "deep",
            source: None,
        };
        let middle = Layered {
            msg: "inner",
            source: Some(Box::new(inner)),
        };
        let outer = Layered {
            msg: "outer",
            source: Some(Box::new(middle)),
        };
        let chain = full_error_chain(&outer);
        assert_eq!(chain, "outer: inner: deep");
    }
}
