//! Background-image replacement. Loads a PNG/JPEG file once in
//! `prepare()`, resizes it to the negotiated frame dimensions and
//! `memcpy`s the cached buffer into the plane on every frame.
//!
//! Use cases:
//! * `[background]` — corporate brand background, scenery, fixed
//!   stock photo. The most-requested feature for work-call virtual
//!   cameras.
//! * `[foreground]` — rare; would replace the speaker with the
//!   image (silhouette mode).
//!
//! Cost: `O(width × height)` per frame — a single `memcpy` from the
//! pre-resized scratch buffer.

use std::path::{Path, PathBuf};

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::EffectError;
use fluxframe_core::plane::{FramePlane, PlaneEffect};
use fluxframe_core::traits::RawEffectParams;
use image::imageops::FilterType;
use serde::Deserialize;

use crate::processing::fit::{fit_contain_rgb, fit_cover_rgb};

/// Hard upper bound on the source image dimensions accepted by
/// `ImageReader::limits`.  Picked to cover typical UHD/4K background
/// stills (8192 wide) without letting a malicious or accidental
/// gigapixel input run the decoder out of memory.
const MAX_IMAGE_DIM: u32 = 8192;

/// Hard upper bound on the in-flight allocation the decoder is allowed
/// to make while decoding.  256 MiB is enough for an 8192×8192 RGBA
/// frame (~256 MB) but stops the typical "decompression bomb" PNG
/// from claiming gigabytes of memory.
const MAX_IMAGE_ALLOC_BYTES: u64 = 256 * 1024 * 1024;

/// How to fit the loaded image into the negotiated frame dimensions.
///
/// * `Cover` (default) — scale the image so it fully covers the frame,
///   then centre-crop. Preserves aspect ratio; clips the long axis.
/// * `Stretch` — scale each axis independently; the image fills every
///   pixel but its aspect ratio can be wrong.
/// * `Contain` — scale the image so it fits entirely within the frame
///   and fill the remaining space with `letterbox_rgb`. Preserves
///   aspect ratio; leaves bars on one axis.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FitMode {
    /// Centre-crop after a max-axis scale.
    #[default]
    Cover,
    /// Independent X/Y scale.
    Stretch,
    /// Letterbox after a min-axis scale.
    Contain,
}

/// TOML schema:
///
/// ```toml
/// [background.image_fill]
/// path = "./assets/office-bg.jpg"
/// fit = "cover"           # cover | stretch | contain
/// letterbox_rgb = [0, 0, 0]
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageFillConfig {
    /// Path to the background image. Resolved relative to the current
    /// working directory of the `fluxframe` process when relative.
    pub path: PathBuf,
    /// Fit mode — see [`FitMode`] for the semantics.
    #[serde(default)]
    pub(crate) fit: FitMode,
    /// Letterbox colour used by [`FitMode::Contain`]; ignored for the
    /// other modes. Defaults to black.
    #[serde(default = "default_letterbox_rgb")]
    pub letterbox_rgb: [u8; 3],
}

fn default_letterbox_rgb() -> [u8; 3] {
    [0, 0, 0]
}

/// `PlaneEffect` that broadcasts a pre-loaded, pre-resized image into
/// the plane on every frame.
pub struct ImageFillEffect {
    config: Option<ImageFillConfig>,
    /// Pre-resized RGB buffer matching the negotiated frame dimensions.
    /// Built once in `prepare()`, lazily re-built in `process()` when
    /// a live `configure()` changed `path`/`fit`/`letterbox_rgb`.
    resized: Vec<u8>,
    frame_w: u32,
    frame_h: u32,
    /// Set by `prepare()` on success.  Guards `process()` against
    /// being called before `prepare()` so the error reason is honest
    /// instead of the misleading "plane MxN differs from prepared 0x0".
    prepared: bool,
    /// Snapshot of `(path, fit, letterbox_rgb)` the current `resized`
    /// buffer was generated against. When the live config drifts from
    /// this snapshot, `process()` reloads + resizes lazily before
    /// using the buffer.
    loaded_for: Option<(PathBuf, FitMode, [u8; 3])>,
}

impl ImageFillEffect {
    /// Effect name as registered in the plane registry.
    pub const NAME: &'static str = "image_fill";

    /// Construct without configuration — must be `configure`d and
    /// `prepare`d before use.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: None,
            resized: Vec::new(),
            frame_w: 0,
            frame_h: 0,
            prepared: false,
            loaded_for: None,
        }
    }
}

impl Default for ImageFillEffect {
    fn default() -> Self {
        Self::new()
    }
}

/// Walk an error chain via `Error::source()` and join every link with
/// ": " separators.  `image::ImageError`'s `Display` impl reports only
/// the top-level message; the IO/format error that caused the failure
/// lives one level deeper and is invisible to a plain `format!("{e}")`.
fn full_error_chain(err: &dyn std::error::Error) -> String {
    let mut s = err.to_string();
    let mut src = err.source();
    while let Some(inner) = src {
        s.push_str(": ");
        s.push_str(&inner.to_string());
        src = inner.source();
    }
    s
}

fn prepare_err(reason: impl Into<String>) -> EffectError {
    EffectError::PrepareFailed {
        name: ImageFillEffect::NAME.to_string(),
        reason: reason.into(),
    }
}

/// Load, decode and resize the image at `path` into a `width × height`
/// RGB buffer according to the fit mode.  Returns the resulting buffer
/// of length `width * height * 3`.
///
/// Decompression-bomb protection: `ImageReader` is configured with
/// per-axis dimension caps and a max-allocation budget before the
/// decoder runs, so a malicious PNG/JPEG cannot exhaust memory before
/// the resize stage even sees it.
fn load_and_resize(
    path: &Path,
    fit: FitMode,
    letterbox_rgb: [u8; 3],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, EffectError> {
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }

    let mut reader = image::ImageReader::open(path)
        .map_err(|e| prepare_err(format!("failed to open image {}: {}", path.display(), e)))?
        .with_guessed_format()
        .map_err(|e| {
            prepare_err(format!(
                "failed to guess format for {}: {}",
                path.display(),
                e
            ))
        })?;

    // `image::Limits` is `#[non_exhaustive]`, so we cannot use a
    // struct-literal expression to construct it from outside the
    // `image` crate.  Mutate the default-constructed instance instead.
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIM);
    limits.max_image_height = Some(MAX_IMAGE_DIM);
    limits.max_alloc = Some(MAX_IMAGE_ALLOC_BYTES);
    reader.limits(limits);

    let img = reader.decode().map_err(|e| {
        prepare_err(format!(
            "failed to decode image {}: {}",
            path.display(),
            full_error_chain(&e)
        ))
    })?;
    let rgb = img.into_rgb8();
    let (src_w, src_h) = rgb.dimensions();
    if src_w == 0 || src_h == 0 {
        return Err(prepare_err(format!(
            "image {} has zero-sized dimension",
            path.display()
        )));
    }

    match fit {
        FitMode::Stretch => {
            let resized = image::imageops::resize(&rgb, width, height, FilterType::Triangle);
            Ok(resized.into_raw())
        }
        FitMode::Cover => Ok(fit_cover_rgb(rgb.as_raw(), src_w, src_h, width, height)),
        FitMode::Contain => Ok(fit_contain_rgb(
            rgb.as_raw(),
            src_w,
            src_h,
            width,
            height,
            letterbox_rgb,
        )),
    }
}

impl PlaneEffect for ImageFillEffect {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn configure(&mut self, params: RawEffectParams) -> Result<(), EffectError> {
        let cfg: ImageFillConfig = params
            .try_into()
            .map_err(|e: toml::de::Error| super::invalid_config(Self::NAME, e.to_string()))?;
        self.config = Some(cfg);
        Ok(())
    }

    fn prepare(&mut self, context: &ProcessingContext) -> Result<(), EffectError> {
        let cfg = self
            .config
            .as_ref()
            .ok_or_else(|| prepare_err("prepare called before configure"))?;
        self.frame_w = context.width;
        self.frame_h = context.height;
        self.resized = load_and_resize(
            &cfg.path,
            cfg.fit,
            cfg.letterbox_rgb,
            context.width,
            context.height,
        )?;
        self.loaded_for = Some((cfg.path.clone(), cfg.fit, cfg.letterbox_rgb));
        self.prepared = true;
        tracing::info!(
            effect = Self::NAME,
            path = %cfg.path.display(),
            width = context.width,
            height = context.height,
            "loaded background image"
        );
        Ok(())
    }

    fn process(
        &mut self,
        plane: &mut FramePlane<'_>,
        _ctx: &mut FrameContext,
    ) -> Result<(), EffectError> {
        if !self.prepared {
            return Err(EffectError::ProcessFailed {
                name: Self::NAME.to_string(),
                reason: "process called before prepare".into(),
            });
        }
        if plane.width != self.frame_w || plane.height != self.frame_h {
            return Err(EffectError::ProcessFailed {
                name: Self::NAME.to_string(),
                reason: format!(
                    "plane {}x{} differs from prepared {}x{}",
                    plane.width, plane.height, self.frame_w, self.frame_h
                ),
            });
        }
        // Live `configure()` between prepare and process can change
        // `path`/`fit`/`letterbox_rgb`. Reload lazily when the active
        // config drifts from the snapshot the `resized` buffer was
        // generated against.
        if let Some(cfg) = self.config.as_ref() {
            let current = (cfg.path.clone(), cfg.fit, cfg.letterbox_rgb);
            let drifted = self
                .loaded_for
                .as_ref()
                .is_none_or(|loaded| loaded != &current);
            if drifted {
                self.resized = load_and_resize(
                    &cfg.path,
                    cfg.fit,
                    cfg.letterbox_rgb,
                    self.frame_w,
                    self.frame_h,
                )
                .map_err(|e| EffectError::ProcessFailed {
                    name: Self::NAME.to_string(),
                    reason: format!("live reload failed: {e}"),
                })?;
                self.loaded_for = Some(current);
            }
        }
        if self.resized.len() != plane.data.len() {
            return Err(EffectError::ProcessFailed {
                name: Self::NAME.to_string(),
                reason: format!(
                    "resized buffer ({} B) does not match plane ({} B)",
                    self.resized.len(),
                    plane.data.len()
                ),
            });
        }
        plane.data.copy_from_slice(&self.resized);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn make_png(width: u32, height: u32, rgb: [u8; 3]) -> NamedTempFile {
        let img = image::RgbImage::from_pixel(width, height, image::Rgb(rgb));
        let file = NamedTempFile::with_suffix(".png").expect("tempfile");
        img.save(file.path()).expect("save png");
        file
    }

    fn parse(text: &str) -> RawEffectParams {
        toml::from_str(text).unwrap()
    }

    fn ctx(w: u32, h: u32) -> ProcessingContext {
        ProcessingContext {
            width: w,
            height: h,
            format: fluxframe_core::PixelFormat::Rgb,
            fps: 30,
            counters: None,
        }
    }

    #[test]
    fn configure_parses_minimal_toml() {
        let mut effect = ImageFillEffect::new();
        let params = parse(r#"path = "/tmp/bg.png""#);
        effect.configure(params).expect("configure");
        let cfg = effect.config.as_ref().unwrap();
        assert_eq!(cfg.path, std::path::Path::new("/tmp/bg.png"));
        assert_eq!(cfg.fit, FitMode::Cover);
        assert_eq!(cfg.letterbox_rgb, [0, 0, 0]);
    }

    #[test]
    fn configure_parses_all_fit_modes() {
        for fit_str in ["cover", "stretch", "contain"] {
            let mut effect = ImageFillEffect::new();
            let params = parse(&format!(
                r#"
path = "/tmp/bg.png"
fit = "{fit_str}"
"#
            ));
            effect.configure(params).expect("configure");
        }
    }

    #[test]
    fn configure_rejects_unknown_fit() {
        let mut effect = ImageFillEffect::new();
        let params: Result<RawEffectParams, _> = toml::from_str(
            r#"
path = "/tmp/bg.png"
fit = "weird"
"#,
        );
        // Either toml-level parse fails (acceptable) or effect.configure
        // surfaces an InvalidConfig.
        match params {
            Err(_) => {} // OK — serde rejected
            Ok(p) => assert!(effect.configure(p).is_err()),
        }
    }

    #[test]
    fn prepare_fails_when_file_missing() {
        let mut effect = ImageFillEffect::new();
        effect
            .configure(parse(r#"path = "/nonexistent/bg.png""#))
            .unwrap();
        let err = effect
            .prepare(&ctx(4, 4))
            .expect_err("missing file must surface PrepareFailed");
        match err {
            EffectError::PrepareFailed { reason, .. } => {
                assert!(reason.contains("failed to open image"), "reason: {reason}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn stretch_fills_with_image_color() {
        let bg = make_png(2, 2, [200, 50, 100]);
        let mut effect = ImageFillEffect::new();
        effect
            .configure(parse(&format!(
                r#"
path = {:?}
fit = "stretch"
"#,
                bg.path()
            )))
            .unwrap();
        effect.prepare(&ctx(4, 4)).expect("prepare");
        let mut data = vec![0u8; 4 * 4 * 3];
        let mut plane = FramePlane::new(&mut data, 4, 4);
        let mut fctx = FrameContext::default();
        effect.process(&mut plane, &mut fctx).expect("process");
        // Stretching a single-colour image: every pixel must match the
        // source colour exactly (no border, no resampling artefact).
        for chunk in data.chunks_exact(3) {
            assert_eq!(chunk, [200, 50, 100]);
        }
    }

    #[test]
    fn contain_letterboxes_with_configured_colour() {
        // A 4×1 wide source resized into a 4×4 frame with `contain` must
        // produce one row of source and three rows of letterbox.
        let bg = make_png(4, 1, [255, 0, 0]);
        let mut effect = ImageFillEffect::new();
        effect
            .configure(parse(&format!(
                r#"
path = {:?}
fit = "contain"
letterbox_rgb = [10, 20, 30]
"#,
                bg.path()
            )))
            .unwrap();
        effect.prepare(&ctx(4, 4)).expect("prepare");
        let mut data = vec![0u8; 4 * 4 * 3];
        let mut plane = FramePlane::new(&mut data, 4, 4);
        let mut fctx = FrameContext::default();
        effect.process(&mut plane, &mut fctx).expect("process");
        // First and last rows should be letterbox.
        let row0 = &data[0..12];
        let row3 = &data[36..48];
        for chunk in row0.chunks_exact(3) {
            assert_eq!(chunk, [10, 20, 30], "top row letterbox");
        }
        for chunk in row3.chunks_exact(3) {
            assert_eq!(chunk, [10, 20, 30], "bottom row letterbox");
        }
        // At least one middle row should be source colour.
        let mut found_source = false;
        for row in [&data[12..24], &data[24..36]] {
            if row.chunks_exact(3).all(|c| c == [255, 0, 0]) {
                found_source = true;
            }
        }
        assert!(found_source, "expected at least one row of source colour");
    }

    #[test]
    fn cover_preserves_aspect_via_centre_crop() {
        // A 2×4 (tall) source covered into a 4×4 frame must produce a
        // single colour everywhere (the source is monochrome) — the
        // test only verifies cover doesn't crash and fills the plane.
        let bg = make_png(2, 4, [50, 150, 250]);
        let mut effect = ImageFillEffect::new();
        effect
            .configure(parse(&format!(
                r#"
path = {:?}
fit = "cover"
"#,
                bg.path()
            )))
            .unwrap();
        effect.prepare(&ctx(4, 4)).expect("prepare");
        let mut data = vec![0u8; 4 * 4 * 3];
        let mut plane = FramePlane::new(&mut data, 4, 4);
        let mut fctx = FrameContext::default();
        effect.process(&mut plane, &mut fctx).expect("process");
        // Monochrome source → every pixel still that colour after the
        // cover-scale + crop.
        for chunk in data.chunks_exact(3) {
            assert_eq!(chunk, [50, 150, 250]);
        }
    }

    #[test]
    fn process_fails_before_prepare() {
        let mut effect = ImageFillEffect::new();
        effect.configure(parse(r#"path = "/tmp/bg.png""#)).unwrap();
        // Skip prepare — process should refuse rather than panic.
        let mut data = vec![0u8; 4 * 4 * 3];
        let mut plane = FramePlane::new(&mut data, 4, 4);
        let mut fctx = FrameContext::default();
        let err = effect.process(&mut plane, &mut fctx).unwrap_err();
        match err {
            EffectError::ProcessFailed { reason, .. } => {
                assert!(
                    reason.contains("process called before prepare"),
                    "reason: {reason}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn process_fails_on_dimension_mismatch() {
        // Configure + prepare for 8×8 then call process with a 4×4
        // plane: the dimension check must fire with a reason that
        // mentions the mismatch (not the misleading 0×0 from the
        // un-prepared state, since `prepared` is now true).
        let bg = make_png(2, 2, [123, 45, 67]);
        let mut effect = ImageFillEffect::new();
        effect
            .configure(parse(&format!(
                r#"
path = {:?}
fit = "cover"
"#,
                bg.path()
            )))
            .unwrap();
        effect.prepare(&ctx(8, 8)).expect("prepare");
        let mut data = vec![0u8; 4 * 4 * 3];
        let mut plane = FramePlane::new(&mut data, 4, 4);
        let mut fctx = FrameContext::default();
        let err = effect.process(&mut plane, &mut fctx).unwrap_err();
        match err {
            EffectError::ProcessFailed { reason, .. } => {
                assert!(
                    reason.contains("4x4") && reason.contains("8x8"),
                    "expected mismatch reason mentioning both 4x4 and 8x8, got: {reason}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn configure_after_prepare_reloads_image_lazily() {
        // Live-reconfig contract: change `path` post-prepare; next
        // `process()` must lazily reload + resize the new image.
        let bg_a = make_png(2, 2, [200, 0, 0]); // pure red
        let bg_b = make_png(2, 2, [0, 200, 0]); // pure green
        let mut effect = ImageFillEffect::new();
        effect
            .configure(parse(&format!(
                r#"
path = {:?}
fit = "stretch"
"#,
                bg_a.path()
            )))
            .unwrap();
        effect.prepare(&ctx(4, 4)).expect("prepare");

        // First process: red.
        let mut data = vec![0u8; 4 * 4 * 3];
        let mut plane = FramePlane::new(&mut data, 4, 4);
        let mut fctx = FrameContext::default();
        effect.process(&mut plane, &mut fctx).expect("process A");
        assert_eq!(&data[0..3], &[200, 0, 0]);

        // Live re-configure to point at the green image.
        effect
            .configure(parse(&format!(
                r#"
path = {:?}
fit = "stretch"
"#,
                bg_b.path()
            )))
            .expect("reconfigure ok");

        // Second process: green — lazy reload via `loaded_for` drift
        // detection.
        let mut data2 = vec![0u8; 4 * 4 * 3];
        let mut plane2 = FramePlane::new(&mut data2, 4, 4);
        effect.process(&mut plane2, &mut fctx).expect("process B");
        assert_eq!(&data2[0..3], &[0, 200, 0]);
    }
}
