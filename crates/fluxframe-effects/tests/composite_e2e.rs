//! End-to-end test for [`CompositeEffect::process`].
//!
//! Runs the full composite path (segmentation → mask chain → bg/fg
//! chains → alpha composite) against a deterministic 2×2 RGB frame,
//! using a mock [`InferenceEngine`] in place of a real ONNX model.
//! The mock returns a fixed mask (top-left pixel = foreground, rest =
//! background), so the assertions can pin the exact bytes that come
//! out of the composite — top-left pixel is unchanged, the others are
//! replaced by the [`ColorFillEffect`] in the background chain.
//!
//! The composite's segmentation stage reads a `<model>.toml` sidecar
//! via [`fluxframe_effects::ml::load_sidecar_or_placeholder`]. We
//! write a tiny sidecar to a temp dir so the model dimensions match
//! the mock's output (2×2). Without the sidecar the loader falls back
//! to a 1×1 placeholder and the test would never exercise the real
//! resize/decode path.
//!
//! Gated on `feature = "ml"` because the composite types (and the
//! `InferenceEngine` mock) are only compiled when ml is active. The
//! file is `#![cfg(feature = "ml")]` rather than declared in
//! `[[test]] required-features` to avoid touching Cargo.toml.

#![cfg(feature = "ml")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use fluxframe_core::context::{FrameContext, ProcessingContext};
use fluxframe_core::error::InferenceError;
use fluxframe_core::frame::{FrameBuffer, FrameMeta, PixelFormat, VideoFrame};
use fluxframe_core::metrics::{EffectTelemetry, StageKey, StageTimings};
use fluxframe_core::plane::{MaskEffect, PlaneEffect, PostEffect, SubchainKind};
use fluxframe_core::traits::{
    InferenceEngine, InferenceInput, InferenceOutput, ModelInfo, RawEffectParams, VideoEffect,
};

use fluxframe_effects::composite::{CompositeEffect, SegmentationBase, SegmentationConfig, Slot};
use fluxframe_effects::ml::ModelConfig;
use fluxframe_effects::plane_effects::ColorFillEffect;
use fluxframe_effects::{mask_effects, plane_effects, post_effects};

// -----------------------------------------------------------------
// Mock inference engine. Returns a fixed 2×2 mask regardless of input.
// -----------------------------------------------------------------

/// Mock engine: returns a deterministic 2×2 mask on every `infer` call.
///
/// Mask layout (HW): row 0 = `[1.0, 0.0]`, row 1 = `[0.0, 0.0]`. The
/// `(0, 0)` pixel is the only foreground location so the assertions
/// can verify both branches of the alpha composite from a single frame.
struct MockEngine;

impl InferenceEngine for MockEngine {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: Arc::from("mock-2x2"),
            input_width: 2,
            input_height: 2,
            input_format: PixelFormat::Rgb,
        }
    }

    fn infer(&mut self, _input: InferenceInput<'_>) -> Result<InferenceOutput, InferenceError> {
        // Foreground at (0,0), background everywhere else.
        Ok(InferenceOutput {
            data: vec![1.0_f32, 0.0, 0.0, 0.0],
            shape: vec![2, 2],
        })
    }
}

// -----------------------------------------------------------------
// Sidecar TOML writer. The temp dir is process-id-scoped so parallel
// `cargo test` runs do not collide, and the directory is removed at
// the end of the test on success. On panic the leftover artefacts
// are tiny (`/tmp/fluxframe-composite-e2e-*/dummy.{onnx,toml}`) and
// the OS cleans them up on reboot.
// -----------------------------------------------------------------

struct ModelSidecar {
    dir: PathBuf,
    model: PathBuf,
}

impl ModelSidecar {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "fluxframe-composite-e2e-{}-{}",
            std::process::id(),
            // include a per-test salt in case multiple tests in this
            // file ever want their own sidecars
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let model = dir.join("dummy.onnx");
        std::fs::write(&model, b"").expect("write dummy.onnx");
        // 2×2 mask output matching the mock engine. `output_layout = "HW"`
        // is the default but spelled out for clarity. `input_scale = 1.0`
        // because the mock ignores the input anyway and we want
        // `validate()` to pass without a divide.
        let sidecar = dir.join("dummy.toml");
        std::fs::write(
            &sidecar,
            r#"
name = "mock"
input_width = 2
input_height = 2
input_layout = "NHWC"
output_layout = "HW"
output_type = "mask"
input_scale = 1.0
"#,
        )
        .expect("write dummy.toml");
        // Sanity: the sidecar must parse with the same parser the
        // composite pipeline uses.
        let _ = ModelConfig::load(&sidecar).expect("sidecar parses");
        Self { dir, model }
    }

    fn model_path(&self) -> &Path {
        &self.model
    }
}

impl Drop for ModelSidecar {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

// -----------------------------------------------------------------
// The test itself.
// -----------------------------------------------------------------

/// Inference factory matching [`fluxframe_effects::composite::segmentation::InferenceFactory`].
///
/// Declared as a free `fn` (not a closure) so its `for<'a>` HRTB
/// lifetime is inferred unambiguously; a bare closure trips an
/// `FnOnce` generality check. The factory contract is fallible — the
/// `Result` wrapper is part of the signature even though this mock
/// cannot fail.
#[allow(
    clippy::unnecessary_wraps,
    reason = "signature dictated by InferenceFactory trait alias"
)]
fn mock_factory(
    _path: &Path,
    _cfg: ModelConfig,
) -> Result<Box<dyn InferenceEngine + Send>, InferenceError> {
    Ok(Box::new(MockEngine))
}

#[test]
fn composite_alpha_composites_foreground_over_background_fill() {
    let sidecar = ModelSidecar::new();

    // Segmentation stage configured with the dummy model + mock engine.
    let seg_cfg = SegmentationConfig {
        model: sidecar.model_path().to_path_buf(),
        fallback_threshold: 3,
    };
    let segmentation =
        SegmentationBase::new(seg_cfg).with_inference_factory(Box::new(mock_factory));

    // Background chain: a single colour fill with a distinctive colour
    // so the assertion can distinguish bg from fg without ambiguity.
    let mut bg_color = ColorFillEffect::new([200, 100, 50]);
    let params: RawEffectParams = toml::from_str("rgb = [200, 100, 50]").expect("toml ok");
    bg_color.configure(params).expect("configure ok");
    let bg_chain: Vec<Slot<dyn PlaneEffect>> = vec![Slot::new(Box::new(bg_color), true)];

    let mut composite =
        CompositeEffect::new(segmentation, Vec::new(), bg_chain, Vec::new(), Vec::new());

    // ProcessingContext at 2×2 RGB — matches the mock's mask resolution
    // so no resize is actually exercised on the mask side (the bilinear
    // upscale collapses to a copy when src and dst dimensions match).
    let context = ProcessingContext {
        width: 2,
        height: 2,
        format: PixelFormat::Rgb,
        fps: 30,
        counters: None,
    };
    composite.prepare(&context).expect("prepare ok");

    // 2×2 RGB frame with one distinctive byte triple per pixel.
    let original_pixels: [[u8; 3]; 4] = [
        [10, 20, 30],    // (0,0) — foreground
        [40, 50, 60],    // (0,1) — background → fill
        [70, 80, 90],    // (1,0) — background → fill
        [100, 110, 120], // (1,1) — background → fill
    ];
    let mut data = Vec::with_capacity(4 * 3);
    for px in &original_pixels {
        data.extend_from_slice(px);
    }
    let mut frame = VideoFrame::new_packed(
        FrameBuffer::Owned(data),
        2,
        2,
        PixelFormat::Rgb,
        FrameMeta::default(),
    )
    .expect("frame ok");
    // Per-stage timing sink: the composite must publish its own
    // stages plus one key per configured sub-effect.
    let stages = Arc::new(StageTimings::with_capacity(8));
    let mut ctx = FrameContext {
        telemetry: EffectTelemetry::default().with_stages(Arc::clone(&stages)),
        ..FrameContext::default()
    };

    composite.process(&mut frame, &mut ctx).expect("process ok");

    let recorded = stages.snapshot();
    for expected in [
        StageKey::new("composite", "seg"),
        StageKey::new("seg", "preprocess"),
        StageKey::new("seg", "decode"),
        StageKey::new("composite", "mask_chain"),
        StageKey::new("composite", "mask_resize"),
        StageKey::new("composite", "bg_copy"),
        StageKey::new("composite", "bg_chain"),
        StageKey::new("background", "color_fill"),
        StageKey::new("composite", "fg_chain"),
        StageKey::new("composite", "compose"),
        StageKey::new("composite", "post_chain"),
    ] {
        let stage = recorded
            .iter()
            .find(|s| s.key == expected)
            .unwrap_or_else(|| panic!("stage {expected} must be recorded"));
        assert_eq!(
            stage.latency.len(),
            1,
            "one frame → one sample for {expected}"
        );
    }

    let out = frame.data.as_slice();
    assert_eq!(out.len(), 12, "2x2 RGB frame must be 12 bytes");

    // Top-left pixel: foreground passes through unchanged.
    assert_eq!(
        &out[0..3],
        &original_pixels[0],
        "foreground pixel (0,0) must be preserved (mask=1.0)",
    );

    // The other three pixels: background fill colour.
    for (idx, slot) in (1..4).enumerate() {
        let start = slot * 3;
        let actual = &out[start..start + 3];
        assert_eq!(
            actual,
            &[200, 100, 50],
            "pixel {slot} (idx {idx}) must be the bg fill colour (mask=0.0)",
        );
    }

    assert!(
        !ctx.fallback_active,
        "successful inference must not flag fallback",
    );
}

// -----------------------------------------------------------------
// Enable / disable: one test per sub-chain.
// -----------------------------------------------------------------

/// The four sub-chains of a composite under test.
#[derive(Default)]
struct Chains {
    mask: Vec<Slot<dyn MaskEffect>>,
    bg: Vec<Slot<dyn PlaneEffect>>,
    fg: Vec<Slot<dyn PlaneEffect>>,
    post: Vec<Slot<dyn PostEffect>>,
}

fn configured<E: ?Sized>(
    effect: Option<Box<E>>,
    configure: impl FnOnce(&mut E, RawEffectParams) -> Result<(), fluxframe_core::EffectError>,
    params: &str,
) -> Slot<E> {
    let mut effect = effect.expect("effect is registered");
    configure(&mut effect, toml::from_str(params).expect("toml ok")).expect("configure ok");
    Slot::new(effect, true)
}

fn mask_fx(name: &str) -> Slot<dyn MaskEffect> {
    configured(
        mask_effects::default_registry().build(name),
        MaskEffect::configure,
        "",
    )
}

fn fill(rgb: [u8; 3]) -> Slot<dyn PlaneEffect> {
    configured(
        plane_effects::default_registry().build("color_fill"),
        PlaneEffect::configure,
        &format!("rgb = {rgb:?}"),
    )
}

fn post_fx(name: &str) -> Slot<dyn PostEffect> {
    configured(
        post_effects::default_registry().build(name),
        PostEffect::configure,
        "",
    )
}

/// Prepared composite over the mock segmentation.
fn prepared(sidecar: &ModelSidecar, chains: Chains) -> CompositeEffect {
    let segmentation = SegmentationBase::new(SegmentationConfig {
        model: sidecar.model_path().to_path_buf(),
        fallback_threshold: 3,
    })
    .with_inference_factory(Box::new(mock_factory));
    let mut composite =
        CompositeEffect::new(segmentation, chains.mask, chains.bg, chains.fg, chains.post);
    composite
        .prepare(&ProcessingContext {
            width: 2,
            height: 2,
            format: PixelFormat::Rgb,
            fps: 30,
            counters: None,
        })
        .expect("prepare ok");
    composite
}

/// Run one distinctive 2×2 frame through `composite`; returns its bytes.
fn run_frame(composite: &mut CompositeEffect) -> Vec<u8> {
    let data = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let mut frame = VideoFrame::new_packed(
        FrameBuffer::Owned(data),
        2,
        2,
        PixelFormat::Rgb,
        FrameMeta::default(),
    )
    .expect("frame ok");
    composite
        .process(&mut frame, &mut FrameContext::default())
        .expect("process ok");
    frame.data.as_slice().to_vec()
}

/// Build the composite twice — with and without the effect under test —
/// and check that disabling the effect reproduces the "without" output,
/// that the effect was visible in the first place, and that re-enabling
/// brings it back.
fn assert_toggle(section: SubchainKind, name: &str, with: fn() -> Chains, without: fn() -> Chains) {
    let sidecar = ModelSidecar::new();
    let reference = run_frame(&mut prepared(&sidecar, without()));
    let mut composite = prepared(&sidecar, with());

    let on = run_frame(&mut composite);
    assert_ne!(on, reference, "{section}.{name} must change the frame");

    assert_eq!(
        composite.set_effect_enabled(section, name, false).ok(),
        Some(true)
    );
    assert_eq!(
        run_frame(&mut composite),
        reference,
        "disabled {section}.{name} must be bypassed"
    );

    assert_eq!(
        composite.set_effect_enabled(section, name, true).ok(),
        Some(true)
    );
    assert_eq!(run_frame(&mut composite), on, "re-enabled {section}.{name}");
}

#[test]
fn disabled_mask_effect_is_bypassed() {
    // `invert` alone is invisible (bg and fg are both the frame), so a
    // background fill makes the mask observable.
    assert_toggle(
        SubchainKind::Mask,
        "invert",
        || Chains {
            mask: vec![mask_fx("invert")],
            bg: vec![fill([1, 2, 3])],
            ..Chains::default()
        },
        || Chains {
            bg: vec![fill([1, 2, 3])],
            ..Chains::default()
        },
    );
}

#[test]
fn disabled_background_effect_is_bypassed() {
    assert_toggle(
        SubchainKind::Background,
        "color_fill",
        || Chains {
            bg: vec![fill([1, 2, 3])],
            ..Chains::default()
        },
        Chains::default,
    );
}

#[test]
fn disabled_foreground_effect_is_bypassed() {
    assert_toggle(
        SubchainKind::Foreground,
        "color_fill",
        || Chains {
            fg: vec![fill([1, 2, 3])],
            ..Chains::default()
        },
        Chains::default,
    );
}

#[test]
fn disabled_post_effect_is_bypassed() {
    assert_toggle(
        SubchainKind::Post,
        "mirror",
        || Chains {
            post: vec![post_fx("mirror")],
            ..Chains::default()
        },
        Chains::default,
    );
}

#[test]
fn every_duplicate_of_a_disabled_name_is_bypassed() {
    let sidecar = ModelSidecar::new();
    let reference = run_frame(&mut prepared(&sidecar, Chains::default()));
    let mut composite = prepared(
        &sidecar,
        Chains {
            bg: vec![fill([1, 2, 3]), fill([4, 5, 6])],
            ..Chains::default()
        },
    );
    composite
        .set_effect_enabled(SubchainKind::Background, "color_fill", false)
        .expect("toggle ok");
    assert_eq!(run_frame(&mut composite), reference);
}
