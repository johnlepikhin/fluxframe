//! Core types, traits and error model for FluxFrame.
//!
//! This crate is the architectural hub: it defines the abstractions
//! (`VideoFrame`, `VideoEffect`, `InferenceEngine`, `VideoSource`, `VideoSink`)
//! that every other crate in the workspace depends on, and it has zero
//! runtime dependencies on GStreamer, ONNX Runtime, or V4L2.  This isolation
//! is intentional — see `doc/plan/stage-0-core-scaffolding.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod config;
pub mod context;
pub mod error;
pub mod frame;
pub mod metrics;
pub mod traits;

pub use config::{
    BackendKind, EffectsConfig, FluxConfig, InputConfig, LoggingConfig, OutputConfig,
    RealtimeConfig,
};
pub use context::{FrameContext, ProcessingContext, RuntimeState};
pub use error::{Diagnostic, EffectError, FluxError, InferenceError, PipelineError};
pub use frame::{FrameBuffer, FrameMeta, PixelFormat, Timestamp, VideoFrame};
pub use metrics::{
    CounterValues, Counters, LatencyHistogram, LatencySnapshot, MetricsSnapshot,
};
pub use traits::{
    InferenceEngine, InferenceInput, InferenceOutput, ModelInfo, RawEffectParams, VideoEffect,
    VideoSink, VideoSource,
};

/// Normalise an effect name from CLI form (`background-blur`) to internal
/// registry form (`background_blur`).
///
/// This is the single point of truth for the kebab↔snake mapping required
/// by §14 of the spec, so CLI parsing and config validation never disagree.
#[must_use]
pub fn normalise_effect_name(name: &str) -> String {
    name.replace('-', "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kebab_to_snake_effect_name() {
        assert_eq!(normalise_effect_name("background-blur"), "background_blur");
        assert_eq!(normalise_effect_name("passthrough"), "passthrough");
        assert_eq!(normalise_effect_name("auto-crop"), "auto_crop");
        assert_eq!(
            normalise_effect_name("already_snake"),
            "already_snake",
            "snake input must be preserved"
        );
    }

    #[test]
    fn config_parses_minimal_toml() {
        let cfg = FluxConfig::from_toml_str(
            r#"
[input]
device = "/dev/video1"

[effects]
chain = ["passthrough"]
"#,
        )
        .expect("minimal toml parses");

        assert_eq!(cfg.input.device, "/dev/video1");
        assert_eq!(cfg.input.width, 1280, "missing width falls back to default");
        assert_eq!(
            cfg.input.format,
            PixelFormat::Rgb,
            "missing format falls back to RGB"
        );
        assert_eq!(cfg.effects.chain, vec!["passthrough".to_string()]);
    }

    #[test]
    fn config_validation_rejects_zero_dimensions() {
        let mut cfg = FluxConfig::default();
        cfg.input.width = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validation_rejects_oversized_width() {
        let mut cfg = FluxConfig::default();
        cfg.input.width = 30_000;
        let err = cfg
            .validate()
            .expect_err("oversized width must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("16384"),
            "error must mention the upper bound, got: {msg}"
        );
    }

    #[test]
    fn config_validation_rejects_excessive_fps() {
        let mut cfg = FluxConfig::default();
        cfg.input.fps = 999;
        let err = cfg.validate().expect_err("excessive fps must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("240"),
            "error must mention the fps upper bound, got: {msg}"
        );
    }

    #[test]
    fn config_validation_rejects_unbounded_inflight() {
        let mut cfg = FluxConfig::default();
        cfg.realtime.max_inflight_frames = 100;
        let err = cfg
            .validate()
            .expect_err("oversized max_inflight_frames must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("max_inflight_frames"),
            "error must mention the offending field, got: {msg}"
        );
    }

    #[test]
    fn config_validation_accepts_defaults() {
        FluxConfig::default()
            .validate()
            .expect("default config must validate cleanly");
    }

    #[test]
    fn config_rejects_unknown_field() {
        let err = FluxConfig::from_toml_str(
            r#"
[input]
foo = "bar"
"#,
        )
        .expect_err("unknown field must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("foo"),
            "error must mention the unknown field, got: {msg}"
        );
    }

    #[test]
    fn config_format_parses_as_enum() {
        let cfg = FluxConfig::from_toml_str(
            r#"
[input]
format = "YUY2"
"#,
        )
        .expect("YUY2 format parses");
        assert_eq!(cfg.input.format, PixelFormat::Yuy2);
    }

    #[test]
    fn pixel_format_serde_roundtrip_all_variants() {
        use crate::config::InputConfig;
        for fmt in [
            PixelFormat::Rgb,
            PixelFormat::Rgba,
            PixelFormat::Bgr,
            PixelFormat::Yuy2,
            PixelFormat::Nv12,
            PixelFormat::Gray8,
        ] {
            let cfg = InputConfig {
                format: fmt,
                ..InputConfig::default()
            };
            let serialised = toml::to_string(&cfg).expect("serialise");
            let parsed: InputConfig = toml::from_str(&serialised).expect("parse");
            assert_eq!(parsed.format, fmt, "roundtrip failed for {fmt:?}");
        }
    }
}
