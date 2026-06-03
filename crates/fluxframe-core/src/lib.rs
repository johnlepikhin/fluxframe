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
pub mod metadata;
pub mod metrics;
pub mod plane;
pub mod protocol;
pub mod traits;

pub use config::{
    AutoInputConfig, BackendKind, ControlConfig, FluxConfig, InputConfig, InputDevice,
    LoggingConfig, OutputConfig, OutputScale, PipelineSection, Preset, RealtimeConfig,
};
pub use context::{FrameContext, ProcessingContext, RuntimeState};
pub use error::{Diagnostic, EffectError, FluxError, InferenceError, PipelineError};
pub use frame::{FrameBuffer, FrameMeta, PixelFormat, Timestamp, VideoFrame};
pub use metadata::{CommitStrategy, EffectMetadata, ParamDescriptor, ParamKind, Scale};
pub use metrics::{
    CounterValues, Counters, EffectTelemetry, LatencyHistogram, LatencySnapshot, MetricsSnapshot,
};
pub use plane::{FramePlane, MaskEffect, MaskPlane, PlaneEffect, PostEffect, SubchainKind};
pub use protocol::{Command, Response, SetPath, default_socket_path, parse_set_path};
pub use traits::{
    InferenceEngine, InferenceInput, InferenceOutput, ModelInfo, RawEffectParams, VideoEffect,
    VideoSink, VideoSource,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_parses_minimal_toml() {
        let cfg = FluxConfig::from_toml_str(
            r#"
[input]
device = "/dev/video1"
"#,
        )
        .expect("minimal toml parses");

        assert_eq!(
            cfg.input.device,
            InputDevice::Path(std::path::PathBuf::from("/dev/video1"))
        );
        assert_eq!(cfg.input.width, 1280, "missing width falls back to default");
        assert_eq!(
            cfg.input.format,
            PixelFormat::Rgb,
            "missing format falls back to RGB"
        );
        assert!(
            cfg.presets.is_empty(),
            "no [presets.*] in minimal toml → empty map"
        );
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
    fn config_validation_rejects_excessive_metrics_interval() {
        let mut cfg = FluxConfig::default();
        cfg.realtime.metrics_interval_secs = 50_000;
        let err = cfg
            .validate()
            .expect_err("oversized metrics_interval_secs must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("metrics_interval_secs"),
            "error must mention the offending field, got: {msg}"
        );
        assert!(
            msg.contains("3600"),
            "error must mention the upper bound, got: {msg}"
        );
    }

    #[test]
    fn config_validation_accepts_zero_metrics_interval_as_disabled() {
        let mut cfg = FluxConfig::default();
        cfg.realtime.metrics_interval_secs = 0;
        cfg.validate()
            .expect("0 must be accepted as the disabled sentinel");
    }

    #[test]
    fn output_scale_rejects_nonpositive() {
        let err = config::OutputScale::new(0.0).expect_err("zero must be rejected");
        assert!(format!("{err}").contains("output.scale"));
        let err = config::OutputScale::new(-0.5).expect_err("negative must be rejected");
        assert!(format!("{err}").contains("output.scale"));
    }

    #[test]
    fn output_scale_rejects_nan_and_infinity() {
        let err = config::OutputScale::new(f32::NAN).expect_err("NaN must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("NaN"), "NaN-specific message expected: {msg}");
        let err = config::OutputScale::new(f32::INFINITY).expect_err("+inf must be rejected");
        assert!(format!("{err}").contains("output.scale"));
        let err = config::OutputScale::new(f32::NEG_INFINITY).expect_err("-inf must be rejected");
        assert!(format!("{err}").contains("output.scale"));
    }

    #[test]
    fn output_scale_upper_bound_is_inclusive_at_one() {
        // 1.0 must pass (identity), 1.0 + ε must fail.
        config::OutputScale::new(1.0).expect("1.0 inclusive");
        let just_over = 1.0_f32 + f32::EPSILON * 4.0; // a few ULPs above 1.
        let err = config::OutputScale::new(just_over)
            .expect_err("anything strictly above 1.0 must be rejected");
        assert!(format!("{err}").contains("upscaling"));
    }

    #[test]
    fn output_scale_rejects_below_min() {
        let err = config::OutputScale::new(1e-30).expect_err("sub-min scale must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("0.05"), "min bound must be reported: {msg}");
    }

    #[test]
    fn output_scale_rejects_upscale() {
        let err = config::OutputScale::new(1.5).expect_err("upscale must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("upscaling"),
            "hint must mention upscaling: {msg}"
        );
    }

    #[test]
    fn output_scale_accepts_valid_range() {
        config::OutputScale::new(1.0).expect("1.0 = identity must pass");
        config::OutputScale::new(0.5).expect("0.5 must pass");
        config::OutputScale::new(0.05).expect("MIN_OUTPUT_SCALE must pass (inclusive)");
    }

    #[test]
    fn output_scale_default_is_identity() {
        let s = config::OutputScale::default();
        assert!((s.value() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn output_scale_deserialize_validates_at_parse_time() {
        // Valid value through TOML.
        let cfg = FluxConfig::from_toml_str(
            "
[output]
scale = 0.5
",
        )
        .expect("0.5 parses");
        assert!((cfg.output.scale.value() - 0.5).abs() < f32::EPSILON);

        // Out-of-range value rejected during parse, not at validate().
        let err = FluxConfig::from_toml_str(
            "
[output]
scale = 2.0
",
        )
        .expect_err("upscale must fail at parse time");
        let msg = format!("{err}");
        assert!(
            msg.contains("output.scale"),
            "serde error must mention output.scale: {msg}"
        );
    }

    #[test]
    fn output_effective_dimensions_scales_and_rounds_even() {
        let mut cfg = FluxConfig::default();
        cfg.output.scale = config::OutputScale::IDENTITY;
        assert_eq!(cfg.output.effective_dimensions(1280, 720), (1280, 720));
        cfg.output.scale = config::OutputScale::new(0.5).expect("0.5 valid");
        assert_eq!(cfg.output.effective_dimensions(1280, 720), (640, 360));
        // 2/3 of 1920×1080 = 1280×720 exactly — confirm fraction works.
        cfg.output.scale = config::OutputScale::new(2.0 / 3.0).expect("2/3 valid");
        assert_eq!(cfg.output.effective_dimensions(1920, 1080), (1280, 720));
        // Odd result rounded down to even.
        cfg.output.scale = config::OutputScale::new(0.5).expect("0.5 valid");
        assert_eq!(cfg.output.effective_dimensions(641, 481), (320, 240));
        // Minimum scale produces a small but valid even output.
        cfg.output.scale = config::OutputScale::new(0.05).expect("0.05 = MIN valid");
        let (w, h) = cfg.output.effective_dimensions(1280, 720);
        assert!(w >= 2 && h >= 2, "got {w}x{h}");
        assert!(w & 1 == 0 && h & 1 == 0, "must be even: {w}x{h}");
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
    fn input_device_serde_deserialises_auto_testsrc_path() {
        // `"auto"` (any case) → InputDevice::Auto
        let cfg = FluxConfig::from_toml_str(
            r#"
[input]
device = "auto"
"#,
        )
        .expect("auto parses");
        assert_eq!(cfg.input.device, InputDevice::Auto);

        let cfg = FluxConfig::from_toml_str(
            r#"
[input]
device = "AUTO"
"#,
        )
        .expect("AUTO parses case-insensitively");
        assert_eq!(cfg.input.device, InputDevice::Auto);

        // `"testsrc"` → InputDevice::Testsrc
        let cfg = FluxConfig::from_toml_str(
            r#"
[input]
device = "testsrc"
"#,
        )
        .expect("testsrc parses");
        assert_eq!(cfg.input.device, InputDevice::Testsrc);

        // anything else → InputDevice::Path
        let cfg = FluxConfig::from_toml_str(
            r#"
[input]
device = "/dev/video0"
"#,
        )
        .expect("explicit path parses");
        assert_eq!(
            cfg.input.device,
            InputDevice::Path(std::path::PathBuf::from("/dev/video0"))
        );

        // Display roundtrip: Auto/Testsrc/Path render as their TOML form.
        assert_eq!(InputDevice::Auto.to_string(), "auto");
        assert_eq!(InputDevice::Testsrc.to_string(), "testsrc");
        assert_eq!(
            InputDevice::Path(std::path::PathBuf::from("/dev/video0")).to_string(),
            "/dev/video0"
        );
    }

    #[test]
    fn auto_input_defaults_are_sensible() {
        let auto = AutoInputConfig::default();
        assert!(auto.poll_interval_secs >= 1);
        assert!(auto.exclude_devices.is_empty());
    }

    #[test]
    fn config_parses_input_auto_section() {
        let cfg = FluxConfig::from_toml_str(
            r#"
[input]
device = "auto"

[input.auto]
poll_interval_secs = 5
exclude_devices = ["/dev/video20"]
"#,
        )
        .expect("parses");
        assert_eq!(cfg.input.device, InputDevice::Auto);
        assert_eq!(cfg.input.auto.poll_interval_secs, 5);
        assert_eq!(
            cfg.input.auto.exclude_devices,
            vec![std::path::PathBuf::from("/dev/video20")]
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
    fn presets_parse_empty_map_when_section_absent() {
        // No `[presets.*]` in the TOML — the map is empty (a runtime
        // error at preset-resolution, not a parse error).
        let cfg = FluxConfig::from_toml_str(
            r#"
[input]
device = "/dev/video1"
"#,
        )
        .expect("parses without presets");
        assert!(
            cfg.presets.is_empty(),
            "missing [presets] table → empty map"
        );
    }

    #[test]
    fn presets_parse_named_subtables() {
        let cfg = FluxConfig::from_toml_str(
            r#"
[presets.default]

[presets.default.mask]
model = "./models/selfie_segmentation.onnx"
chain = ["threshold", "feather"]
fallback_threshold = 3

[presets.default.mask.threshold]
level = 0.5

[presets.default.background]
chain = ["blur"]

[presets.default.background.blur]
radius = 20

[presets.default.foreground]
chain = ["passthrough"]
"#,
        )
        .expect("named preset parses");

        let preset = cfg.presets.get("default").expect("default preset present");
        let mask = preset.mask.as_ref().expect("mask sub-section present");
        assert_eq!(
            mask.model.as_deref(),
            Some(std::path::Path::new("./models/selfie_segmentation.onnx"))
        );
        assert_eq!(
            mask.chain,
            vec!["threshold".to_string(), "feather".to_string()]
        );
        assert_eq!(mask.fallback_threshold, Some(3));
        let threshold = mask
            .per_effect
            .get("threshold")
            .and_then(toml::Value::as_table)
            .expect("threshold sub-table present");
        let level = threshold
            .get("level")
            .and_then(toml::Value::as_float)
            .expect("level field present");
        assert!((level - 0.5).abs() < 1e-6);

        let bg = preset.background.as_ref().expect("background present");
        assert_eq!(bg.chain, vec!["blur".to_string()]);

        let fg = preset.foreground.as_ref().expect("foreground present");
        assert_eq!(fg.chain, vec!["passthrough".to_string()]);
    }

    #[test]
    fn presets_parse_multiple_named_entries() {
        let cfg = FluxConfig::from_toml_str(
            r#"
[presets.blur]
[presets.blur.background]
chain = ["blur"]

[presets.green]
[presets.green.background]
chain = ["color_fill"]
[presets.green.background.color_fill]
rgb = [0, 255, 0]
"#,
        )
        .expect("two presets parse");
        assert_eq!(cfg.presets.len(), 2);
        assert!(cfg.presets.contains_key("blur"));
        assert!(cfg.presets.contains_key("green"));
    }

    #[test]
    fn preset_empty_subsections_are_all_none() {
        // A preset with no `mask`/`background`/`foreground` keys at
        // all — useful as the "raw passthrough" preset.
        let cfg = FluxConfig::from_toml_str(
            r"
[presets.raw]
",
        )
        .expect("empty preset parses");
        let preset = cfg.presets.get("raw").expect("raw preset present");
        assert!(preset.mask.is_none());
        assert!(preset.background.is_none());
        assert!(preset.foreground.is_none());
    }

    #[test]
    fn preset_rejects_unknown_top_level_field() {
        // `deny_unknown_fields` on `Preset` keeps typos like
        // `forground` (sic) from being silently ignored.
        let err = FluxConfig::from_toml_str(
            r#"
[presets.broken]
forground = "typo"
"#,
        )
        .expect_err("typo in preset section must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("forground") || msg.contains("unknown"),
            "diagnostic must point at the typo, got: {msg}"
        );
    }

    #[test]
    fn control_section_defaults_disabled() {
        // No [control] in TOML — section is present in the struct
        // through `#[serde(default)]` and disabled.
        let cfg = FluxConfig::from_toml_str(
            r#"
[input]
device = "testsrc"
"#,
        )
        .expect("parses without [control]");
        assert!(!cfg.control.enabled);
        assert!(cfg.control.socket_path.is_none());
    }

    #[test]
    fn control_section_parses_enabled_and_path() {
        let cfg = FluxConfig::from_toml_str(
            r#"
[control]
enabled = true
socket_path = "/run/user/1000/fluxframe.sock"
"#,
        )
        .expect("parses [control]");
        assert!(cfg.control.enabled);
        assert_eq!(
            cfg.control.socket_path.as_deref(),
            Some(std::path::Path::new("/run/user/1000/fluxframe.sock"))
        );
    }

    #[test]
    fn control_rejects_unknown_field() {
        let err = FluxConfig::from_toml_str(
            r"
[control]
enabled = true
bogus = 42
",
        )
        .expect_err("unknown [control] field must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("bogus"), "got: {msg}");
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
