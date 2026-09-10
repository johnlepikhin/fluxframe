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
    AutoInputConfig, BackendKind, ControlConfig, FluxConfig, IdleConfig, IdlePlaceholderKind,
    IdlePresenceSource, InputConfig, InputDevice, LoggingConfig, OutputConfig, OutputScale,
    PipelineSection, Preset, RealtimeConfig,
};
pub use context::{FrameContext, ProcessingContext, RuntimeState};
pub use error::{Diagnostic, EffectError, FluxError, InferenceError, PipelineError};
pub use frame::{FrameBuffer, FrameMeta, PixelFormat, Timestamp, VideoFrame};
pub use metadata::{CommitStrategy, EffectMetadata, ParamDescriptor, ParamKind, Scale};
pub use metrics::{
    CONSUMER_EVENT_AGE_UNSET, CounterValues, Counters, EXTERNAL_OPENERS_UNSET, EffectTelemetry,
    LatencyHistogram, LatencyRing, LatencySnapshot, MetricsSnapshot, OUTPUT_STREAM_UNKNOWN,
    PeriodicExtras, StageKey, StageSnapshot, StageTimings, emit_metrics_line, format_stage_summary,
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
    fn idle_default_is_off_and_validates() {
        let cfg = FluxConfig::default();
        assert!(!cfg.idle.enabled, "Stage 15 default is opt-in");
        assert!(cfg.idle.is_off());
        cfg.validate()
            .expect("default IdleConfig must validate cleanly");
    }

    #[test]
    fn idle_validation_rejects_zero_fps() {
        let mut cfg = FluxConfig::default();
        cfg.idle.fps = 0;
        let err = cfg.validate().expect_err("fps=0 must be rejected");
        assert!(format!("{err}").contains("idle.fps"));
    }

    #[test]
    fn idle_validation_rejects_excessive_fps() {
        let mut cfg = FluxConfig::default();
        cfg.idle.fps = 200;
        let err = cfg.validate().expect_err("fps>60 must be rejected");
        assert!(format!("{err}").contains("60"));
    }

    #[test]
    fn idle_validation_accepts_deep_le_teardown() {
        // DeepIdle removed (Stage 16): the deep > teardown cross-check
        // is gone, so deep_idle_secs == teardown_secs now loads fine.
        let mut cfg = FluxConfig::default();
        cfg.idle.teardown_secs = 10;
        cfg.idle.deep_idle_secs = 10;
        cfg.validate()
            .expect("deep_idle_secs == teardown_secs must be accepted after DeepIdle removal");
    }

    #[test]
    fn idle_validation_rejects_excessive_min_visibility_fps() {
        let mut cfg = FluxConfig::default();
        cfg.idle.min_visibility_fps = 200;
        let err = cfg
            .validate()
            .expect_err("min_visibility_fps>60 must be rejected");
        assert!(format!("{err}").contains("min_visibility_fps"));
    }

    #[test]
    fn input_validation_rejects_zero_acquire_backoff_base() {
        let mut cfg = FluxConfig::default();
        cfg.input.acquire_backoff_base_ms = 0;
        let err = cfg
            .validate()
            .expect_err("acquire_backoff_base_ms=0 must be rejected");
        assert!(format!("{err}").contains("acquire_backoff_base_ms"));
    }

    #[test]
    fn input_validation_rejects_max_below_base() {
        let mut cfg = FluxConfig::default();
        cfg.input.acquire_backoff_base_ms = 2000;
        cfg.input.acquire_backoff_max_ms = 1000;
        let err = cfg.validate().expect_err("max < base must be rejected");
        assert!(format!("{err}").contains("acquire_backoff_max_ms"));
    }

    #[test]
    fn input_validation_accepts_default_backoff() {
        // Defaults (500/5000) must validate cleanly.
        FluxConfig::default()
            .validate()
            .expect("default backoff knobs must validate");
    }

    #[test]
    fn idle_validation_rejects_zero_deep_idle_secs() {
        let mut cfg = FluxConfig::default();
        cfg.idle.deep_idle_secs = 0;
        let err = cfg
            .validate()
            .expect_err("deep_idle_secs=0 must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("deep_idle_secs"),
            "error must mention the field, got: {msg}"
        );
        assert!(
            msg.contains("> 0") || msg.contains("must be > 0"),
            "error must point at the > 0 requirement, got: {msg}"
        );
    }

    #[test]
    fn idle_validation_rejects_poll_interval_below_min() {
        let mut cfg = FluxConfig::default();
        cfg.idle.poll_interval_ms = 50;
        let err = cfg
            .validate()
            .expect_err("poll_interval_ms<100 must be rejected");
        assert!(format!("{err}").contains("poll_interval_ms"));
    }

    #[test]
    fn idle_validation_requires_path_when_image_kind() {
        let mut cfg = FluxConfig::default();
        cfg.idle.placeholder = config::IdlePlaceholderKind::Image;
        let err = cfg
            .validate()
            .expect_err("image kind without path must be rejected");
        assert!(format!("{err}").contains("placeholder_path"));
    }

    #[test]
    fn idle_serializes_only_when_enabled() {
        // Disabled (default) → no `idle` key in the serialised TOML.
        let cfg = FluxConfig::default();
        let dumped = toml::to_string(&cfg).expect("serialise default");
        assert!(
            !dumped.contains("[idle]"),
            "default config must not emit an [idle] table; got:\n{dumped}"
        );

        // Enabled → `idle` table is emitted.
        let mut cfg = FluxConfig::default();
        cfg.idle.enabled = true;
        let dumped = toml::to_string(&cfg).expect("serialise enabled");
        assert!(
            dumped.contains("[idle]"),
            "enabled config must emit an [idle] table; got:\n{dumped}"
        );
        assert!(
            dumped.contains("enabled = true"),
            "enabled flag must be emitted"
        );
    }

    #[test]
    fn idle_round_trips_through_toml() {
        let toml_text = r#"
[idle]
enabled = true
placeholder = "color"
placeholder_rgb = [200, 30, 30]
fps = 5
teardown_secs = 3
deep_idle_secs = 20
poll_interval_ms = 500
"#;
        let cfg = FluxConfig::from_toml_str(toml_text).expect("parses");
        cfg.validate().expect("validates");
        assert!(cfg.idle.enabled);
        assert_eq!(cfg.idle.placeholder_rgb, [200, 30, 30]);
        assert_eq!(cfg.idle.fps, 5);
        assert_eq!(cfg.idle.teardown_secs, 3);
        assert_eq!(cfg.idle.deep_idle_secs, 20);
        assert_eq!(cfg.idle.poll_interval_ms, 500);
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
