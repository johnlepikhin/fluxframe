//! Backend selection overrides (Stage 6 infrastructure).
//!
//! Reads `FLUXFRAME_FORCE_INFERENCE_BACKEND` and
//! `FLUXFRAME_FORCE_BLUR_BACKEND` from the environment.  These are
//! debug knobs — there is intentionally no user-facing TOML key for
//! them (Stage 6 only exposes CPU, and a user-facing override would
//! be misleading until a real GPU backend lands).
//!
//! The parsed [`BackendOverrides`] is cached once per process via
//! [`BackendOverrides::current`].  Tests that need to assert on the
//! parsing logic should call [`BackendOverrides::from_lookup`] with a
//! synthetic env-lookup closure instead of mutating the real
//! environment (which would race with parallel `cargo test` workers).
//!
//! Recognised values, case-insensitive:
//!
//! | env value           | meaning                                  |
//! |---------------------|------------------------------------------|
//! | `auto`, unset, `""` | default — let the factory autodetect     |
//! | `cpu`               | force the CPU implementation             |
//!
//! Unrecognised values fall back to `Auto` and emit a `warn!`.

use std::sync::OnceLock;

use tracing::warn;

/// Inference-backend override.
///
/// `Auto` (the default) lets [`crate::backend`] factories pick the
/// best available candidate.  `Cpu` forces the ONNX-Runtime CPU
/// session even when a GPU candidate would otherwise be selected.
/// `OpenVino` (Stage 8) routes through the OpenVINO runtime —
/// requires the `fluxframe-effects/openvino` Cargo feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum InferenceBackendChoice {
    /// Let the factory autodetect.
    #[default]
    Auto,
    /// Force the CPU ONNX-Runtime session.
    Cpu,
    /// Force the OpenVINO inference runtime.  If the host has no
    /// working OpenVINO stack the factory returns a hard error
    /// (`InferenceError::BackendUnavailable`) rather than silently
    /// falling back to ORT — same "explicit override never silently
    /// demotes" rule as the blur side
    /// (`BlurBackendChoice::Wgpu` in Stage 7).
    OpenVino,
}

/// Blur-backend override.  Same semantics as
/// [`InferenceBackendChoice`] but for the blur stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum BlurBackendChoice {
    /// Let the factory autodetect.
    #[default]
    Auto,
    /// Force the CPU box-blur implementation.
    Cpu,
    /// Force the `wgpu` (Vulkan compute) backend.  If the host has
    /// no working Vulkan stack the factory returns a hard error
    /// (`EffectError::PrepareFailed`) rather than silently falling
    /// back to CPU — see the rationale in `doc/plan/stage-7-wgpu-blur.md`.
    Wgpu,
}

/// Bundle of per-component backend overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct BackendOverrides {
    /// Inference-engine selection.
    pub inference: InferenceBackendChoice,
    /// Blur-backend selection.
    pub blur: BlurBackendChoice,
}

/// Env-var key for [`InferenceBackendChoice`].
pub const ENV_INFERENCE: &str = "FLUXFRAME_FORCE_INFERENCE_BACKEND";
/// Env-var key for [`BlurBackendChoice`].
pub const ENV_BLUR: &str = "FLUXFRAME_FORCE_BLUR_BACKEND";

impl BackendOverrides {
    /// Read overrides from the real process environment.
    ///
    /// Wraps [`std::env::var`] under [`Self::from_lookup`] so the
    /// inner parser stays test-friendly.
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    /// Builder: override the inference choice.  Provided because
    /// `BackendOverrides` is `#[non_exhaustive]` — downstream crates
    /// cannot use struct-literal syntax against it.
    #[must_use]
    pub fn with_inference(mut self, choice: InferenceBackendChoice) -> Self {
        self.inference = choice;
        self
    }

    /// Builder: override the blur choice.  See [`Self::with_inference`].
    #[must_use]
    pub fn with_blur(mut self, choice: BlurBackendChoice) -> Self {
        self.blur = choice;
        self
    }

    /// Read overrides through a caller-supplied lookup closure.
    /// Used by tests to inject synthetic env without mutating the
    /// real one.  `lookup` returns `None` for unset variables.
    #[must_use]
    pub fn from_lookup<F: Fn(&str) -> Option<String>>(lookup: F) -> Self {
        Self {
            inference: parse_inference(lookup(ENV_INFERENCE).as_deref()),
            blur: parse_blur(lookup(ENV_BLUR).as_deref()),
        }
    }

    /// Process-wide cached overrides.  Reads the environment on the
    /// first call and never again — env mutations after this point
    /// are intentionally ignored so the runtime sees a stable
    /// configuration for the entire session.
    #[must_use]
    pub fn current() -> Self {
        static OVERRIDES: OnceLock<BackendOverrides> = OnceLock::new();
        *OVERRIDES.get_or_init(Self::from_env)
    }
}

fn parse_inference(raw: Option<&str>) -> InferenceBackendChoice {
    match normalise(raw).as_deref() {
        None | Some("auto") => InferenceBackendChoice::Auto,
        Some("cpu") => InferenceBackendChoice::Cpu,
        Some("openvino") => InferenceBackendChoice::OpenVino,
        Some(other) => {
            warn!(
                key = ENV_INFERENCE,
                value = other,
                "unrecognised inference backend override; falling back to `auto`",
            );
            InferenceBackendChoice::Auto
        }
    }
}

fn parse_blur(raw: Option<&str>) -> BlurBackendChoice {
    match normalise(raw).as_deref() {
        None | Some("auto") => BlurBackendChoice::Auto,
        Some("cpu") => BlurBackendChoice::Cpu,
        Some("wgpu") => BlurBackendChoice::Wgpu,
        Some(other) => {
            warn!(
                key = ENV_BLUR,
                value = other,
                "unrecognised blur backend override; falling back to `auto`",
            );
            BlurBackendChoice::Auto
        }
    }
}

/// Lower-case and trim a raw env-var value.  Returns `None` for
/// unset or whitespace-only input so the caller can treat it
/// identically to "absent".
fn normalise(raw: Option<&str>) -> Option<String> {
    let trimmed = raw?.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_ascii_lowercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_when_unset() {
        let o = BackendOverrides::from_lookup(|_| None);
        assert_eq!(o.inference, InferenceBackendChoice::Auto);
        assert_eq!(o.blur, BlurBackendChoice::Auto);
    }

    #[test]
    fn empty_string_treated_as_unset() {
        let o = BackendOverrides::from_lookup(|_| Some(String::new()));
        assert_eq!(o.inference, InferenceBackendChoice::Auto);
        assert_eq!(o.blur, BlurBackendChoice::Auto);
    }

    #[test]
    fn whitespace_treated_as_unset() {
        let o = BackendOverrides::from_lookup(|_| Some("   ".into()));
        assert_eq!(o.inference, InferenceBackendChoice::Auto);
        assert_eq!(o.blur, BlurBackendChoice::Auto);
    }

    #[test]
    fn parses_cpu_override_per_component() {
        let o = BackendOverrides::from_lookup(|k| match k {
            ENV_INFERENCE => Some("cpu".into()),
            _ => None,
        });
        assert_eq!(o.inference, InferenceBackendChoice::Cpu);
        assert_eq!(o.blur, BlurBackendChoice::Auto);

        let o = BackendOverrides::from_lookup(|k| match k {
            ENV_BLUR => Some("cpu".into()),
            _ => None,
        });
        assert_eq!(o.inference, InferenceBackendChoice::Auto);
        assert_eq!(o.blur, BlurBackendChoice::Cpu);
    }

    #[test]
    fn parses_case_insensitive() {
        let o = BackendOverrides::from_lookup(|k| match k {
            ENV_INFERENCE => Some("CPU".into()),
            ENV_BLUR => Some("Cpu".into()),
            _ => None,
        });
        assert_eq!(o.inference, InferenceBackendChoice::Cpu);
        assert_eq!(o.blur, BlurBackendChoice::Cpu);
    }

    #[test]
    fn parses_blur_wgpu_override() {
        let o = BackendOverrides::from_lookup(|k| match k {
            ENV_BLUR => Some("wgpu".into()),
            _ => None,
        });
        assert_eq!(o.blur, BlurBackendChoice::Wgpu);
        // Case-insensitive.
        let o = BackendOverrides::from_lookup(|k| match k {
            ENV_BLUR => Some("WGPU".into()),
            _ => None,
        });
        assert_eq!(o.blur, BlurBackendChoice::Wgpu);
    }

    #[test]
    fn parses_auto_explicitly() {
        let o = BackendOverrides::from_lookup(|k| match k {
            ENV_INFERENCE => Some("auto".into()),
            ENV_BLUR => Some("AUTO".into()),
            _ => None,
        });
        assert_eq!(o.inference, InferenceBackendChoice::Auto);
        assert_eq!(o.blur, BlurBackendChoice::Auto);
    }

    #[test]
    fn unrecognised_falls_back_to_auto() {
        // The warning is emitted via `tracing::warn!`; we don't
        // capture log output here — only that the parse degrades to
        // `Auto` instead of panicking or returning an error.  Pick
        // strings that are NOT recognised by any current parser:
        // `tensorrt` is an ORT EP we don't support and `d3d12` is
        // wgpu's Windows-only backend that we never expose here.
        let o = BackendOverrides::from_lookup(|k| match k {
            ENV_INFERENCE => Some("tensorrt".into()),
            ENV_BLUR => Some("d3d12".into()),
            _ => None,
        });
        assert_eq!(o.inference, InferenceBackendChoice::Auto);
        assert_eq!(o.blur, BlurBackendChoice::Auto);
    }

    #[test]
    fn parses_inference_openvino_override() {
        let o = BackendOverrides::from_lookup(|k| match k {
            ENV_INFERENCE => Some("openvino".into()),
            _ => None,
        });
        assert_eq!(o.inference, InferenceBackendChoice::OpenVino);
        // Case-insensitive (handled by `normalise`).
        let o = BackendOverrides::from_lookup(|k| match k {
            ENV_INFERENCE => Some("OpenVINO".into()),
            _ => None,
        });
        assert_eq!(o.inference, InferenceBackendChoice::OpenVino);
    }

    #[test]
    fn current_is_idempotent() {
        // The contract is that `current()` reads env exactly once.
        // Calling it twice must yield the same value without
        // panicking — we can't easily assert "env was not re-read"
        // from the outside, but the OnceLock makes that guarantee
        // structurally.
        let a = BackendOverrides::current();
        let b = BackendOverrides::current();
        assert_eq!(a, b);
    }
}
