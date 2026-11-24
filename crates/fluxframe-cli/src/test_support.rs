//! Fixtures shared by the crate's unit tests.

use fluxframe_core::FluxConfig;

/// A model whose sidecar allows only a device no machine has, so
/// loading it always fails with `AcceleratorUnavailable`. The model
/// and sidecar live in a temporary directory removed on drop.
pub(crate) struct UnrunnableModel {
    dir: std::path::PathBuf,
}

impl UnrunnableModel {
    pub(crate) fn new(tag: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("fluxframe-fallback-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        std::fs::write(dir.join("m.onnx"), b"").expect("model");
        std::fs::write(
            dir.join("m.toml"),
            "name = \"m\"\ninput_width = 4\ninput_height = 4\ndevices = [\"NO_SUCH_DEVICE\"]\n",
        )
        .expect("sidecar");
        Self { dir }
    }

    /// Config with preset `accel` (the unrunnable model, optionally
    /// falling back to `fallback`) and an empty preset `plain`.
    pub(crate) fn config(&self, fallback: Option<&str>) -> FluxConfig {
        let fallback = fallback.map_or(String::new(), |f| format!("fallback_preset = \"{f}\"\n"));
        FluxConfig::from_toml_str(&format!(
            "[presets.accel]\n{fallback}[presets.accel.mask]\nmodel = \"{}\"\n[presets.plain]\n",
            self.dir.join("m.onnx").display()
        ))
        .expect("config parses")
    }
}

impl Drop for UnrunnableModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
