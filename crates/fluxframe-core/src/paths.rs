//! XDG base-directory resolution shared by the daemon and the GUI.
//!
//! Linux-only by construction: `$XDG_CONFIG_HOME` / `$HOME/.config`.

use std::borrow::Cow;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Resolve the XDG config home from raw `$XDG_CONFIG_HOME` / `$HOME` values.
/// An empty `XDG_CONFIG_HOME` is ignored, per the XDG Base Directory spec.
#[must_use]
pub fn resolve_config_home(
    xdg_config_home: Option<&OsStr>,
    home: Option<&OsStr>,
) -> Option<PathBuf> {
    if let Some(xdg) = xdg_config_home.filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(xdg));
    }
    home.filter(|s| !s.is_empty())
        .map(|h| PathBuf::from(h).join(".config"))
}

/// [`resolve_config_home`] applied to the process environment.
#[must_use]
pub fn config_home() -> Option<PathBuf> {
    resolve_config_home(
        std::env::var_os("XDG_CONFIG_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}

/// Directory that relative paths in a configuration file resolve against.
///
/// A relative path in the configuration (a preset's model, an
/// `image_fill` image, the idle placeholder) means "next to the config
/// file", not "wherever the daemon happened to be started from".
/// Resolution happens where a file is opened; the configuration itself
/// keeps paths exactly as written, so saving a preset never rewrites
/// them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConfigBase(Option<PathBuf>);

impl ConfigBase {
    /// Base for the configuration file at `config_path`: its directory,
    /// made absolute against the current directory at the time of the
    /// call. Without a file (`None`) relative paths stay relative to the
    /// working directory.
    #[must_use]
    pub fn for_config(config_path: Option<&Path>) -> Self {
        Self(
            config_path
                .and_then(|path| std::path::absolute(path).ok())
                .and_then(|path| path.parent().map(Path::to_path_buf)),
        )
    }

    /// Resolve `path`: absolute paths are returned unchanged, relative
    /// ones are joined onto the base directory.
    #[must_use]
    pub fn resolve<'a>(&self, path: &'a Path) -> Cow<'a, Path> {
        match &self.0 {
            Some(dir) if path.is_relative() => Cow::Owned(dir.join(path)),
            _ => Cow::Borrowed(path),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_base_joins_relative_paths_onto_the_config_directory() {
        let base = ConfigBase::for_config(Some(Path::new("/etc/ff/fluxframe.toml")));
        assert_eq!(
            base.resolve(Path::new("models/m.onnx")),
            Path::new("/etc/ff/models/m.onnx")
        );
    }

    #[test]
    fn config_base_leaves_absolute_paths_alone() {
        let base = ConfigBase::for_config(Some(Path::new("/etc/ff/fluxframe.toml")));
        assert!(matches!(
            base.resolve(Path::new("/srv/m.onnx")),
            Cow::Borrowed(p) if p == Path::new("/srv/m.onnx")
        ));
    }

    #[test]
    fn config_base_without_a_file_keeps_paths_relative() {
        assert_eq!(ConfigBase::for_config(None), ConfigBase::default());
        assert_eq!(
            ConfigBase::default().resolve(Path::new("m.onnx")),
            Path::new("m.onnx")
        );
    }

    #[test]
    fn config_base_of_a_relative_config_path_is_absolute() {
        let base = ConfigBase::for_config(Some(Path::new("fluxframe.toml")));
        let resolved = base.resolve(Path::new("m.onnx"));
        assert!(resolved.is_absolute(), "got {}", resolved.display());
        assert!(resolved.ends_with("m.onnx"));
    }

    #[test]
    fn empty_xdg_falls_back_to_home_config() {
        assert_eq!(
            resolve_config_home(Some(OsStr::new("")), Some(OsStr::new("/home/u"))),
            Some(PathBuf::from("/home/u/.config"))
        );
    }

    #[test]
    fn non_empty_xdg_wins() {
        assert_eq!(
            resolve_config_home(Some(OsStr::new("/xdg")), Some(OsStr::new("/home/u"))),
            Some(PathBuf::from("/xdg"))
        );
    }

    #[test]
    fn both_absent_is_none() {
        assert_eq!(resolve_config_home(None, None), None);
    }

    #[test]
    fn empty_home_without_xdg_is_none() {
        assert_eq!(resolve_config_home(None, Some(OsStr::new(""))), None);
    }
}
