//! XDG base-directory resolution shared by the daemon and the GUI.
//!
//! Linux-only by construction: `$XDG_CONFIG_HOME` / `$HOME/.config`.

use std::ffi::OsStr;
use std::path::PathBuf;

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

#[cfg(test)]
mod tests {
    use super::*;

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
