//! Write GUI-modified presets back into the operator's TOML config.
//!
//! The control socket exposes [`Command::SavePreset`] and
//! [`Command::SavePresetAs`]; this module owns the file-side semantics
//! they delegate to:
//!
//! * **Format preservation.** Edits go through `toml_edit::DocumentMut`
//!   so comments, blank lines and the order of untouched sections are
//!   kept verbatim. Only the target `[presets.NAME]` sub-tree is
//!   replaced.
//! * **Atomic write.** The new document is staged in a sibling
//!   `*.tmp.<pid>` file (same directory, so the rename stays atomic on
//!   any sane filesystem), `fsync`'d, then `rename`'d over the target.
//!   A crash mid-write leaves either the old file intact or the new
//!   file fully visible — never a partial overwrite.
//! * **Bootstrap.** If the target file does not exist yet (operator
//!   ran the daemon without `--config` and never edited the XDG path),
//!   `save_preset` materialises an empty document plus the preset
//!   block. Parent directories are created with `create_dir_all`.
//!
//! Kept in `fluxframe-cli` (not `fluxframe-core`) so the workspace's
//! pure-data core stays free of file IO. The GUI never imports this
//! module; persistence is daemon-side single-writer by design.
//!
//! [`Command::SavePreset`]: fluxframe_core::Command::SavePreset
//! [`Command::SavePresetAs`]: fluxframe_core::Command::SavePresetAs

use std::fs;
use std::io::Write;
use std::path::Path;

use fluxframe_core::{FluxError, Preset};
use toml_edit::{DocumentMut, Item, Table};

/// Persist the supplied preset into `file` under the section
/// `[presets.<preset_name>]`. Creates the file (and parent directories)
/// when absent, otherwise replaces only the matching sub-tree and
/// leaves the rest of the document untouched.
///
/// # Errors
///
/// * [`FluxError::Io`] — failure to read the existing file, create
///   parent directories, write the staged file, `fsync`, or rename it
///   over the target.
/// * [`FluxError::Config`] — the existing file is not valid TOML, or
///   the preset itself cannot be serialised (e.g. a `per_effect` value
///   that toml cannot round-trip; extremely unlikely in normal
///   operation).
pub fn save_preset(file: &Path, preset_name: &str, preset: &Preset) -> Result<(), FluxError> {
    validate_preset_name(preset_name)?;
    let existing_text = if file.exists() {
        fs::read_to_string(file).map_err(|e| FluxError::io(file.to_path_buf(), e))?
    } else {
        // First-Save bootstrap — same header `load_or_init_document`
        // would have produced via the toml_edit path.
        "# fluxframe configuration\n# Generated on first Save from the GUI. \
         Edits made here are honoured by the daemon on reload.\n"
            .to_string()
    };
    // Strip any existing `[presets.NAME]` and `[presets.NAME.*]`
    // sub-tables out of the document body so the new preset block
    // can be appended as one contiguous section. toml_edit's
    // `Table::insert` reuses the original section's layout — which
    // leaves sub-tables interleaved with `[input]`/`[output]` when
    // the operator's hand-written file declared them between the
    // preset sub-headers. Plain string-level stripping avoids that
    // and gives the saved preset a stable, predictable shape.
    let stripped = strip_preset_sections(&existing_text, preset_name);
    let preset_text = serialise_preset_block(preset_name, preset)?;
    let mut combined = stripped;
    if !combined.is_empty() && !combined.ends_with('\n') {
        combined.push('\n');
    }
    if !combined.is_empty() && !combined.ends_with("\n\n") {
        combined.push('\n');
    }
    combined.push_str(&preset_text);
    if !combined.ends_with('\n') {
        combined.push('\n');
    }
    // Parse the result through toml_edit once to validate the
    // document and catch any whitespace/layout corruption before we
    // hit disk; toml_edit's parser is the same one `load` uses.
    let _check: DocumentMut =
        combined
            .parse()
            .map_err(|e: toml_edit::TomlError| FluxError::Config {
                reason: format!("internal: produced TOML did not re-parse: {e}"),
                hint: None,
            })?;
    atomic_write(file, combined.as_bytes())
}

/// Remove every TOML section whose header is `[presets.NAME]` or
/// `[presets.NAME.<anything>]` from `text`, along with any leading
/// blank/comment lines that decorate the section. Returns the
/// remaining document, preserving comments and tables that belong to
/// other top-level keys (`[input]`, `[output]`, other presets…).
fn strip_preset_sections(text: &str, preset_name: &str) -> String {
    // Pre-compute the two prefixes a section header may start with.
    let exact = format!("[presets.{preset_name}]");
    let dotted = format!("[presets.{preset_name}.");

    // Split the document into "section blocks": each block starts at
    // either the document head (prefix) or a top-level table header
    // and ends just before the next top-level table header. A "top-
    // level header" is a line that begins with `[` followed by an
    // alphanumeric / `_` / `-` / `.` / `"` (i.e. not a `[[…]]` array-
    // table; we do not currently use those).
    let mut blocks: Vec<&str> = Vec::new();
    let mut start = 0usize;
    for (idx, line) in text.match_indices('\n') {
        // `idx` points at the newline of the previous line. The
        // candidate next-section start is `idx + 1`. A header at
        // that position would start at the very beginning of the
        // line.
        let after_newline = idx + 1;
        if after_newline >= text.len() {
            continue;
        }
        let rest = &text[after_newline..];
        if rest.starts_with('[') {
            let _ = line;
            blocks.push(&text[start..after_newline]);
            start = after_newline;
        }
    }
    if start < text.len() {
        blocks.push(&text[start..]);
    }

    let mut kept: Vec<&str> = Vec::with_capacity(blocks.len());
    for block in blocks {
        // First non-blank, non-comment line of the block decides
        // whether it's the section we're stripping. Leading blank /
        // comment lines belong to the next section's trivia and are
        // dropped along with it (matching how toml_edit attributes
        // decor).
        let header_line = block
            .lines()
            .find(|l| {
                let t = l.trim_start();
                !t.is_empty() && !t.starts_with('#')
            })
            .unwrap_or("");
        let stripped_header = header_line.trim_start();
        let drop = stripped_header == exact || stripped_header.starts_with(&dotted);
        if !drop {
            kept.push(block);
        }
    }
    kept.concat()
}

/// Serialise a single `[presets.NAME]` block (parent + sub-tables)
/// as a self-contained TOML string. The returned text always starts
/// with `[presets.NAME]` (possibly empty body) followed by each
/// non-empty sub-section.
fn serialise_preset_block(preset_name: &str, preset: &Preset) -> Result<String, FluxError> {
    // Build a tiny throw-away document holding just `[presets.NAME]`
    // = preset. toml_edit's renderer then handles the section-header
    // composition (`[presets.NAME.background]` etc.) deterministically
    // and we strip the `[presets]` shell at the boundary.
    let mut doc = DocumentMut::new();
    let mut presets_tbl = Table::new();
    presets_tbl.set_implicit(true);
    presets_tbl.insert(preset_name, preset_to_item(preset)?);
    doc.insert("presets", Item::Table(presets_tbl));
    Ok(doc.to_string())
}

/// Persist `preset` under a new name. Fails if the name already exists
/// in `[presets]` — caller must pick a different name or call
/// [`save_preset`] explicitly to overwrite.
///
/// # Errors
///
/// In addition to the errors from [`save_preset`]:
///
/// * [`FluxError::Config`] — the requested name already exists, or
///   fails the same identifier validation as `[presets.NAME]` (must be
///   `[A-Za-z0-9_-]+`, non-empty).
pub fn save_preset_as(file: &Path, new_name: &str, preset: &Preset) -> Result<(), FluxError> {
    validate_preset_name(new_name)?;
    if file.exists() {
        let doc = read_document(file)?;
        if let Some(table) = doc.get("presets").and_then(Item::as_table)
            && table.contains_key(new_name)
        {
            return Err(FluxError::Config {
                reason: format!("preset '{new_name}' already exists"),
                hint: Some(
                    "pick a different name, or use the regular Save button to overwrite the active preset"
                        .into(),
                ),
            });
        }
    }
    save_preset(file, new_name, preset)
}

/// Preset name vocabulary: identifiers used both as `[presets.NAME]`
/// TOML headers and as the `name` argument in
/// [`fluxframe_core::Command::SetPreset`]. Constrained to keep
/// round-tripping through `toml_edit` predictable — anything that
/// would otherwise need bare-key quoting (`[presets."x y"]`) is
/// rejected so the document layout stays uniform.
fn validate_preset_name(name: &str) -> Result<(), FluxError> {
    if name.is_empty() {
        return Err(FluxError::Config {
            reason: "preset name is empty".into(),
            hint: Some(
                "preset names must be non-empty identifiers (letters, digits, _ or -)".into(),
            ),
        });
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(FluxError::Config {
            reason: format!("preset name '{name}' contains invalid characters"),
            hint: Some("use letters, digits, underscores or dashes only".into()),
        });
    }
    Ok(())
}

/// Read and parse `file` as a mutable TOML document. Used by
/// [`save_preset_as`] for the duplicate-name check; [`save_preset`]
/// itself does string-level stripping so it does not need a parsed
/// representation up front.
fn read_document(file: &Path) -> Result<DocumentMut, FluxError> {
    let text = fs::read_to_string(file).map_err(|e| FluxError::io(file.to_path_buf(), e))?;
    text.parse::<DocumentMut>().map_err(|e| FluxError::Config {
        reason: format!("config file is not valid TOML: {e}"),
        hint: Some(format!("path: {}", file.display())),
    })
}

/// Serialise a [`Preset`] to a `toml_edit::Item::Table` by
/// round-tripping through `toml::to_string` and re-parsing as a
/// `DocumentMut`. The round-trip is the only path that keeps
/// `#[serde(flatten)] per_effect` (and other adapters in
/// [`fluxframe_core::config`]) honest; constructing the
/// `toml_edit::Table` manually would duplicate the schema.
///
/// Before serialising, the preset is normalised: any `per_effect`
/// entry whose key is not in the section's `chain` (and is not a
/// reserved key) is dropped. This is a belt-and-braces guard against
/// in-memory drift — the runtime's `SetChain` handler already prunes,
/// but if a future code path produces an orphan, persistence cannot
/// be the layer that lets it reach disk and fail the composite
/// builder's `reject_unknown_table_keys` validation on the next load.
fn preset_to_item(preset: &Preset) -> Result<Item, FluxError> {
    let normalised = normalise_preset(preset);
    let text = toml::to_string(&normalised).map_err(|e| FluxError::Config {
        reason: format!("serialise preset: {e}"),
        hint: None,
    })?;
    let parsed: DocumentMut = text.parse().map_err(|e| FluxError::Config {
        reason: format!("re-parse serialised preset: {e}"),
        hint: None,
    })?;
    let mut table = parsed.as_table().clone();
    // The synthesised preset is a named sub-table (`[presets.NAME]`),
    // so it must be explicit — toml_edit otherwise omits the section
    // header.
    table.set_implicit(false);
    Ok(Item::Table(table))
}

/// Atomically replace `path` with `content`: stage in a sibling temp
/// file in the same directory (so the rename stays inside one
/// filesystem and therefore atomic), `fsync`, then `rename`.
///
/// Parent directories are created on demand — the first Save into the
/// XDG default path is expected to bootstrap `~/.config/fluxframe/`.
fn atomic_write(path: &Path, content: &[u8]) -> Result<(), FluxError> {
    let parent = path.parent().ok_or_else(|| FluxError::Config {
        reason: format!("config path {} has no parent directory", path.display()),
        hint: None,
    })?;
    if !parent.as_os_str().is_empty() {
        fs::create_dir_all(parent).map_err(|e| FluxError::io(parent.to_path_buf(), e))?;
    }

    // Sibling temp filename keyed on PID so concurrent daemons (an
    // operator running two instances against the same config) cannot
    // race on the same staging path. The actual content overwrite is
    // still last-write-wins after rename.
    let tmp = staging_path(path);
    {
        let mut f = fs::File::create(&tmp).map_err(|e| FluxError::io(tmp.clone(), e))?;
        f.write_all(content)
            .map_err(|e| FluxError::io(tmp.clone(), e))?;
        f.sync_all().map_err(|e| FluxError::io(tmp.clone(), e))?;
    }
    // Best-effort cleanup of the staging file on rename failure — we
    // do not leave `.tmp.<pid>` debris under the operator's config
    // directory if the final rename trips on EXDEV / EACCES.
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(FluxError::io(path.to_path_buf(), e));
    }
    Ok(())
}

/// Drop `per_effect` entries that do not correspond to any chain
/// member (and are not one of the reserved keys `model` /
/// `model_config` / `fallback_threshold`). Operates on a clone so
/// the caller's in-memory state is untouched — the runtime owns
/// in-memory consistency via the `SetChain` handler; persistence
/// just enforces it at the on-disk boundary.
fn normalise_preset(preset: &Preset) -> Preset {
    let mut out = preset.clone();
    for slot in [
        out.mask.as_mut(),
        out.background.as_mut(),
        out.foreground.as_mut(),
        out.post.as_mut(),
    ]
    .into_iter()
    .flatten()
    {
        slot.per_effect.retain(|key, _| {
            slot.chain.iter().any(|n| n == key)
                || matches!(
                    key.as_str(),
                    "model" | "model_config" | "fallback_threshold"
                )
        });
    }
    out
}

fn staging_path(path: &Path) -> std::path::PathBuf {
    let mut name = path
        .file_name()
        .map(std::ffi::OsString::from)
        .unwrap_or_default();
    name.push(format!(".tmp.{}", std::process::id()));
    match path.parent() {
        Some(p) => p.join(name),
        None => std::path::PathBuf::from(name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxframe_core::PipelineSection;
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    fn sample_preset() -> Preset {
        let mut per_effect: BTreeMap<String, toml::Value> = BTreeMap::new();
        let mut blur_table = toml::Table::new();
        blur_table.insert("radius".into(), toml::Value::Integer(40));
        per_effect.insert("blur".into(), toml::Value::Table(blur_table));
        Preset {
            mask: None,
            background: Some(PipelineSection {
                chain: vec!["blur".into()],
                per_effect,
                ..PipelineSection::default()
            }),
            foreground: None,
            post: None,
        }
    }

    #[test]
    fn save_preset_into_missing_file_creates_parent_dirs() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("nested").join("fluxframe.toml");
        save_preset(&path, "default", &sample_preset()).expect("save into nested path");
        assert!(path.exists(), "config file was not created");
        let text = fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("[presets.default"),
            "preset header missing in:\n{text}"
        );
    }

    #[test]
    fn save_preset_preserves_unrelated_sections_and_comments() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cfg.toml");
        let initial = "\
# Top-level comment kept across saves.
[input]
device = \"/dev/video2\"  # inline comment

[presets.default]
[presets.default.background]
chain = [\"vignette\"]
[presets.default.background.vignette]
strength = 0.3
";
        fs::write(&path, initial).unwrap();
        save_preset(&path, "default", &sample_preset()).expect("overwrite default preset");
        let after = fs::read_to_string(&path).unwrap();
        assert!(
            after.contains("# Top-level comment kept across saves."),
            "leading comment lost:\n{after}"
        );
        assert!(
            after.contains("device = \"/dev/video2\""),
            "[input] table lost:\n{after}"
        );
        assert!(
            after.contains("# inline comment"),
            "inline comment lost:\n{after}"
        );
        // The new preset should be in place.
        assert!(after.contains("radius = 40"), "new value missing:\n{after}");
        // The old `vignette = 0.3` setting must be gone — Save
        // *replaces* the preset.
        assert!(
            !after.contains("strength = 0.3"),
            "stale preset data not replaced:\n{after}"
        );
    }

    #[test]
    fn save_preset_round_trips_through_load() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cfg.toml");
        save_preset(&path, "demo", &sample_preset()).unwrap();
        let cfg = fluxframe_core::FluxConfig::from_toml_str(&fs::read_to_string(&path).unwrap())
            .expect("re-parse saved file");
        let demo = cfg.presets.get("demo").expect("demo preset present");
        let bg = demo.background.as_ref().expect("background present");
        assert_eq!(bg.chain, vec!["blur".to_string()]);
        let blur_cfg = bg
            .per_effect
            .get("blur")
            .and_then(toml::Value::as_table)
            .unwrap();
        assert_eq!(
            blur_cfg.get("radius").and_then(toml::Value::as_integer),
            Some(40)
        );
    }

    #[test]
    fn save_preset_as_rejects_existing_name() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cfg.toml");
        save_preset(&path, "existing", &sample_preset()).unwrap();
        let err = save_preset_as(&path, "existing", &sample_preset())
            .expect_err("duplicate name should fail");
        match err {
            FluxError::Config { reason, .. } => assert!(
                reason.contains("existing"),
                "error should mention the conflicting name: {reason}"
            ),
            other => panic!("expected Config error, got {other}"),
        }
    }

    #[test]
    fn save_preset_as_appends_a_second_preset() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cfg.toml");
        save_preset(&path, "first", &sample_preset()).unwrap();
        save_preset_as(&path, "second", &sample_preset()).unwrap();
        let after = fs::read_to_string(&path).unwrap();
        assert!(after.contains("[presets.first"), "first preset gone");
        assert!(after.contains("[presets.second"), "second preset missing");
    }

    #[test]
    fn save_preset_rejects_invalid_name() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cfg.toml");
        let err =
            save_preset(&path, "has spaces", &sample_preset()).expect_err("space should fail");
        assert!(matches!(err, FluxError::Config { .. }));
        let err = save_preset(&path, "", &sample_preset()).expect_err("empty should fail");
        assert!(matches!(err, FluxError::Config { .. }));
    }

    /// Realistic main preset matching the operator's actual config:
    /// mask + background + post, each with several configured effects.
    fn realistic_main_preset() -> Preset {
        let mut mask_per_effect: BTreeMap<String, toml::Value> = BTreeMap::new();
        let mut feather = toml::Table::new();
        feather.insert("radius".into(), toml::Value::Integer(2));
        mask_per_effect.insert("feather".into(), toml::Value::Table(feather));
        let mut smooth = toml::Table::new();
        smooth.insert("factor".into(), toml::Value::Float(0.85));
        mask_per_effect.insert("smooth_temporal".into(), toml::Value::Table(smooth));
        let mut threshold = toml::Table::new();
        threshold.insert("level".into(), toml::Value::Float(0.8));
        mask_per_effect.insert("threshold".into(), toml::Value::Table(threshold));

        let mut bg_per_effect: BTreeMap<String, toml::Value> = BTreeMap::new();
        let mut blur = toml::Table::new();
        blur.insert("passes".into(), toml::Value::Integer(2));
        blur.insert("radius".into(), toml::Value::Integer(10));
        bg_per_effect.insert("blur".into(), toml::Value::Table(blur));
        let mut vignette = toml::Table::new();
        vignette.insert("strength".into(), toml::Value::Float(0.71));
        vignette.insert("inner_radius".into(), toml::Value::Float(0.4));
        bg_per_effect.insert("vignette".into(), toml::Value::Table(vignette));

        let mut post_per_effect: BTreeMap<String, toml::Value> = BTreeMap::new();
        let mut auto_frame = toml::Table::new();
        auto_frame.insert("threshold".into(), toml::Value::Float(0.43));
        auto_frame.insert("smoothing".into(), toml::Value::Float(0.97));
        auto_frame.insert("zoom_max".into(), toml::Value::Float(2.41));
        post_per_effect.insert("auto_frame".into(), toml::Value::Table(auto_frame));

        Preset {
            mask: Some(PipelineSection {
                chain: vec![
                    "smooth_temporal".into(),
                    "threshold".into(),
                    "largest_blob".into(),
                    "feather".into(),
                ],
                model: Some("./models/selfie_segmentation.onnx".into()),
                model_config: None,
                fallback_threshold: None,
                per_effect: mask_per_effect,
            }),
            background: Some(PipelineSection {
                chain: vec!["blur".into(), "vignette".into()],
                per_effect: bg_per_effect,
                ..PipelineSection::default()
            }),
            foreground: None,
            post: Some(PipelineSection {
                chain: vec!["auto_frame".into(), "mirror".into()],
                per_effect: post_per_effect,
                ..PipelineSection::default()
            }),
        }
    }

    /// The realistic main preset, when saved to a fresh file and
    /// reloaded as `FluxConfig`, produces a `Preset` that is
    /// structurally equal to what went in.
    #[test]
    fn realistic_preset_full_round_trip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cfg.toml");
        let original = realistic_main_preset();
        save_preset(&path, "main", &original).expect("save realistic preset");

        let text = fs::read_to_string(&path).unwrap();
        // Sanity: every chain effect of every section must have its
        // sub-table emitted under the right preset.section path. The
        // composite builder's `reject_unknown_table_keys` later checks
        // this in the reverse direction; here we pin the forward shape.
        for header in [
            "[presets.main.mask]",
            "[presets.main.background]",
            "[presets.main.post]",
        ] {
            assert!(
                text.contains(header),
                "missing top section header {header} in:\n{text}"
            );
        }

        let cfg = fluxframe_core::FluxConfig::from_toml_str(&text)
            .expect("saved file parses as FluxConfig");
        let reloaded = cfg.presets.get("main").expect("main preset present");
        assert_eq!(
            reloaded, &original,
            "round-trip changed the preset:\nbefore: {original:?}\nafter:  {reloaded:?}"
        );
    }

    /// Saving twice in a row (simulating GUI's
    /// "Save → tweak → Save again") leaves the document parseable and
    /// the second save's values authoritative. The text layout must
    /// not duplicate the `[presets.main]` header or grow unbounded.
    #[test]
    fn second_save_does_not_duplicate_sections() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cfg.toml");
        save_preset(&path, "main", &realistic_main_preset()).unwrap();
        // Tweak one value and save again.
        let mut p = realistic_main_preset();
        if let Some(bg) = p.background.as_mut() {
            let mut blur = toml::Table::new();
            blur.insert("passes".into(), toml::Value::Integer(4));
            blur.insert("radius".into(), toml::Value::Integer(20));
            bg.per_effect
                .insert("blur".into(), toml::Value::Table(blur));
        }
        save_preset(&path, "main", &p).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let mask_headers = text.matches("[presets.main.mask]").count();
        assert_eq!(
            mask_headers, 1,
            "duplicated `[presets.main.mask]` header:\n{text}"
        );
        let cfg = fluxframe_core::FluxConfig::from_toml_str(&text).unwrap();
        let bg = cfg
            .presets
            .get("main")
            .and_then(|p| p.background.as_ref())
            .expect("background present after second save");
        let blur = bg
            .per_effect
            .get("blur")
            .and_then(toml::Value::as_table)
            .unwrap();
        assert_eq!(
            blur.get("passes").and_then(toml::Value::as_integer),
            Some(4)
        );
    }

    /// Re-saving over a file that the operator placed inside an
    /// existing larger config (with `[input]`, `[output]`, comments
    /// between preset sub-sections) must NOT leave preset sub-tables
    /// fragmented across the file — the new save groups everything
    /// belonging to `[presets.NAME]` together. This is the regression
    /// test for the user's "main preset sub-sections interleaved with
    /// [input]/[output]" report.
    #[test]
    fn save_groups_preset_subsections_together() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cfg.toml");
        let initial = "\
[control]
enabled = true

[presets.main.mask]
chain = [\"smooth_temporal\"]
[presets.main.mask.smooth_temporal]
factor = 0.5

[input]
device = \"/dev/video0\"

[presets.main.background]
chain = [\"blur\"]
[presets.main.background.blur]
radius = 30

[output]
device = \"/dev/video10\"
";
        fs::write(&path, initial).unwrap();
        save_preset(&path, "main", &realistic_main_preset()).unwrap();
        let after = fs::read_to_string(&path).unwrap();

        // Find the byte offsets of [input], [output] and each
        // [presets.main.*] header. The contract: all preset.main
        // headers must lie on one contiguous side of every non-preset
        // root table — either entirely before or entirely after, but
        // never interleaved.
        let positions = |needle: &str| -> Vec<usize> {
            after.match_indices(needle).map(|(idx, _)| idx).collect()
        };
        let input_at = *positions("[input]").first().expect("[input] still present");
        let output_at = *positions("[output]")
            .first()
            .expect("[output] still present");
        let main_positions = positions("[presets.main");
        assert!(!main_positions.is_empty(), "main preset headers gone");

        let all_before_input = main_positions.iter().all(|&p| p < input_at);
        let all_after_output = main_positions.iter().all(|&p| p > output_at);
        let all_between = main_positions
            .iter()
            .all(|&p| p > input_at && p < output_at)
            || main_positions.iter().all(|&p| {
                // Allow the contiguous block to sit anywhere that
                // does not cut through [input] or [output] headers.
                p < input_at.min(output_at) || (p > input_at.max(output_at))
            });
        assert!(
            all_before_input || all_after_output || all_between,
            "preset.main sub-sections are interleaved with [input]/[output]:\n{after}"
        );

        // And the file must still be valid + round-trip-equal.
        let cfg = fluxframe_core::FluxConfig::from_toml_str(&after).unwrap();
        assert_eq!(cfg.presets.get("main"), Some(&realistic_main_preset()));
    }

    /// `per_effect` keys that have no matching chain entry must NOT
    /// reach the on-disk file — they would otherwise trip the
    /// composite builder's `reject_unknown_table_keys` validation on
    /// the next preset load. This is the regression test for the
    /// user's "[background.pixelate] sub-table has no matching entry
    /// in [background].chain" report.
    #[test]
    fn save_prunes_orphan_per_effect_entries() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cfg.toml");
        // Build a preset whose background has an orphan `pixelate`
        // sub-table — chain only contains "blur".
        let mut p = sample_preset();
        if let Some(bg) = p.background.as_mut() {
            let mut pixelate = toml::Table::new();
            pixelate.insert("block_size".into(), toml::Value::Integer(16));
            bg.per_effect
                .insert("pixelate".into(), toml::Value::Table(pixelate));
        }
        save_preset(&path, "main", &p).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(
            !text.contains("[presets.main.background.pixelate]"),
            "orphan sub-table reached disk:\n{text}"
        );
        // Re-load through FluxConfig and run the composite-builder
        // validation that the runtime uses on SetPreset — must not
        // raise.
        let cfg = fluxframe_core::FluxConfig::from_toml_str(&text).unwrap();
        let bg = cfg
            .presets
            .get("main")
            .and_then(|x| x.background.as_ref())
            .expect("bg present");
        assert!(
            !bg.per_effect.contains_key("pixelate"),
            "orphan reached the loaded Preset"
        );
    }

    /// A preset with empty `chain = []` and no `per_effect` survives
    /// a round-trip. Edge case for "I removed everything from the
    /// background" workflow.
    #[test]
    fn empty_section_round_trips() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cfg.toml");
        let p = Preset {
            background: Some(PipelineSection::default()),
            ..Preset::default()
        };
        save_preset(&path, "empty", &p).unwrap();
        let cfg =
            fluxframe_core::FluxConfig::from_toml_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let reloaded = cfg.presets.get("empty").unwrap();
        // Sub-tables that serde-deserialised back from the on-disk
        // file may be `None` (toml omits empty Options) — both
        // shapes count as "no background content".
        if let Some(bg) = &reloaded.background {
            assert!(bg.chain.is_empty());
            assert!(bg.per_effect.is_empty());
        }
    }

    /// A preset whose `mask.chain` references an effect with a
    /// configured sub-table must keep both halves in sync after a
    /// round-trip. The mask section is special because it also
    /// carries `model = "/path"` — pruning must not drop that.
    #[test]
    fn mask_model_path_survives_round_trip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cfg.toml");
        let mut mask_per_effect: BTreeMap<String, toml::Value> = BTreeMap::new();
        let mut threshold = toml::Table::new();
        threshold.insert("level".into(), toml::Value::Float(0.7));
        mask_per_effect.insert("threshold".into(), toml::Value::Table(threshold));
        let preset = Preset {
            mask: Some(PipelineSection {
                chain: vec!["threshold".into()],
                model: Some("./models/selfie.onnx".into()),
                model_config: Some("./models/selfie.toml".into()),
                fallback_threshold: Some(3),
                per_effect: mask_per_effect,
            }),
            ..Preset::default()
        };
        save_preset(&path, "demo", &preset).unwrap();
        let cfg =
            fluxframe_core::FluxConfig::from_toml_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let reloaded = cfg.presets.get("demo").unwrap();
        let mask = reloaded.mask.as_ref().expect("mask present");
        assert_eq!(
            mask.model.as_deref().and_then(std::path::Path::to_str),
            Some("./models/selfie.onnx")
        );
        assert_eq!(mask.fallback_threshold, Some(3));
    }

    /// Stripping must not corrupt a file that contains another
    /// preset (or other non-`presets` tables) — only the target
    /// preset's sub-sections vanish.
    #[test]
    fn strip_leaves_other_presets_intact() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cfg.toml");
        let initial = "\
[presets.main]
[presets.main.background]
chain = [\"blur\"]
[presets.main.background.blur]
radius = 5

[presets.other]
[presets.other.background]
chain = [\"vignette\"]
[presets.other.background.vignette]
strength = 0.2
";
        fs::write(&path, initial).unwrap();
        save_preset(&path, "main", &sample_preset()).unwrap();
        let after = fs::read_to_string(&path).unwrap();
        // `other` survives in full.
        assert!(after.contains("[presets.other.background]"));
        assert!(after.contains("strength = 0.2"));
        // `main` reflects the new preset.
        let cfg = fluxframe_core::FluxConfig::from_toml_str(&after).unwrap();
        assert_eq!(
            cfg.presets
                .get("main")
                .and_then(|p| p.background.as_ref())
                .map(|s| s.chain.clone()),
            Some(vec!["blur".to_string()])
        );
        // And `other` is preserved as data, not just text.
        assert_eq!(
            cfg.presets
                .get("other")
                .and_then(|p| p.background.as_ref())
                .map(|s| s.chain.clone()),
            Some(vec!["vignette".to_string()])
        );
    }

    /// `strip_preset_sections` should NOT match a preset whose name
    /// is a prefix of the target (`main` vs `mainline`). The dotted
    /// boundary is what separates the two — without it a preset
    /// named `mainline` would be silently deleted when saving `main`.
    #[test]
    fn strip_does_not_match_prefix_names() {
        let initial = "\
[presets.main]
[presets.main.background]
chain = [\"blur\"]

[presets.mainline]
[presets.mainline.background]
chain = [\"vignette\"]
";
        let stripped = strip_preset_sections(initial, "main");
        assert!(
            !stripped.contains("[presets.main.background]"),
            "main not stripped:\n{stripped}"
        );
        // The `[presets.main]` parent header (no dot suffix) is also
        // exactly the target — stripped too.
        assert!(
            !stripped.contains("[presets.main]\n"),
            "main parent not stripped:\n{stripped}"
        );
        assert!(
            stripped.contains("[presets.mainline]"),
            "mainline lost:\n{stripped}"
        );
        assert!(
            stripped.contains("[presets.mainline.background]"),
            "mainline.background lost:\n{stripped}"
        );
    }

    #[test]
    fn atomic_write_leaves_no_tmp_file_on_success() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("cfg.toml");
        save_preset(&path, "default", &sample_preset()).unwrap();
        let entries: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        for name in &entries {
            assert!(
                !name.contains(".tmp."),
                "found leftover staging file {name} in {entries:?}"
            );
        }
    }
}
