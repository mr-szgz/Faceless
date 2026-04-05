# Changelog

All notable changes to this project will be documented in this file.

> [!WARNING] REQUIRED AGENT GUIDANCE
> 
> - Read the changelog specification first at [references/keepachangelog.com.md](references/keepachangelog.com.md).
> - Follow [AGENTS.md](AGENTS.md) before making repository changes.
> - Keep format aligned with [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
> - Keep versioning aligned with [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-04-05
### Added
- Add the `run-faceless` Windows launcher with `--install`/`--uninstall` helpers for the Explorer context menu.

### Changed
- Group non-matching files into label-named subfolders by default; remove the `-Auto` flag.
- Refresh example assets and move archived reference material under `references/`.

### Removed
- Retire legacy installer and regression helper scripts.

## [0.4.2] - 2026-03-29
### Fixed
- Use a single user-home cache location (`~/.faceless/models`) for YOLO model downloads to prevent cwd-based re-downloads.

## [0.4.0] - 2026-03-29
### Added
- Write `main.log` and `error.log` under userdata and expose CLI log inspection commands.
- Require Python `3.10+` for installation and runtime.

### Fixed
- Resolve source paths with glob metacharacters so YOLO discovery runs reliably.
- Map `labels/imageN.txt` outputs back to source filenames before moving files.
- Escape glob metacharacters in PowerShell label invocation source paths.
- Render context-menu registry output with the correct executable path.

## [0.3.1] - 2026-03-29
### Added
- Publish a user-focused `README.md` with model-card-style sections.
- Document release-first usage with `faceless.exe` commands.

## [0.3.0] - 2026-03-29
### Added
- Accept `-Force` and `--force` as aliases for `-Label` and `--label`.
- Add Explorer context menu entries to run Faceless with `-Auto -Force`.

## [0.2.2] - 2026-03-29
### Added
- Prompt to open the source folder after each run, then optionally open `labels`.

### Changed
- Resolve required girl/woman and face classes from labels or model metadata before fallback IDs.

### Fixed
- Parse YOLO label files correctly when UTF-8 BOM or integer-like class tokens are present.

## [0.2.1] - 2026-03-28
### Changed
- Align packaged version and release tag at `0.2.1`.

## [0.2.0] - 2026-03-28
### Added
- Route non-matching media into label-based subfolders under the configured output directory.
- Route non-matching media into group-based subfolders using prioritized YAML group definitions.
- Enforce mutual exclusivity between `-Auto` and `-Group`.

### Fixed
- Resolve `-Auto` folder names from YAML mappings or model metadata instead of `class_<id>` fallbacks.
