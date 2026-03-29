# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-03-29
### Added
- Added `-Force`/`--force` as aliases for `-Label`/`--label` to force YOLO label regeneration.
- Added Windows Explorer context menu entries for `Run Faceless (Force Labels)` and `Run Faceless Here (Force Labels)` using the full `faceless.exe` path with `-Auto -Force`.

## [0.2.2] - 2026-03-29
### Added
- Added an interactive post-run prompt flow to optionally open the source input directory in File Explorer, and when declined, optionally open the `labels` folder (`[Y/n]` then `[y/N]`).
- Added `.agents/TECH_ARCH` reference architecture assets and examples to document reusable packaging and release workflow patterns.

### Changed
- Improved required-class detection by resolving girl/woman and face class IDs from label names/model metadata before falling back to hard-coded IDs.
- Updated release instructions to require running clean and build scripts before creating or pushing release tags, preventing stale artifacts from being published.

### Fixed
- Improved YOLO label parsing robustness for UTF-8 BOM and integer-like numeric class tokens in label files.

## [0.2.1] - 2026-03-28
### Changed
- Maintenance release to align packaged version and release tag at `0.2.1`.

## [0.2.0] - 2026-03-28
### Added
- Added `-Auto`/`--auto`/`-a` to route non-matching media into label-based subfolders under the configured `-Directory`/`--directory` root (default `noface`), while unlabeled items are moved to the directory root.
- Added `-Group`/`--group`/`-g` to route non-matching media into group-based subfolders under `-Directory` using prioritized YAML definitions in `faceless/labels/*.yaml` (for example `1_People`, `2_Clothing`).
- Added an argparse mutually exclusive mode for `-Auto` and `-Group` so only one output augmenter can be selected per run.
- Added `scripts/Invoke-Build-Faceless-Wheels.ps1` to build wheel/sdist artifacts for distribution with optional `cibuildwheel` support.
- Updated `scripts/Invoke-Build-Faceless-Wheels.ps1` and `mise run build` to prefer the project-local `.venv` interpreter by default.
- Added `scripts/Invoke-Clean-Faceless-Build.ps1` to remove build outputs and common Python build/cache artifacts.
- Added GitHub Actions workflows for CI packaging checks and release publishing with wheel/sdist attachments.
- Added `.github/instructions` agent guidance for release cutting and semantic version bump decisions.
- Added `scripts/Invoke-Bump-Faceless-Version.ps1` to recommend/apply semantic version bumps from the changelog.

### Fixed
- Fixed `-Auto` label name resolution so class folder names no longer degrade to `class_<id>` when YAML mapping files are unavailable; missing IDs now resolve from model metadata at runtime.
