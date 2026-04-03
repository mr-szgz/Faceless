# Faceless

Faceless is a CLI tool. Replace this description with a short project summary.

## Install

1. pip install <!-- .whl not found -->
2. download [run-faceless.exe](<!-- run-faceless.exe not found -->) from release
3. run-faceless.exe --install

## Usage

**Example Command**

```sh
faceless "p:/path/to/resource" --option
```

```sh
$ faceless --help # or from source: uv run python -m faceless --help

usage: faceless [-h] [-Path PATH_OPTION] [-Label] [-Conf CONF] [-Match MATCH]
                [-Directory DIRECTORY] [-Auto] [--version]
                [path]

positional arguments:
  path                  Source directory containing images

options:
  -h, --help            show this help message and exit
  -Path PATH_OPTION, --path PATH_OPTION
                        Source directory containing images
  -Label, --label, -Force, --force
                        Force regeneration of labels
  -Conf CONF, --conf CONF
                        Model confidence threshold
  -Match MATCH, --match MATCH
                        Required match class IDs, comma-separated (example:
                        "1,43,51"). Default: "216,594".
  -Directory DIRECTORY, --directory DIRECTORY
                        Output directory name for moved files (default:
                        faceless)
  -Auto, --auto, -a     Move non-matching files into per-label folders under
                        the output directory
  --version, -Version, -v
                        show program's version number and exit
```

# Changelog

All notable changes to this project will be documented in this file.

> [!WARNING] REQUIRED AGENT GUIDANCE
> 
> - Read the changelog specification first at [references/keepachangelog.com.md](references/keepachangelog.com.md).
> - Follow [AGENTS.md](AGENTS.md) before making repository changes.
> - Keep format aligned with [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
> - Keep versioning aligned with [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-04-03
### Changed
- BREAKING: use `run-faceless.exe` as the dedicated Windows entrypoint instead of `faceless.exe`.

### Added
- Add the `run-faceless` launcher module and PyInstaller spec.
- Add a dedicated build helper script for the launcher executable.

## [0.5.1] - 2026-04-03
### Added
- Generate README.md from the release template.

## [0.5.0] - 2026-04-03
### Fixed
- Respect YOLO per-frame video labels so videos are only moved when they truly miss required classes.

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
