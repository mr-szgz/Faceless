# AGENTS.md

This file is the canonical in-repo guide for coding agents and developers working on Faceless.

## Project Purpose

Faceless is a CLI tool that uses YOLO object detection to separate image files that do not contain both:

- a girl/woman class
- a human face class

Files that fail this rule are moved into a destination folder (default: `noface`) under the source directory.

## Repository Layout

- `faceless/__init__.py`: Main CLI logic and processing loop.
- `faceless/__main__.py`: Module entry point (`python -m faceless ...`).
- `scripts/Invoke-Label-Faceless.ps1`: PowerShell helper for label generation with YOLO.
- `labels/`: Dataset and class-group metadata YAML files.
- `pyproject.toml`: Packaging metadata and Python dependencies.
- `faceless.spec`: PyInstaller one-file executable build configuration.

## Key Packages

Runtime packages:

- `ultralytics`: Runs YOLO inference and writes YOLO txt labels.
- `argparse`: Built-in CLI parser. Keep CLI parsing based on argparse built-ins.
- `pathlib` (stdlib): Path normalization and filesystem operations.
- `shutil` (stdlib): Moves unmatched files.

Packaging/dev packages:

- `setuptools`, `wheel`: Build backend.
- `pyinstaller`: Builds standalone Windows executable from `faceless.spec`.
- `build`: Optional package build tooling.

Notes:

- `pyproject.toml` currently requires Python `>=3.11`.
- `mise.toml` pins local toolchain to Python `3.13`.

## Runtime Behavior and Patterns

### CLI contract

Current parser in `faceless/__init__.py` supports:

- positional source path: `python -m faceless <path>`
- `-Path` / `--path`
- `-Label` / `--label`
- `-Conf` / `--conf`
- `-Directory` / `--directory`

### Label generation phase

Labels are generated when either condition is true:

- label folder does not exist at `<source>/labels`
- `-Label` is explicitly provided

Generation uses `YOLO(...).predict(...)` with project output set to source and label files stored in `<source>/labels`.

### Filtering/move phase

Per source file:

1. Read `<source>/labels/<stem>.txt` if present.
2. Detect whether both class constraints are present.
3. If both are not present, move the file to `<source>/<directory>`.

Current class constants in code:

- girl/woman classes: `216`, `594`
- human face class: `264`

## Change Guidelines

- Keep CLI parsing in `parse_arguments()` on argparse built-ins only.
- Preserve backward compatibility for existing PowerShell-style flags where practical.
- Use `Path(...).expanduser().resolve()` for user-provided paths.
- Prefer explicit, actionable parser errors for invalid/missing CLI input.
- Keep image-processing behavior deterministic; avoid adding hidden side effects.
- Keep edits minimal and focused. Avoid unrelated refactors.

## Validation Checklist

When modifying behavior, validate all of the following:

1. `python -m faceless <path>` works.
2. `python -m faceless -Path <path>` works.
3. `--help` reflects all options and defaults.
4. Labels are produced in `<source>/labels` when forced or missing.
5. Non-matching files move to the configured destination directory.

## Build and Execution

Install dependencies:

- `pip install -r requirements.txt`

Run as module:

- `python -m faceless "<path>"`

Run via script entry point:

- `faceless "<path>"`

Build executable:

- `pyinstaller faceless.spec`

## Known Operational Gotchas

- Environment mismatch can cause `No module named faceless` if the wrong Python interpreter is used.
- File moves are destructive to original folder layout; test against a copy of data when changing logic.
- The source directory may contain non-image files; current logic iterates all files and relies on label presence.
