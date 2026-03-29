# Faceless

Faceless is a CLI tool that uses YOLO labels to move files that do **not** contain both:

- a `girl`/`woman` class
- a `human face` class

By default, unmatched files are moved to `noface` inside your source folder.

## Model Description

Faceless runs inference with `yolov8n-oiv7.pt` and reads YOLO label files from `<source>/labels`.

- If labels are missing, it generates them first.
- If labels already exist, it reuses them.
- If a file does not have both required classes, it is moved.

## Intended Uses & Limitations

### Intended Uses

- Sorting large image folders by a simple class rule.
- Quickly separating images that do not match your target content.

### Limitations

- File moves are destructive to the original folder layout.
- The tool checks class IDs from labels; detection quality depends on the model output.
- The source folder may contain non-image files; items without matching labels can still be moved.

## How to Use

### 1) Download the release

Download the latest Windows binary from GitHub Releases:

- <https://github.com/mr-szgz/faceless/releases>

Download `faceless.exe`.

No Python setup is required.

### 2) Run in PowerShell

```bash
.\faceless.exe "<path-to-images>"
```

## Common Commands

Use PowerShell-style flags or long flags.

```bash
# positional path
.\faceless.exe "S:\Images\input"

# explicit path flag
.\faceless.exe "S:\Images\input"

# force label regeneration
.\faceless.exe "S:\Images\input" -Force

# change confidence threshold (default: 0.2)
.\faceless.exe "S:\Images\input" -Conf 0.2

# change destination folder name (default: noface)
.\faceless.exe "S:\Images\input" -Directory "faceless"

# split non-matching files into per-label folders
.\faceless.exe "S:\Images\input" -Auto

# split non-matching files into grouped folders from faceless/labels/*.yaml
.\faceless.exe "S:\Images\input" -Group
```

## Output

- Labels are stored in `<source>/labels`.
- Unmatched files are moved to `<source>/<directory>` (default: `<source>/noface`).
- With `-Auto` or `-Group`, unmatched files are moved into subfolders under the destination directory.
