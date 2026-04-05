# Faceless

Faceless analyzes images and video with Ultralytics YOLO, then moves media that does not contain both a face (class `264`) and at least one required match class (default: `216,594`). Non-matching files land in the output directory (default: `faceless`) and are grouped into label-named subfolders when labels are available.

## Install

Download the Windows exe or wheel from the GitHub releases page: [https://github.com/mr-szgz/faceless/releases](https://github.com/mr-szgz/faceless/releases).

Install a wheel directly:

```sh
pip install ./faceless-0.7.0-py3-none-any.whl --force
```

Or clone the repo and install in editable mode:

```sh
pip install -e .
```

## Usage

Move non-matching media (grouped by label when available):

```sh
faceless "p:/path/to/media"
```

Show help:

```sh
python -m faceless --help
usage: faceless [-h] [-Path PATH_OPTION] [-Label] [-Conf CONF] [-Match MATCH] [-Directory DIRECTORY] [path]

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
                        Required match class IDs, comma-separated (example: "1,43,51"). Default: "216,594".
  -Directory DIRECTORY, --directory DIRECTORY
                        Output directory name for moved files (default: faceless)
```

## Windows Context Menu

If you downloaded `run-faceless.exe` from a release, you can install or remove the Explorer context menu entry:

```sh
run-faceless --install
run-faceless --uninstall
run-faceless --info
```

## Changes

See [CHANGELOG.md](./CHANGELOG.md) for release notes and version history.
