# Faceless

Faceless analyzes images and video with Ultralytics YOLO, then moves media that does not contain both a face (class `264`) and at least one required match class (default: `216,594`). Non-matching files land in the output directory (default: `faceless`) and are grouped into label-named subfolders when labels are available.

## Install

Download the Windows exe or wheel from the GitHub releases page: [https://github.com/mr-szgz/faceless/releases](https://github.com/mr-szgz/faceless/releases).

Install a wheel directly:

```sh
pip install ./faceless-0.7.1-py3-none-any.whl --force
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
$ faceless --help                                                                                                            
usage: faceless [-h] [-Force] [-Confidence CONF_FLOAT] [-RequireIds YOLO_CLASS_INTS] [-Directory OUTPUT_DIR] [path]                                              

positional arguments:
  path                  Source directory containing videos/images

options:
  -h, --help            show this help message and exit
  -Force, --force       Force regeneration of labels
  -Confidence CONF_FLOAT, --confidence CONF_FLOAT
                        Model confidence threshold
  -RequireIds YOLO_CLASS_INTS, --require-ids YOLO_CLASS_INTS
                        YOLO class IDs to keep comma-separated. All classes available in faceless/datasets/OpenImagesV7.yaml. Default: "216,594"
  -Directory OUTPUT_DIR, --directory OUTPUT_DIR
                        Override output directory. Default: ./faceless
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
