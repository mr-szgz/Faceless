# Faceless

Faceless analyzes images and video with Ultralytics YOLO and automatically labels and organizes media by labels

**Features**

- YOLO with `yolov8n-oiv7` (pretrained OpenImagesV7 YOLO). See [OpenImagesV7.yaml](.\faceless\datasets\OpenImagesV7.yaml).
- Low Quality images are moved into `./Top Label Name/.` where Top Label is one of [OpenImagesV7.yaml](.\faceless\datasets\OpenImagesV7.yaml). <br />
_this is currently images without a visible face (class id `264`) and not at least one of the `-RequireIds/--require-ids` (default classes `216,594`)_
- Automatically saves crops to `./Faces`
- Insightface to group faces into `./Faces/person_$hash/.` where $hash is calculated md5 of the embeddings grouped.

## Install

Download the Windows exe or wheel from the GitHub releases page: [https://github.com/mr-szgz/faceless/releases](https://github.com/mr-szgz/faceless/releases).

Install a wheel directly:

```sh
pip install ./faceless-0.9.0-py3-none-any.whl --force
```

Or clone the repo and install in editable mode:

```sh
pip install -e .
```

Or uv

```sh

$ uv sync
$ uv run python -m faceless --help
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
