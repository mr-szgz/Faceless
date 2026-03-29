# Faceless

Faceless analyzes images and video using Ultralytics YOLO to segregate faceless media and/or group media into YOLO labelled folders.

## Install

Download Binary for your system or install .whl from latest Release

- <https://github.com/mr-szgz/faceless/releases>

## Usage

**Quickstart with..**

```bash
.\faceless "<path-to-images>" -Auto
```

```bash
$ faceless --help                                                                                                                                                                                                         
usage: faceless [-h] [-Path PATH_OPTION] [-Label] [-Conf CONF] [-Directory DIRECTORY] [-Auto | -Group] [path]

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
  -Directory DIRECTORY, --directory DIRECTORY
                        Output directory name for moved files (default: noface)
  -Auto, --auto, -a     Move non-matching files into per-label folders under the output directory
  -Group, --group, -g   Move non-matching files into grouped folders under the output directory based on faceless/labels/*.yaml priority
```

## Changes

See [CHANGELOG.md](./CHANGELOG.md) for release notes and version history.
