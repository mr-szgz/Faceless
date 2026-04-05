# Faceless

Faceless analyzes images and video using Ultralytics YOLO to segregate faceless media and/or group media into YOLO labelled folders.

## Install

Download Binary for your system or install .whl from latest Release
Visit release and find latest <https://github.com/mr-szgz/faceless/releases>

Install the whl directly like 

```sh
pip install ./faceless-0.4.2-py3-none-any.whl --force
```

or clone the repo and you can `pip install -e .` (`mise run dev` reccomended)


## Usage

**Automatically Move Faceless Media and Group by Label**

```sh
$ faceless "p:/path/to/media"
```

```sh
$ uv run python -m faceless --help                                                                                                                                
usage: faceless [-h] [-Path PATH_OPTION] [-Label] [-Conf CONF] [-Match MATCH] [-Directory DIRECTORY] [--install] [--uninstall] [--install-info] [path]            

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
  --install             Install the File Explorer context-menu entry (Windows only).
  --uninstall           Remove the File Explorer context-menu entry (Windows only).
  --install-info        Show resolved paths and registry command.
```

## Changes

See [CHANGELOG.md](./CHANGELOG.md) for release notes and version history.
