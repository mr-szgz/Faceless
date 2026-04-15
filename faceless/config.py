from __future__ import annotations
from pathlib import Path
from configparser import ConfigParser
from faceless import __version__
from typing import Any

import os

FACELESS_CONFIG_DIR = Path(os.environ.get("FACELESS_CONFIG_DIR", (Path.home() / ".faceless").expanduser().resolve()))
FACELESS_CONFIG_FILE = (FACELESS_CONFIG_DIR / "faceless.ini").resolve()
DEFAULT_MODEL_NAME = "yolov8n-oiv7.pt"
DEFAULT_YOLO_MATCH_CLASSES = "216,594"
DEFAULT_YOLO_FACE_CLASSES = "264"

DEPENDENCIES_DIR = FACELESS_CONFIG_DIR / "dependencies"
# print(f"[CONFIG] Dependencies dir {DEPENDENCIES_DIR}")

cfg = ConfigParser()

def set_config(section: str, option: str, value):
    cfg.read(str(FACELESS_CONFIG_FILE), encoding="utf-8")
    cfg.set(str(section).strip(), str(option).strip(), value)
    with FACELESS_CONFIG_FILE.open("w", encoding="utf-8") as handle:
        cfg.write(handle)

def get_config(section: str, option: str, fallback: Any | None = None):
    cfg.read_dict({
        "project": {
            "version": str(__version__),
            "output_folder": "faceless",
            "prefix": FACELESS_CONFIG_DIR,
            "dependencies": DEPENDENCIES_DIR
        },
        "detect": {
            "yolo_confidence": 0.2,
            "yolo_face_classes": DEFAULT_YOLO_FACE_CLASSES,
            "yolo_match_classes": DEFAULT_YOLO_MATCH_CLASSES
        }
    })
    cfg.read(str(FACELESS_CONFIG_FILE), encoding="utf-8")

    if not cfg.has_option(section, option) and option and fallback:
        cfg.set(section, str(option), fallback)
    if not cfg.has_section(section):
        cfg.add_section(section)
    
    FACELESS_CONFIG_FILE.parent.mkdir(exist_ok=True)
    FACELESS_CONFIG_FILE.touch(exist_ok=True)
    
    with FACELESS_CONFIG_FILE.open("w", encoding="utf-8") as handle:
        cfg.write(handle)
    cfg.read(str(FACELESS_CONFIG_FILE), encoding="utf-8")
    return cfg.get(section, option)
