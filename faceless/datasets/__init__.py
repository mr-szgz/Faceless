from pathlib import Path

from faceless.config import FACELESS_CONFIG_DIR

_OPENIMAGES_CONFIG = Path(FACELESS_CONFIG_DIR) / "configs" / "OpenImagesV7.yaml"

def _class_id_to_class_name(class_id: int, class_names: dict[int, str]):
    class_name = class_names.get(class_id)
    if class_name:
        return class_name

def _parse_openimages_class_names(config_path: Path) -> dict[int, str]:
    if not config_path.is_file():
        return {}

    names: dict[int, str] = {}
    in_names = False
    with config_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not in_names:
                if stripped == "names:":
                    in_names = True
                continue
            if line.lstrip() == line:
                break
            key, sep, value = stripped.partition(":")
            if not sep:
                continue
            key = key.strip()
            if not key:
                continue
            try:
                class_id = int(key)
            except ValueError:
                continue
            class_name = value.strip()
            if class_name and class_name[0] in "\"'" and class_name[-1:] == class_name[0]:
                class_name = class_name[1:-1]
            names[class_id] = class_name
    return names


OPENIMAGES_CLASS_NAMES = _parse_openimages_class_names(_OPENIMAGES_CONFIG)
from urllib.request import urlretrieve
def download_configs():
    if _OPENIMAGES_CONFIG.is_file() and _OPENIMAGES_CONFIG.stat().st_size > 0:
        return _OPENIMAGES_CONFIG
    urlretrieve("https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/OpenImagesV7.yaml", _OPENIMAGES_CONFIG)
    return _OPENIMAGES_CONFIG
