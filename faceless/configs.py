import json
from pathlib import Path
from urllib.request import urlretrieve

USER_HOME = Path("~").expanduser().resolve()
USER_DATA = USER_HOME / ".faceless"
FACELESS_CONFIG = USER_DATA / "faceless.json"


def ensure_user_data_layout() -> dict[str, Path]:
    models_dir = USER_DATA / "models"
    configs_dir = USER_DATA / "configs"

    USER_DATA.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "home": str(USER_HOME),
        "user_data_dir": str(USER_DATA),
        "models_dir": str(models_dir),
        "configs_dir": str(configs_dir),
    }
    if not FACELESS_CONFIG.is_file():
        FACELESS_CONFIG.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    return {
        "user_data": USER_DATA,
        "models": models_dir,
        "configs": configs_dir,
    }


def download_configs():
    paths = ensure_user_data_layout()
    destination = paths["configs"] / "OpenImagesV7.yaml"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    urlretrieve("https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/OpenImagesV7.yaml", destination)
    return destination
