from pathlib import Path

from ultralytics.utils.downloads import attempt_download_asset

from .configs import ensure_user_data_layout


def download_models(model_name: str = "yolov8n-oiv7.pt"):
    models_dir = ensure_user_data_layout()["models"]
    destination = models_dir / model_name

    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    downloaded_path = Path(attempt_download_asset(str(destination))).resolve()
    if downloaded_path != destination.resolve():
        destination.write_bytes(downloaded_path.read_bytes())

    return destination
