from __future__ import annotations

import logging
from pathlib import Path

from .directories import get_directories

APP_NAME = "faceless"
MAIN_LOG_NAME = "main.log"
ERROR_LOG_NAME = "error.log"


class LazyFileHandler(logging.FileHandler):
    def _open(self):  # type: ignore[override]
        Path(self.baseFilename).parent.mkdir(parents=True, exist_ok=True)
        return super()._open()


def resolve_log_dir(app_name: str = APP_NAME) -> Path:
    local_root = Path(get_directories().get("Local", Path.home())).expanduser().resolve()
    return (local_root / app_name / "logs").expanduser().resolve()


def setup_logging(
    verbose: bool = False,
    app_name: str = APP_NAME,
    log_dir: Path | None = None,
) -> tuple[logging.Logger, Path, Path]:
    logger = logging.getLogger(app_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    active_log_dir = (log_dir or resolve_log_dir(app_name)).expanduser().resolve()
    main_log_path = active_log_dir / MAIN_LOG_NAME
    error_log_path = active_log_dir / ERROR_LOG_NAME

    detailed_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console_formatter = detailed_formatter if verbose else logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(console_formatter)

    main_handler = LazyFileHandler(main_log_path, encoding="utf-8", delay=True)
    main_handler.setLevel(logging.DEBUG)
    main_handler.setFormatter(detailed_formatter)

    error_handler = LazyFileHandler(error_log_path, encoding="utf-8", delay=True)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(main_handler)
    logger.addHandler(error_handler)
    return logger, main_log_path, error_log_path
