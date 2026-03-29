from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .logging import APP_NAME, resolve_log_dir


def add_userdata_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    userdata_parser = subparsers.add_parser(
        "userdata",
        help="Userdata utilities (info/logs/tail)",
    )
    userdata_parser.add_argument(
        "-Info",
        "--info",
        action="store_true",
        dest="info",
        help="Show faceless install/runtime paths as JSON",
    )
    userdata_parser.add_argument(
        "-Logs",
        "--logs",
        action="store_true",
        dest="logs",
        help="Show logs path and optionally open it",
    )
    userdata_parser.add_argument(
        "-Tail",
        "--tail",
        action="store_true",
        dest="tail",
        help="Tail faceless main log output in console",
    )


def handle_userdata_subparser(args: argparse.Namespace) -> bool:
    if args.command != "userdata":
        return False

    if not args.info and not args.logs and not args.tail:
        args.info = True

    if args.info:
        _show_info()

    if args.logs or args.tail:
        logs_path = resolve_log_dir()
        print(logs_path)

        if args.logs and os.isatty(0):
            try:
                if _prompt_yes_no("Open logs directory in File Explorer?", default_yes=False):
                    _open_directory_in_explorer(logs_path)
            except (EOFError, KeyboardInterrupt):
                print()

        if args.tail:
            try:
                _tail_main_log(logs_path / "main.log")
            except KeyboardInterrupt:
                print()
    return True


def _resolve_faceless_version() -> str:
    try:
        return version(APP_NAME)
    except PackageNotFoundError:
        return "unknown"


def _show_info() -> None:
    faceless_path = Path(__file__).resolve().parent.parent / "__init__.py"
    faceless_command = shutil.which("faceless")
    if faceless_command:
        faceless_path = Path(faceless_command).expanduser().resolve()

    logs_path = resolve_log_dir()
    data_path = logs_path.parent
    print(
        json.dumps(
            {
                "FACELESS PATH": str(faceless_path),
                "DATA PATH": str(data_path),
                "LOGS PATH": str(logs_path),
                "FACELESS VERSION": _resolve_faceless_version(),
            }
        )
    )


def _tail_main_log(log_path: Path, lines: int = 40) -> None:
    if not log_path.is_file():
        print(f"Log file not found: {log_path}")
        return

    existing_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in existing_lines[-lines:]:
        print(line)

    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        log_file.seek(0, os.SEEK_END)
        while True:
            line = log_file.readline()
            if line:
                print(line, end="")
            else:
                time.sleep(0.5)


def _prompt_yes_no(prompt: str, default_yes: bool) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        response = input(f"{prompt} {suffix} ").strip().lower()
        if not response:
            return default_yes
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please answer with y or n.")


def _open_directory_in_explorer(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    if os.name == "posix":
        opener = "open" if "darwin" in os.sys.platform else "xdg-open"
        subprocess.run([opener, str(path)], check=False)
