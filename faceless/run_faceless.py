from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-faceless")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the File Explorer context-menu entry (Windows only).",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove the File Explorer context-menu entry (Windows only).",
    )
    parser.add_argument(
        "--info",
        "-Info",
        "-i",
        action="store_true",
        help="Show resolved paths and registry command.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, remainder = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    project_root = Path(sys.executable).resolve().parent.parent
    launcher_exe = Path(sys.executable).resolve()
    python_exe = project_root / ".venv" / "Scripts" / "python.exe"
    command_value = f"\"{launcher_exe}\" \"%1\""

    if args.info:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Classes\Directory\shell\Faceless\command", 0, winreg.KEY_READ
        ) as key:
            command_value, _ = winreg.QueryValueEx(key, "")
        print(f"[winreg] project_root: {project_root}")
        print(f"[winreg] python_exe: {python_exe}")
        print(f"[winreg] command_value: {command_value}")
        return 0

    if args.install:
        import winreg

        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, r"Software\Classes\Directory\shell\Faceless", 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "Run Faceless")
            winreg.SetValueEx(key, "Icon", 0, winreg.REG_SZ, str(launcher_exe))
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            r"Software\Classes\Directory\shell\Faceless\command",
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, command_value)
        print(f"Installed context menu: {launcher_exe}")
        return 0

    if args.uninstall:
        import winreg

        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\Directory\shell\Faceless\command")
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\Directory\shell\Faceless")
        print("Removed Faceless context menu entry.")
        return 0

    cmd = [str(python_exe), "-m", "faceless", *remainder]
    result = subprocess.run(cmd, cwd=str(project_root))
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
