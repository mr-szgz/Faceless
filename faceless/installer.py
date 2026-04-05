from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

COMMAND_NAME = "faceless"
COMMAND_TITLE = COMMAND_NAME.capitalize()

def _add_installer_argparser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
        "--install-info",
        action="store_true",
        help="Show resolved paths and registry command.",
    )
    return parser


project_root = Path(sys.executable).resolve().parent.parent
launcher_exe = Path(sys.executable).resolve()
python_exe = project_root / ".venv" / "Scripts" / "python.exe"
command_value = f"\"{launcher_exe}\" \"%1\""

def installer_info():
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        fr"Software\Classes\Directory\shell\{COMMAND_TITLE}\command",
        0,
        winreg.KEY_READ,
    ) as key:
        command_value, _ = winreg.QueryValueEx(key, "")
    print(f"[winreg] project_root: {project_root}")
    print(f"[winreg] python_exe: {python_exe}")
    print(f"[winreg] command_value: {command_value}")

def installer_install():
    import winreg

    with winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER,
        fr"Software\Classes\Directory\shell\{COMMAND_TITLE}",
        0,
        winreg.KEY_SET_VALUE,
    ) as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, f"Run {COMMAND_TITLE}")
        winreg.SetValueEx(key, "Icon", 0, winreg.REG_SZ, str(launcher_exe))
    with winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER,
        fr"Software\Classes\Directory\shell\{COMMAND_TITLE}\command",
        0,
        winreg.KEY_SET_VALUE,
    ) as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, command_value)
    print(f"Installed context menu: {launcher_exe}")

def installer_uninstall():
    import winreg

    winreg.DeleteKey(
        winreg.HKEY_CURRENT_USER,
        fr"Software\Classes\Directory\shell\{COMMAND_TITLE}\command",
    )
    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, fr"Software\Classes\Directory\shell\{COMMAND_TITLE}")
    print(f"Removed {COMMAND_TITLE} context menu entry.")
