# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for run-faceless — lightweight Windows launcher
# Build: pyinstaller run_faceless.spec
import sys
from pathlib import Path

def _resolve_project_root() -> Path:
    for arg in reversed(sys.argv):
        if arg.lower().endswith(".spec"):
            return Path(arg).expanduser().resolve().parent
    return Path.cwd().resolve()


PROJECT_ROOT = _resolve_project_root()
ICON_FILE = PROJECT_ROOT / "assets" / "logo.ico"
PNG_FILE = PROJECT_ROOT / "assets" / "logo.png"

a = Analysis(
    ["faceless/run_faceless.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[
        (str(ICON_FILE), "assets"),
        (str(PNG_FILE), "assets"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="run-faceless",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=True,
    onefile=True,
    icon=str(ICON_FILE),
)
