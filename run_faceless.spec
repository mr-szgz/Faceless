# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for run-faceless — lightweight Windows launcher
# Build: pyinstaller run_faceless.spec

a = Analysis(
    ["faceless/run_faceless.py"],
    pathex=[],
    binaries=[],
    datas=[],
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
)
