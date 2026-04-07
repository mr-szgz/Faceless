from __future__ import annotations

import argparse
from pathlib import Path
import sys


REGFILE_JINJA_TPL = """Windows Registry Editor Version 5.00

[HKEY_CURRENT_USER\\Software\\Classes\\Directory\\shell\\Faceless]
@="Run Faceless"
"Icon"="{{ faceless_exe_path }}"

[HKEY_CURRENT_USER\\Software\\Classes\\Directory\\shell\\Faceless\\command]
@="\\"{{ faceless_exe_path }}\\" \\"%1\\""
"""

def add_build_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    build_parser = subparsers.add_parser(
        "build",
        help="Build Faceless context-menu .reg file",
    )
    build_parser.add_argument(
        "-Exe",
        "--exe",
        required=True,
        dest="exe",
        help="Path to Faceless executable",
    )
    return build_parser

def parse_arguments() -> Path:
    """
    handles all argparser implementation, subparsers, and argument handling - DO NOT ADD HELPERS
    """
    
    parser = argparse.ArgumentParser(prog="New-ContextMenu")
    subparsers = parser.add_subparsers(dest="command")
    build_parser = add_build_subparser(subparsers)

    argv = sys.argv[1:]
    if not argv or argv[0] not in {"build", "-h", "--help"}:
        argv = ["build", *argv]
    args = parser.parse_args(argv)
    print(f"[New-ContextMenu] {args}")

    exe_path = Path(args.exe).expanduser().resolve()
    print(f"[New-ContextMenu] exe_path: {exe_path}")
    return exe_path


def main() -> None:
    exe_path = parse_arguments()

    # prepare 
    rendered = REGFILE_JINJA_TPL.replace("{{ faceless_exe_path }}", str(exe_path)).replace(
        "{{faceless_exe_path}}", str(exe_path)
    )

    output = Path.cwd() / "Install-FacelessContextMenu.reg"
    output.write_text(rendered, encoding="utf-8")
    print(f"[New-ContextMenu] wrote: {output}")


if __name__ == "__main__":
    main()
