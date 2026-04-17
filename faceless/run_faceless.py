from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from faceless.config import DEPENDENCIES_DIR, set_config, get_config
import os
env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-faceless")
    parser.add_argument(
        "--install",
        "-Install",
        action="store_true",
        dest="install_dependencies",
        help="Download and Install Faceless",
    )
    parser.add_argument(
        "--install-menu",
        action="store_true",
        dest="install_menu",
        help="Install the File Explorer context-menu entry.",
    )
    parser.add_argument(
        "--uninstall-menu",
        action="store_true",
        dest="uninstall_menu",
        help="Remove the File Explorer context-menu entry.",
    )
    parser.add_argument(
        "--set-config-option",
        "-SetConfigOption",
        dest="set_config_value",
        metavar="<section>.<option>=<value>",
        help="Update a config option value in ~/.faceless/faceless.ini",
    )
    parser.add_argument(
        "-Force",
        "--force",
        dest="force_reinstall",
        help="adds --force-reinstall"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, remainder = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    
    dependencies = Path(str(get_config("project", "dependencies", DEPENDENCIES_DIR)))
    
    python_embed_dir = dependencies / "python_embed"
    pyexe = str(python_embed_dir / "python.exe")
    print(f"[CHECK] Python {pyexe}")
    
    if args.install_dependencies:
        git_exe = dependencies / "git" / "cmd" / "git.exe"
        (dependencies / "git").mkdir(exist_ok=True)
        (dependencies / "git").mkdir(parents=True, exist_ok=True)
        # install git
        if not git_exe.exists():
            import urllib.request
            (dependencies / "git").mkdir(parents=True, exist_ok=True)
            git_download_url = "https://github.com/git-for-windows/git/releases/download/v2.53.0.windows.2/PortableGit-2.53.0.2-64-bit.7z.exe"
            git_installer = (dependencies / "git/PortableGit-2.53.0.2-64-bit.7z.exe")
            with urllib.request.urlopen(git_download_url) as response, git_installer.open("wb") as handle:
                handle.write(response.read())
            subprocess.check_call(
                [str(git_installer), f"-o{dependencies / 'git'}", "-y"]
            )
            
        if not (dependencies / "ffprobe.exe").exists():
            import urllib.request
            import zipfile

            ffprobe_url = "https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v6.1/ffprobe-6.1-win-64.zip"
            ffprobe_zip = dependencies / "ffprobe-6.1-win-64.zip"
            if not ffprobe_zip.exists():
                with urllib.request.urlopen(ffprobe_url) as response, ffprobe_zip.open("wb") as handle:
                    handle.write(response.read())
            with zipfile.ZipFile(ffprobe_zip) as archive:
                archive.extractall(dependencies)
        
        if not (dependencies / "ffmpeg.exe").exists():
            import urllib.request
            import zipfile

            ffmpeg_url = "https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v6.1/ffmpeg-6.1-win-64.zip"
            ffmpeg_zip = dependencies / "ffmpeg-6.1-win-64.zip"
            if not ffmpeg_zip.exists():
                with urllib.request.urlopen(ffmpeg_url) as response, ffmpeg_zip.open("wb") as handle:
                    handle.write(response.read())
            with zipfile.ZipFile(ffmpeg_zip) as archive:
                archive.extractall(dependencies)
        
        if not (dependencies / "Faceless").exists():
            # clone project repo
            subprocess.check_call(
                [str(git_exe), "clone", "https://github.com/mr-szgz/Faceless.git", str(dependencies / "Faceless")]
            )
    
        if not python_embed_dir.exists():
            import urllib.request
            import zipfile

            python_embed_dir.mkdir(parents=True, exist_ok=True)
            embed_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
            embed_zip = dependencies / "python-3.10.11-embed-amd64.zip"
            if not embed_zip.exists():
                with urllib.request.urlopen(embed_url) as response, embed_zip.open("wb") as handle:
                    handle.write(response.read())
            with zipfile.ZipFile(embed_zip) as archive:
                archive.extractall(python_embed_dir)
        embed_pth = python_embed_dir / "python310._pth"
        if embed_pth.exists():
            lines = embed_pth.read_text(encoding="utf-8").splitlines()
            if "Lib\\site-packages" not in lines:
                lines.append("Lib\\site-packages")
            if "import site" not in lines:
                lines.append("import site")
            embed_pth.write_text("\n".join(lines) + "\n", encoding="utf-8")

        import urllib.request
        get_pip_path = dependencies / "get-pip.py"
        if not get_pip_path.exists():
            with urllib.request.urlopen("https://bootstrap.pypa.io/get-pip.py") as response, get_pip_path.open("wb") as handle:
                handle.write(response.read())
        
        subprocess.check_call(
            [str(pyexe), str(get_pip_path)],
            env=env,
        )
        subprocess.check_call(
            [str(pyexe), "-m", "pip", "install", "--upgrade", "pip", "cython"],
            env=env,
        )
        # check if this machine can support Cuda12 and if torch cpu is installed then uninstall it and allow below to isntall cua enabled torch properly
        subprocess.check_call(
            [str(pyexe), "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cu128", "torch", "torchvision", "torchaudio", "--force-reinstall" if args.force_reinstall else ""],
            env=env,
        )
        
        insightface_whl = str(sorted(Path("dist/whl").glob("insightface-*.whl"))[0])
        subprocess.check_call(
            [str(pyexe), "-m", "pip", "install", "numpy", "onnxruntime-gpu", "triton-windows>=3.5.0.post21",
                                                "ultralytics", "tqdm", "argparse", "rich", "opencv-python", "scikit-image", insightface_whl, "huggingface_hub" ], env=env)
        subprocess.check_call(
            [str(pyexe), "-m", "pip", "uninstall", "-y", "faceless"], env=env,
        )
        subprocess.check_call([str(pyexe), "-m", "pip", "install",
                               r"S:\Spaces\Image-Classification\Faceless\dist\faceless-0.8.0-py3-none-any.whl",
                               "--no-warn-script-location"
                            ], env=env)
        # subprocess.check_call([str(pyexe), "-m", "faceless", "--help"])
        args.install_menu = True

    if args.set_config_value:
        key, value = args.set_config_value.split("=", 1)
        section, option = key.split(".", 1)
        set_config(section.strip(), option, value)
        return 0

    if args.install_menu:
        import winreg
        launcher_exe = Path(sys.executable).resolve()
        launcher_icon = Path(__file__).resolve().parent.parent / "assets" / "logo.ico"
        if getattr(sys, "frozen", False):
            launcher_icon = launcher_exe
        command_value = f"\"{launcher_exe}\" \"%1\""
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, r"Software\Classes\Directory\shell\Faceless", 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "Run Faceless")
            winreg.SetValueEx(key, "Icon", 0, winreg.REG_SZ, str(launcher_icon))
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            r"Software\Classes\Directory\shell\Faceless\command",
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, command_value)
        print(f"[OK] Installed context menu: {launcher_exe}")
        return 0

    if args.uninstall_menu:
        import winreg

        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\Directory\shell\Faceless\command")
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\Directory\shell\Faceless")
        print("[OK] Removed Faceless context menu entry.")
        return 0

    if remainder:
        subprocess.check_call(
            [pyexe, "-m", "faceless", *remainder],
            env=env,
        )
    return 0



import sys
import traceback

def _wait_for_keypress() -> None:
    try:
        import msvcrt

        print("Press any key to close...", flush=True)
        msvcrt.getch()
    except Exception:
        try:
            input("Press Enter to close...")
        except Exception:
            pass

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        _wait_for_keypress()
        sys.exit(1)
