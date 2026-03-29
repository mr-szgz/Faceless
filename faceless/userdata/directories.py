from pathlib import Path
import os
import json
import platform
import argparse

def get_xdg(extra=False):
    """Get XDG directories."""
    home = Path.home()
    dirs = {
        "Profile": str(home.expanduser().resolve()),
        "Config": str(
            Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
            .expanduser()
            .resolve()
        ),
        "Local": str(
            Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
            .expanduser()
            .resolve()
        ),
        "Cache": str(
            Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
            .expanduser()
            .resolve()
        ),
    }
    if extra:
        dirs["Desktop"] = str((home / "Desktop").expanduser().resolve())
        dirs["Documents"] = str((home / "Documents").expanduser().resolve())
        dirs["Downloads"] = str((home / "Downloads").expanduser().resolve())
        dirs["Pictures"] = str((home / "Pictures").expanduser().resolve())
        dirs["Videos"] = str((home / "Videos").expanduser().resolve())
        dirs["Music"] = str((home / "Music").expanduser().resolve())
    return dirs


def get_linux(extra=False):
    """Get Linux default directories without XDG env overrides."""
    home = Path.home()
    dirs = {
        "Profile": str(home.expanduser().resolve()),
        "Config": str((home / ".config").expanduser().resolve()),
        "Local": str((home / ".local" / "share").expanduser().resolve()),
        "Cache": str((home / ".cache").expanduser().resolve()),
    }
    if extra:
        dirs["Desktop"] = str((home / "Desktop").expanduser().resolve())
        dirs["Documents"] = str((home / "Documents").expanduser().resolve())
        dirs["Downloads"] = str((home / "Downloads").expanduser().resolve())
        dirs["Pictures"] = str((home / "Pictures").expanduser().resolve())
        dirs["Videos"] = str((home / "Videos").expanduser().resolve())
        dirs["Music"] = str((home / "Music").expanduser().resolve())
    return dirs


def get_macos(extra=False):
    """Get macOS default directories."""
    home = Path.home()
    dirs = {
        "Profile": str(home.expanduser().resolve()),
        "Config": str((home / "Library" / "Application Support").expanduser().resolve()),
        "Local": str((home / "Library" / "Application Support").expanduser().resolve()),
        "Cache": str((home / "Library" / "Caches").expanduser().resolve()),
    }
    if extra:
        dirs["Desktop"] = str((home / "Desktop").expanduser().resolve())
        dirs["Documents"] = str((home / "Documents").expanduser().resolve())
        dirs["Downloads"] = str((home / "Downloads").expanduser().resolve())
        dirs["Pictures"] = str((home / "Pictures").expanduser().resolve())
        dirs["Videos"] = str((home / "Movies").expanduser().resolve())
        dirs["Music"] = str((home / "Music").expanduser().resolve())
    return dirs


def get_windows(extra=False):
    """
    Get specific Windows user directories.

    Returns:
            dict: A dictionary mapping directory names to their full resolved paths as strings.
    """
    """Get specific directories."""
    UserHome = Path(os.environ.get("USERPROFILE", Path.home()))
    RoamingDirectory = str(
        Path(os.environ.get("APPDATA", UserHome / "AppData" / "Roaming"))
        .expanduser()
        .resolve()
    )
    LocalDirectory = str(
            Path(os.environ.get("LOCALAPPDATA", UserHome / "AppData" / "Local"))
            .expanduser()
            .resolve()
        )
    dirs = {
        "Profile": str(UserHome.expanduser().resolve()),
        "Local": LocalDirectory,
        "Config": RoamingDirectory,
        "Cache": str(
            (Path(LocalDirectory) / "Cache")
            .expanduser()
            .resolve()
        ),
    }
    if extra:
        dirs["Roaming"] = RoamingDirectory
        dirs["Desktop"] = str((UserHome / "Desktop").expanduser().resolve())
        dirs["Documents"] = str((UserHome / "Documents").expanduser().resolve())
        dirs["Downloads"] = str((UserHome / "Downloads").expanduser().resolve())
        dirs["Pictures"] = str((UserHome / "Pictures").expanduser().resolve())
        dirs["Videos"] = str((UserHome / "Videos").expanduser().resolve())
        dirs["Music"] = str((UserHome / "Music").expanduser().resolve())
    return dirs


def get_directories(platform_name=None, extra=False):
    if platform_name is None:
        platform_name = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}[platform.system()]

    if platform_name == "windows":
        dirs = get_windows(extra)
        dirs["Temp"] = str((Path(dirs["Local"]) / "Temp").expanduser().resolve())
        return dirs

    if platform_name == "xdg":
        dirs = get_xdg(extra)
        dirs["Temp"] = str((Path(dirs["Local"]) / ".temp").expanduser().resolve())
        return dirs

    if platform_name == "linux":
        dirs = get_linux(extra)
        dirs["Temp"] = str((Path(dirs["Local"]) / ".temp").expanduser().resolve())
        return dirs

    if platform_name == "macos":
        dirs = get_macos(extra)
        dirs["Temp"] = str((Path(dirs["Local"]) / ".temp").expanduser().resolve())
        return dirs


def main():
    parser = argparse.ArgumentParser(description="Print user directories as JSON")
    platform_group = parser.add_mutually_exclusive_group()
    platform_group.add_argument("-Linux", "--linux", action="store_true", help="Use Linux directories")
    platform_group.add_argument("-Windows", "--windows", action="store_true", help="Use Windows directories")
    platform_group.add_argument("-Xdg", "--xdg", action="store_true", help="Use XDG directories")
    platform_group.add_argument("-Macos", "--macos", action="store_true", help="Use macOS directories")
    parser.add_argument(
        "-Extra",
        "--extra",
        action="store_true",
        help="Include non-core user directories",
    )
    args = parser.parse_args()

    selected_platform = (
        "linux"
        if args.linux
        else "windows"
        if args.windows
        else "xdg"
        if args.xdg
        else "macos"
        if args.macos
        else None
    )

    print(json.dumps(get_directories(selected_platform, args.extra), indent=2))


if __name__ == "__main__":
    main()
