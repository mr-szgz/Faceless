#!/usr/bin/env python3
import argparse
import glob
import logging
import os
from pathlib import Path
import re
import subprocess
import shutil
import sys
from typing import Literal
from ultralytics import YOLO
from .userdata import add_userdata_subparser, get_directories, handle_userdata_subparser, setup_logging


logger = logging.getLogger("faceless")
DEFAULT_MODEL_NAME = "yolov8n-oiv7.pt"


def parse_yaml_id_name_map(labels_file: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    for line in labels_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key.isdigit():
            names[int(key)] = value
    return names


def load_label_names() -> dict[int, str]:
    """Load id-to-name mapping from known YAML locations without extra deps."""
    repo_root = Path(__file__).resolve().parent.parent
    candidate_files = (
        repo_root / "labels" / "Labels.yaml",
        Path(__file__).resolve().parent / "labels" / "Labels.yaml",
    )
    for labels_file in candidate_files:
        if labels_file.is_file():
            names = parse_yaml_id_name_map(labels_file)
            if names:
                return names
    return {}


def normalize_label_name(name: str) -> str:
    return "".join(char for char in name.lower() if char.isalnum())


def resolve_user_data_dir(app_name: str = "faceless") -> Path:
    try:
        local_root = get_directories().get("Local")
        if local_root:
            return (Path(local_root) / app_name).expanduser().resolve()
    except Exception:
        pass

    home = Path.home()
    if os.name == "nt":
        return (home / "AppData" / "Local" / app_name).expanduser().resolve()
    if os.sys.platform == "darwin":
        return (home / "Library" / "Application Support" / app_name).expanduser().resolve()
    return (home / ".local" / "share" / app_name).expanduser().resolve()


def resolve_model_path(model_name: str) -> Path:
    model_dir = resolve_user_data_dir() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_name

    if model_path.is_file():
        return model_path

    # Reuse an existing local model file before falling back to a fresh download.
    candidate_paths = (
        Path.cwd() / model_name,
        Path(__file__).resolve().parent.parent / model_name,
    )
    for candidate in candidate_paths:
        if candidate.is_file():
            shutil.copy2(candidate, model_path)
            return model_path

    return model_path


def resolve_yolo_source_pattern(source: Path) -> str:
    # Use an escaped glob pattern so literal [] in folder names don't break file discovery.
    escaped_source = glob.escape(str(source))
    if escaped_source.endswith(("\\", "/")):
        return f"{escaped_source}*.*"
    return f"{escaped_source}{os.sep}*.*"


def build_legacy_index_label_map(labels: Path, source_files: list[Path]) -> dict[str, Path]:
    """
    Backward compatibility for runs that wrote labels as image{index}.txt.
    Map those labels back to source file stems using sorted source-file order.
    """
    index_pattern = re.compile(r"^image(\d+)\.txt$", re.IGNORECASE)
    label_map: dict[str, Path] = {}

    if not labels.is_dir() or not source_files:
        return label_map

    for label_file in labels.glob("*.txt"):
        match = index_pattern.match(label_file.name)
        if not match:
            continue
        index = int(match.group(1))
        if 0 <= index < len(source_files):
            source_stem = source_files[index].stem
            # Keep direct <stem>.txt precedence by only filling missing fallback entries.
            label_map.setdefault(source_stem, label_file)
    return label_map


def update_label_names_from_model(model: YOLO, label_names: dict[int, str]) -> None:
    model_names = model.names
    if isinstance(model_names, dict):
        label_names.update({int(key): str(value) for key, value in model_names.items()})
        return
    label_names.update({index: str(value) for index, value in enumerate(model_names)})


def resolve_required_class_ids(
    model_name: str,
    label_names: dict[int, str],
    fallback_girl_or_woman_classes: set[int],
    fallback_human_face_class: int,
) -> tuple[set[int], set[int], YOLO | None]:
    girl_or_woman_labels = {"girl", "woman", "women"}
    human_face_labels = {"human face", "face"}
    normalized_girl_or_woman_labels = {normalize_label_name(name) for name in girl_or_woman_labels}
    normalized_human_face_labels = {normalize_label_name(name) for name in human_face_labels}
    model: YOLO | None = None

    girl_or_woman_classes = {
        class_id
        for class_id, class_name in label_names.items()
        if normalize_label_name(class_name) in normalized_girl_or_woman_labels
    }
    human_face_classes = {
        class_id
        for class_id, class_name in label_names.items()
        if normalize_label_name(class_name) in normalized_human_face_labels
    }

    if not girl_or_woman_classes or not human_face_classes:
        try:
            model = YOLO(model_name)
            update_label_names_from_model(model, label_names)
        except Exception:
            model = None

        girl_or_woman_classes = {
            class_id
            for class_id, class_name in label_names.items()
            if normalize_label_name(class_name) in normalized_girl_or_woman_labels
        }
        human_face_classes = {
            class_id
            for class_id, class_name in label_names.items()
            if normalize_label_name(class_name) in normalized_human_face_labels
        }

    if not girl_or_woman_classes:
        girl_or_woman_classes = set(fallback_girl_or_woman_classes)
    if not human_face_classes:
        human_face_classes = {fallback_human_face_class}

    return girl_or_woman_classes, human_face_classes, model


def parse_label_counts(label_path: Path) -> dict[int, int]:
    if not label_path.is_file():
        return {}

    counts: dict[int, int] = {}
    for line in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        parts: list[str] = line.split()
        if not parts:
            continue
        class_token = parts[0].lstrip("\ufeff")
        try:
            class_id = int(class_token)
        except ValueError:
            try:
                class_id_float = float(class_token)
            except ValueError:
                continue
            if not class_id_float.is_integer():
                continue
            class_id = int(class_id_float)
        counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def sanitize_folder_name(name: str) -> str:
    sanitized = "".join("_" if char in '<>:"/\\|?*' else char for char in name).strip(" .")
    return sanitized or "unlabeled"


def parse_group_class_ids(group_file: Path) -> set[int]:
    class_ids: set[int] = set()
    for line in group_file.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key = stripped.split(":", 1)[0].strip().strip("'\"")
        if key.isdigit():
            class_ids.add(int(key))
    return class_ids


def load_group_definitions() -> list[tuple[str, set[int]]]:
    groups_dir = Path(__file__).resolve().parent / "labels"
    definitions: list[tuple[int, str, set[int]]] = []

    for group_file in groups_dir.glob("*.yaml"):
        class_ids = parse_group_class_ids(group_file)
        if not class_ids:
            continue
        stem = group_file.stem
        prefix, _, _ = stem.partition("_")
        priority = int(prefix) if prefix.isdigit() else 10**9
        definitions.append((priority, stem, class_ids))

    definitions.sort(key=lambda item: (item[0], item[1].lower()))
    return [(group_name, class_ids) for _, group_name, class_ids in definitions]


def resolve_group_folder_name(label_counts: dict[int, int], group_definitions: list[tuple[str, set[int]]]) -> str | None:
    if not label_counts:
        return None

    class_ids = set(label_counts)
    for group_name, group_class_ids in group_definitions:
        if class_ids & group_class_ids:
            return group_name
    return None


def prompt_yes_no(prompt: str, default_yes: bool) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        response = input(f"{prompt} {suffix} ").strip().lower()
        if not response:
            return default_yes
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        logger.warning("Please answer with y or n.")


def open_directory_in_explorer(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
            return
        if os.name == "posix":
            opener = "open" if "darwin" in os.sys.platform else "xdg-open"
            subprocess.run([opener, str(path)], check=False)
            return
    except Exception as error:
        logger.error("Could not open %s: %s", path, error)


def prompt_open_folders(source: Path, labels: Path) -> None:
    if not os.isatty(0):
        return

    try:
        if prompt_yes_no("Open source input directory in File Explorer?", default_yes=True):
            open_directory_in_explorer(source)
            return
        if prompt_yes_no("Open labels folder?", default_yes=False):
            if labels.is_dir():
                open_directory_in_explorer(labels)
            else:
                logger.warning("Labels folder does not exist: %s", labels)
    except (EOFError, KeyboardInterrupt):
        logger.info("Folder prompt cancelled.")


def parse_arguments() -> tuple[
    Literal["yolov8n-oiv7.pt"],
    set[int],
    int,
    bool,
    float,
    Path | None,
    Path | None,
    Path | None,
    bool,
    bool,
    bool,
    argparse.Namespace,
]:
    MODEL_NAME = DEFAULT_MODEL_NAME
    GIRL_OR_WOMAN_CLASSES: set[int] = {216, 594}
    HUMAN_FACE_CLASS = 264

    ForceLabels = False
    Conf = 0.2
    Directory = "noface"

    parser = argparse.ArgumentParser(prog="faceless")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run image filtering")
    run_parser.add_argument("path", nargs="?", help="Source directory containing images")
    run_parser.add_argument("-Path", "--path", dest="path_option", help="Source directory containing images")
    run_parser.add_argument("-Label", "--label", "-Force", "--force", action="store_true", dest="force_labels", help="Force regeneration of labels")
    run_parser.add_argument("-Conf", "--conf", type=float, default=Conf, help="Model confidence threshold")
    run_parser.add_argument("-Directory", "--directory", help=f"Output directory name for moved files (default: {Directory})")
    run_parser.add_argument(
        "-Verbose",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Enable verbose logging in console and log files",
    )
    move_augment_group = run_parser.add_mutually_exclusive_group()
    move_augment_group.add_argument("-Auto", "--auto", "-a", action="store_true", dest="auto_directory", help="Move non-matching files into per-label folders under the output directory")
    move_augment_group.add_argument("-Group", "--group", "-g", action="store_true", dest="group_directory", help="Move non-matching files into grouped folders under the output directory based on faceless/labels/*.yaml priority")
    add_userdata_subparser(subparsers)

    argv = sys.argv[1:]
    userdata_flags = {"-Info", "--info", "-Logs", "--logs", "-Tail", "--tail"}
    if argv and argv[0] in userdata_flags:
        argv = ["userdata", *argv]
    elif not argv or argv[0] not in {"run", "userdata", "-h", "--help"}:
        argv = ["run", *argv]
    args = parser.parse_args(argv)

    if args.command == "userdata":
        return (
            MODEL_NAME,
            GIRL_OR_WOMAN_CLASSES,
            HUMAN_FACE_CLASS,
            ForceLabels,
            Conf,
            None,
            None,
            None,
            False,
            False,
            False,
            args,
        )

    path = args.path_option or args.path
    if path is None:
        run_parser.error("Source path is required. Use -Path/--path or pass it positionally.")
    ForceLabels = args.force_labels
    Conf = args.conf
    if args.directory is not None:
        Directory = args.directory
    AutoDirectory = args.auto_directory
    GroupDirectory = args.group_directory
    Verbose = args.verbose

    source: Path | None = None
    labels: Path | None = None
    Destination: Path | None = None
    if path is not None:
        source = Path(path).expanduser().resolve()
        labels = source / "labels"
        Destination = source / Directory

    return (
        MODEL_NAME,
        GIRL_OR_WOMAN_CLASSES,
        HUMAN_FACE_CLASS,
        ForceLabels,
        Conf,
        source,
        labels,
        Destination,
        AutoDirectory,
        GroupDirectory,
        Verbose,
        args,
    )

def main() -> None:
    (
        MODEL_NAME,
        GIRL_OR_WOMAN_CLASSES,
        HUMAN_FACE_CLASS,
        ForceLabels,
        conf,
        source,
        labels,
        Destination,
        AutoDirectory,
        GroupDirectory,
        Verbose,
        args,
    ) = parse_arguments()

    if handle_userdata_subparser(args):
        return

    if source is None or labels is None or Destination is None:
        raise RuntimeError("Source, labels, and destination paths must be set before processing.")

    _, main_log_path, error_log_path = setup_logging(verbose=Verbose)

    logger.debug("Main log file: %s", main_log_path)
    logger.debug("Error log file: %s", error_log_path)
    logger.info(
        "Starting run: source=%s destination=%s force_labels=%s verbose=%s",
        source,
        Destination,
        ForceLabels,
        Verbose,
    )

    try:
        model_path = resolve_model_path(MODEL_NAME)

        label_names = load_label_names()
        auto_directory = AutoDirectory
        group_directory = GroupDirectory
        group_definitions = load_group_definitions() if group_directory else []
        (
            GIRL_OR_WOMAN_CLASSES,
            HUMAN_FACE_CLASSES,
            model,
        ) = resolve_required_class_ids(
            model_name=str(model_path),
            label_names=label_names,
            fallback_girl_or_woman_classes=GIRL_OR_WOMAN_CLASSES,
            fallback_human_face_class=HUMAN_FACE_CLASS,
        )

        if ForceLabels or not labels.is_dir():
            logger.info("Generating labels in %s", labels)
            if model is None:
                model = YOLO(str(model_path))
            model.predict(
                source=resolve_yolo_source_pattern(source),
                conf=conf,
                project=str(source),
                name=".",
                vid_stride=50, # controls frame skip
                save=False,
                save_txt=True,
                save_conf=True, 
                stream=True,
                exist_ok=True,
                verbose=True, # ALWAYS True. Not releated to argparse.
            )

        source_files = sorted(file_path for file_path in source.iterdir() if file_path.is_file())
        logger.info("Discovered %d source file(s).", len(source_files))
        legacy_index_label_map = build_legacy_index_label_map(labels=labels, source_files=source_files)

        moved_count = 0
        for file_path in source_files:
            label_path: Path = labels / f"{file_path.stem}.txt"
            if not label_path.is_file():
                label_path = legacy_index_label_map.get(file_path.stem, label_path)

            label_counts = parse_label_counts(label_path)
            has_girl_or_woman = any(class_id in GIRL_OR_WOMAN_CLASSES for class_id in label_counts)
            has_human_face = any(class_id in HUMAN_FACE_CLASSES for class_id in label_counts)

            if has_girl_or_woman and has_human_face:
                continue

            if auto_directory:
                if label_counts:
                    primary_class_id = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
                    folder_name = label_names.get(primary_class_id)
                    if folder_name is None:
                        if model is None:
                            model = YOLO(str(model_path))
                        update_label_names_from_model(model, label_names)
                        folder_name = label_names.get(primary_class_id, f"class_{primary_class_id}")
                    destination_path = Destination / sanitize_folder_name(folder_name)
                else:
                    destination_path = Destination
            elif group_directory:
                group_folder_name = resolve_group_folder_name(label_counts, group_definitions)
                if group_folder_name is None:
                    destination_path = Destination
                else:
                    destination_path = Destination / sanitize_folder_name(group_folder_name)
            else:
                destination_path = Destination

            destination_path.mkdir(parents=True, exist_ok=True)
            destination_file = destination_path / file_path.name
            shutil.move(str(file_path), str(destination_file))
            moved_count += 1
            logger.debug("Moved %s -> %s", file_path, destination_file)

        logger.info("Moved %d non-matching file(s).", moved_count)
        prompt_open_folders(source=source, labels=labels)
    except Exception:
        logger.exception("Unhandled error while running faceless.")
        raise

if __name__ == "__main__":
    main()
