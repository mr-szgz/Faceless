#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
from typing import Literal
import tqdm
from ultralytics import YOLO


def load_label_names() -> dict[int, str]:
    """Load id-to-name mapping from labels/Labels.yaml without extra deps."""
    labels_file = Path(__file__).resolve().parent.parent / "labels" / "Labels.yaml"
    names: dict[int, str] = {}
    try:
        for line in labels_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key.isdigit():
                names[int(key)] = value
    except FileNotFoundError:
        return {}
    return names


def parse_label_counts(label_path: Path) -> dict[int, int]:
    if not label_path.is_file():
        return {}

    counts: dict[int, int] = {}
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts: list[str] = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
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


def parse_arguments() -> tuple[Literal['yolov8n-oiv7.pt'], set[int], Literal[264], bool, float, Path, Path, Path, bool, bool]:
    MODEL_NAME = "yolov8n-oiv7.pt"
    GIRL_OR_WOMAN_CLASSES: set[int] = {216, 594}
    HUMAN_FACE_CLASS = 264

    ForceLabels = False
    Conf = 0.2
    Directory = "noface"

    parser = argparse.ArgumentParser(prog="faceless")
    parser.add_argument("path", nargs="?", help="Source directory containing images")
    parser.add_argument("-Path", "--path", dest="path_option", help="Source directory containing images")
    parser.add_argument("-Label", "--label", action="store_true", dest="force_labels", help="Force regeneration of labels")
    parser.add_argument("-Conf", "--conf", type=float, default=Conf, help="Model confidence threshold")
    parser.add_argument("-Directory", "--directory", help=f"Output directory name for moved files (default: {Directory})")
    move_augment_group = parser.add_mutually_exclusive_group()
    move_augment_group.add_argument("-Auto", "--auto", "-a", action="store_true", dest="auto_directory", help="Move non-matching files into per-label folders under the output directory")
    move_augment_group.add_argument("-Group", "--group", "-g", action="store_true", dest="group_directory", help="Move non-matching files into grouped folders under the output directory based on faceless/labels/*.yaml priority")

    args = parser.parse_args()

    path = args.path_option or args.path
    if path is None:
        parser.error("Source path is required. Use -Path/--path or pass it positionally.")

    ForceLabels = args.force_labels
    Conf = args.conf
    if args.directory is not None:
        Directory = args.directory
    AutoDirectory = args.auto_directory
    GroupDirectory = args.group_directory

    source: Path = Path(path).expanduser().resolve()
    labels: Path = source / "labels"
    Destination: Path = source / Directory

    return MODEL_NAME, GIRL_OR_WOMAN_CLASSES, HUMAN_FACE_CLASS, ForceLabels, Conf, source, labels, Destination, AutoDirectory, GroupDirectory

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
    ) = parse_arguments()

    label_names = load_label_names()
    auto_directory = AutoDirectory
    group_directory = GroupDirectory
    group_definitions = load_group_definitions() if group_directory else []
    if ForceLabels or not labels.is_dir():
        YOLO(MODEL_NAME).predict(
            source=str(source),
            conf=conf,
            save=False,
            save_txt=True,
            save_conf=True,
            project=str(source),
            name=".",
            exist_ok=True,
            verbose=True,
            vid_stride=50,
        )

    for file_path in source.iterdir():
        if not file_path.is_file():
            continue

        label_path: Path = labels / f"{file_path.stem}.txt"

        label_counts = parse_label_counts(label_path)
        has_girl_or_woman = any(class_id in GIRL_OR_WOMAN_CLASSES for class_id in label_counts)
        has_human_face = HUMAN_FACE_CLASS in label_counts

        if has_girl_or_woman and has_human_face:
            continue

        if auto_directory:
            if label_counts:
                primary_class_id = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
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
        shutil.move(str(file_path), str(destination_path / file_path.name))

if __name__ == "__main__":
    main()
