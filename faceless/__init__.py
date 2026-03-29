#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

DEFAULT_MODEL_NAME = "yolov8n-oiv7.pt"


def main() -> None:
    parser = argparse.ArgumentParser(prog="faceless")
    parser.add_argument("path", nargs="?", help="Source directory containing images")
    parser.add_argument("-Path", "--path", dest="path_option", help="Source directory containing images")
    parser.add_argument(
        "-Label",
        "--label",
        "-Force",
        "--force",
        action="store_true",
        dest="force_labels",
        help="Force regeneration of labels",
    )
    parser.add_argument("-Conf", "--conf", type=float, default=0.2, help="Model confidence threshold")
    parser.add_argument(
        "-Directory",
        "--directory",
        default="noface",
        help="Output directory name for moved files (default: noface)",
    )
    parser.add_argument(
        "-Auto",
        "--auto",
        "-a",
        action="store_true",
        dest="auto_directory",
        help="Move non-matching files into per-label folders under the output directory",
    )

    args = parser.parse_args(sys.argv[1:])
    source_text = args.path_option or args.path
    if source_text is None:
        parser.error("Source path is required. Use -Path/--path or pass it positionally.")

    source = Path(source_text).expanduser().resolve()
    if not source.exists():
        parser.error(f"Source path does not exist: {source}")
    if not source.is_dir():
        parser.error(f"Source path is not a directory: {source}")

    labels = source / "labels"
    destination_root = source / args.directory
    source_files = sorted(path for path in source.iterdir() if path.is_file())

    generate_labels = args.force_labels or not labels.is_dir() or any(
        not (labels / f"{path.stem}.txt").is_file() for path in source_files
    )

    model = None
    if generate_labels or args.auto_directory:
        model = YOLO(DEFAULT_MODEL_NAME)

    if generate_labels and source_files:
        labels.mkdir(parents=True, exist_ok=True)
        escaped_source = glob.escape(str(source))
        source_pattern = f"{escaped_source}{'' if escaped_source.endswith(('/', '\\')) else os.sep}*.*"
        print(f"Generating labels in {labels}")
        for _ in model.predict(
            source=source_pattern,
            conf=args.conf,
            project=str(source),
            name=".",
            save=False,
            save_txt=True,
            save_conf=True,
            stream=True,
            exist_ok=True,
            verbose=True,
        ):
            pass

    girl_or_woman_classes = {216, 594}
    human_face_classes = {264}
    label_names: dict[int, str] = {}

    if model is not None:
        if isinstance(model.names, dict):
            label_names = {int(key): str(value) for key, value in model.names.items()}
        else:
            label_names = {index: str(value) for index, value in enumerate(model.names)}

        normalized_girl_or_woman = {"girl", "woman", "women"}
        normalized_human_face = {"humanface", "face"}

        resolved_girl_or_woman = {
            class_id
            for class_id, class_name in label_names.items()
            if "".join(ch for ch in class_name.lower() if ch.isalnum()) in normalized_girl_or_woman
        }
        resolved_human_face = {
            class_id
            for class_id, class_name in label_names.items()
            if "".join(ch for ch in class_name.lower() if ch.isalnum()) in normalized_human_face
        }

        if resolved_girl_or_woman:
            girl_or_woman_classes = resolved_girl_or_woman
        if resolved_human_face:
            human_face_classes = resolved_human_face

    moved_count = 0
    for source_file in source_files:
        label_path = labels / f"{source_file.stem}.txt"
        label_counts: dict[int, int] = {}

        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                parts = line.split()
                if not parts:
                    continue
                token = parts[0].lstrip("\ufeff")
                try:
                    token_value = float(token)
                except ValueError:
                    continue
                if not token_value.is_integer():
                    continue
                class_id = int(token_value)
                label_counts[class_id] = label_counts.get(class_id, 0) + 1

        has_girl_or_woman = any(class_id in girl_or_woman_classes for class_id in label_counts)
        has_human_face = any(class_id in human_face_classes for class_id in label_counts)

        if has_girl_or_woman and has_human_face:
            continue

        destination_path = destination_root
        if args.auto_directory and label_counts:
            primary_class_id = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            folder_name = label_names.get(primary_class_id, f"class_{primary_class_id}")
            folder_name = "".join("_" if char in '<>:"/\\|?*' else char for char in folder_name).strip(" .")
            destination_path = destination_root / (folder_name or "unlabeled")

        destination_path.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_file), str(destination_path / source_file.name))
        moved_count += 1

    print(f"Moved {moved_count} non-matching file(s).")


if __name__ == "__main__":
    main()
