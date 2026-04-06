#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO
from faceless.models import download_models
DEFAULT_MODEL_NAME = "yolov8n-oiv7.pt"
DEFAULT_MATCH_SELECTORS = "216,594"
REQUIRED_FACE_SELECTORS = "264"

def main() -> None:
    parser = argparse.ArgumentParser(prog="faceless")
    parser.add_argument(
        "path", nargs="?", help="Source directory containing videos/images"
    )
    parser.add_argument(
        "-Force",
        "--force",
        action="store_true",
        dest="force_labels",
        help="Force regeneration of labels",
    )
    parser.add_argument(
        "-Confidence", "--confidence", dest="conf_float", type=float, default=0.2, help="Model confidence threshold"
    )
    parser.add_argument(
        "-RequireIds",
        "--require-ids",
        default=DEFAULT_MATCH_SELECTORS,
        dest="yolo_class_ints",
        help='YOLO class IDs to keep comma-separated. All classes available in faceless/datasets/OpenImagesV7.yaml. Default: "216,594"',
    )
    parser.add_argument(
        "-Directory",
        "--directory",
        default="faceless",
        dest="dir",
        help="Override output directory. Default: ./faceless",
    )

    args = parser.parse_args(sys.argv[1:])
    
    source = Path(args.path).expanduser().resolve()

    labels = source / "labels"
    destination_root = source / args.dir
    source_files = sorted(path for path in source.iterdir() if path.is_file())

    generate_labels = args.force_labels or not labels.is_dir() or any(
        not (labels / f"{path.stem}.txt").is_file() for path in source_files
    )

    # SOMEDAY: be less niave about whether labels exist or not to allow resume
    model = YOLO(str(download_models(DEFAULT_MODEL_NAME)))

    if generate_labels and source_files:
        labels.mkdir(parents=True, exist_ok=True)
        escaped_source = glob.escape(str(source))
        ends_with_sep = escaped_source.endswith(("/", os.sep))
        source_pattern = f"{escaped_source}{'' if ends_with_sep else os.sep}*.*"
        print(f"Generating labels in {labels}")
        for _ in model.predict( # pyright: ignore[reportOptionalMemberAccess]
            source=source_pattern,
            conf=args.conf_float,
            project=str(source),
            name=".",
            save=False,
            save_txt=True,
            save_conf=True,
            vid_stride=50, # bigger numbers skip more frames of video, lower values analyze more frames
            stream=True,
            exist_ok=True,
            verbose=True,
        ):
            pass # 
    
    def parse_class_ids(selector_text: str) -> set[int]:
        return {int(token.strip()) for token in selector_text.split(",") if token.strip()}

    classes_to_keep = parse_class_ids(args.yolo_class_ints)
    required_face_classes = parse_class_ids(REQUIRED_FACE_SELECTORS)

    label_names: dict[int, str] = {}
    if model is not None:
        if isinstance(model.names, dict):
            label_names = {int(key): str(value) for key, value in model.names.items()}
        else:
            label_names = {index: str(value) for index, value in enumerate(model.names)}

    moved_count = 0
    for source_file in source_files:
        label_path = labels / f"{source_file.stem}.txt"
        label_counts: dict[int, int] = {}

        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                class_id = int(float(line.split()[0].lstrip("\ufeff")))
                label_counts[class_id] = label_counts.get(class_id, 0) + 1

        detected_classes = set(label_counts)
        has_required_match = bool(detected_classes & classes_to_keep)
        has_required_face = bool(detected_classes & required_face_classes)

        if has_required_match and has_required_face:
            continue

        destination_path = destination_root
        if label_counts:
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
