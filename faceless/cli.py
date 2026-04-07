#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO
from faceless.config import get_config, DEPENDENCIES_DIR, DEFAULT_YOLO_MATCH_CLASSES, DEFAULT_YOLO_FACE_CLASSES, DEFAULT_MODEL_NAME
from pathlib import Path
from faceless.config import DEPENDENCIES_DIR
from ultralytics.utils.downloads import attempt_download_asset

def main() -> None:
    parser = argparse.ArgumentParser(prog="faceless")
    parser.add_argument(
        "path", nargs="?", help="Source directory containing videos/images"
    )
    conf_float = get_config("detect", "yolo_confidence", 0.2)
    parser.add_argument(
        "-Confidence", "--confidence", dest="conf_float", type=float, default=conf_float, help=f"YOLO confidence"
    )
    yolo_match_classes = get_config("detect", "yolo_match_classes", DEFAULT_YOLO_MATCH_CLASSES)
    parser.add_argument(
        "-RequireIds",
        "--require-ids",
        default=yolo_match_classes,
        dest="yolo_match_classes",
        help=f"YOLO class IDs to keep comma-separated. All classes available in faceless/datasets/OpenImagesV7.yaml",
    )
    
    output_folder = get_config('project', 'output_folder', 'faceless')
    parser.add_argument(
        "-Output",
        "--output",
        default=output_folder,
        dest="output_folder",
        help=f"Override output directory",
    )

    args = parser.parse_args(sys.argv[1:])
    
    sources_path = Path(args.path).expanduser().resolve()

    labels = sources_path / "labels"
    source_files = sorted(path for path in sources_path.iterdir() if path.is_file())

    generate_labels = not labels.is_dir() or any(
        not (labels / f"{path.stem}.txt").is_file() for path in source_files
    )
    
    dependencies = Path(get_config("project", "dependencies", DEPENDENCIES_DIR))

    print(f"downloading models.. {dependencies / 'models'}")
    print(f"use dependency {DEPENDENCIES_DIR / 'models' / DEFAULT_MODEL_NAME}")
    model_name = get_config("detect", "model_name", DEFAULT_MODEL_NAME)
    model_file = Path(attempt_download_asset(DEPENDENCIES_DIR / "models" / model_name)).resolve()
    model = YOLO(model_file)

    if generate_labels and source_files:
        labels.mkdir(parents=True, exist_ok=True)
        escaped_source = glob.escape(str(sources_path))
        ends_with_sep = escaped_source.endswith(("/", os.sep))
        source_pattern = f"{escaped_source}{'' if ends_with_sep else os.sep}*.*"
        print(f"Generating labels in {labels}")
        for _ in model.predict( # pyright: ignore[reportOptionalMemberAccess]
            source=source_pattern,
            conf=args.conf_float,
            project=str(sources_path),
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

    classes_to_keep = parse_class_ids(args.yolo_match_classes)
    required_face_classes = parse_class_ids(DEFAULT_YOLO_FACE_CLASSES)

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
        has_required_id = bool(detected_classes & classes_to_keep)
        has_face_present = bool(detected_classes & required_face_classes)

        if has_required_id and has_face_present:
            continue

        final_destination = sources_path / args.output_folder
        if label_counts:
            top_class_id = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            save_class_name = label_names.get(top_class_id, str(top_class_id))
            final_destination = sources_path / args.output_folder / save_class_name

        final_destination.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_file), str(final_destination / source_file.name))
        moved_count += 1

    print(f"Moved {moved_count} file(s).")


if __name__ == "__main__":
    main()
