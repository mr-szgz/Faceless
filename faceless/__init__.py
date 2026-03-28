#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
from typing import Literal
import tqdm
from ultralytics import YOLO


def parse_arguments() -> tuple[Literal['yolov8n-oiv7.pt'], set[int], Literal[264], bool, float, Path, Path, Path]:
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
    parser.add_argument("-Directory", "--directory", default=Directory, help="Output directory name for moved files")

    args = parser.parse_args()

    path = args.path_option or args.path
    if path is None:
        parser.error("Source path is required. Use -Path/--path or pass it positionally.")

    ForceLabels = args.force_labels
    Conf = args.conf
    Directory = args.directory

    source: Path = Path(path).expanduser().resolve()
    labels: Path = source / "labels"
    Destination: Path = source / Directory

    return MODEL_NAME, GIRL_OR_WOMAN_CLASSES, HUMAN_FACE_CLASS, ForceLabels, Conf, source, labels, Destination

def main() -> None:
    MODEL_NAME, GIRL_OR_WOMAN_CLASSES, HUMAN_FACE_CLASS, ForceLabels, conf, source, labels, Destination = parse_arguments()
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
        has_girl_or_woman = False
        has_human_face = False

        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts: list[str] = line.split()
                if not parts:
                    continue

                class_id = int(parts[0])

                if class_id in GIRL_OR_WOMAN_CLASSES:
                    has_girl_or_woman = True
                elif class_id == HUMAN_FACE_CLASS:
                    has_human_face = True

                if has_girl_or_woman and has_human_face:
                    break

        if not (has_girl_or_woman and has_human_face):
            Destination.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file_path), str(Destination / file_path.name))

if __name__ == "__main__":
    main()
