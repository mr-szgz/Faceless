import glob
import os
import shutil
from pathlib import Path

from ultralytics import YOLO
from faceless.models import download_models
from faceless.users import ask_user_yn, open_user_folder

DEFAULT_MODEL_NAME = "yolov8n-oiv7.pt"
REQUIRED_FACE_SELECTORS = "264"
VERSION = "0.6.0"
COPYRIGHT_YEAR = "2026"
IDENTITY_TEXT = f"faceless {VERSION}\nCopyright (c) {COPYRIGHT_YEAR}"
VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}


def _parse_class_ids(selector_text: str) -> set[int]:
    return {int(token.strip()) for token in selector_text.split(",") if token.strip()}


def run_faceless(args) -> None:
    print(IDENTITY_TEXT)
    source_text = args.path_option or args.path
    source = Path(source_text).expanduser().resolve()

    labels = source / "labels"
    destination_root = source / args.directory
    source_files = sorted(path for path in source.iterdir() if path.is_file())

    generate_labels = args.force_labels or not labels.is_dir() or any(
        not (labels / f"{path.stem}.txt").is_file() for path in source_files
    )

    model = None
    # SOMEDAY: be less niave about whether labels exist or not to allow resume
    if generate_labels or args.auto_directory:
        model = YOLO(str(download_models(DEFAULT_MODEL_NAME)))

    if generate_labels and source_files:
        labels.mkdir(parents=True, exist_ok=True)
        escaped_source = glob.escape(str(source))
        source_pattern = f"{escaped_source}{'' if escaped_source.endswith(('/', '\\')) else os.sep}*.*"
        print(f"Generating labels in {labels}")
        vid_stride = 90  # 30 fps is common; 90 gives one frame every 3 seconds.
        for _ in model.predict(  # pyright: ignore[reportOptionalMemberAccess]
            source=source_pattern,
            conf=args.conf,
            project=str(source),
            name=".",
            save=False,
            save_txt=True,
            save_conf=True,
            vid_stride=vid_stride,
            stream=True,
            exist_ok=True,
            verbose=True,
        ):
            pass

    required_match_classes = _parse_class_ids(args.match)
    required_face_classes = _parse_class_ids(REQUIRED_FACE_SELECTORS)

    label_names: dict[int, str] = {}
    if model is not None and args.auto_directory:
        if isinstance(model.names, dict):
            label_names = {int(key): str(value) for key, value in model.names.items()}
        else:
            label_names = {index: str(value) for index, value in enumerate(model.names)}

    moved_count = 0
    for source_file in source_files:
        label_counts: dict[int, int] = {}
        label_files: list[Path] = []
        label_path = labels / f"{source_file.stem}.txt"

        if source_file.suffix.lower() in VIDEO_EXTENSIONS:
            label_files = sorted(labels.glob(f"{source_file.stem}_*.txt"))
            stem_dir = labels / source_file.stem
            if stem_dir.is_dir():
                label_files.extend(sorted(stem_dir.glob("*.txt")))
            if not label_files and label_path.is_file():
                label_files = [label_path]
        elif label_path.is_file():
            label_files = [label_path]

        for label_file in label_files:
            for line in label_file.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                class_id = int(float(line.split()[0].lstrip("\ufeff")))
                label_counts[class_id] = label_counts.get(class_id, 0) + 1

        detected_classes = set(label_counts)
        has_required_match = bool(detected_classes & required_match_classes)
        has_required_face = bool(detected_classes & required_face_classes)

        if has_required_match and has_required_face:
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
    if ask_user_yn("Open output folder?", defaults_yes=False):
        open_user_folder(destination_root)
