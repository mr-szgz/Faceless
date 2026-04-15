#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO
from faceless.config import get_config, DEPENDENCIES_DIR, DEFAULT_YOLO_MATCH_CLASSES, DEFAULT_YOLO_FACE_CLASSES, DEFAULT_MODEL_NAME
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.plotting import save_one_box

from faceless.arcfaces import arcface_recognize

MISSES_DIR_NAME = "Unknown"
yolo_supported_images = ['jpeg', 'webp', 'jpg', 'png', 'dng', 'tiff', 'tif', 'jpeg2000', 'heif', 'avif', 'heic', 'jp2', 'bmp', 'mpo']
yolo_supported_videos = ['wmv', 'mov', 'avi', 'm4v', 'ts', 'mkv', 'gif', 'mp4', 'mpeg', 'asf', 'mpg', 'webm']

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
    
    output_folder = str(get_config('project', 'output_folder', 'faceless'))
    parser.add_argument(
        "-Output",
        "--output",
        default=output_folder,
        dest="output_folder",
        help=f"Override output directory",
    )
    
    parser.add_argument(
        "-Faces",
        "--faces",
        action="store_true",
        dest="faces",
        help="Save detected faces to a 'Faces' directory"
    )

    args = parser.parse_args(sys.argv[1:])
    
    sources_path = Path(args.path).expanduser().resolve()
    labels_path = sources_path / output_folder / "labels"
    
    dependencies = Path(str(get_config("project", "dependencies", DEPENDENCIES_DIR)))
    print(f"[info] Configured dependencies {dependencies}")
    
    needs_labels = True
    if labels_path.exists() and any(labels_path.glob("*.txt")):
        print(f"[info] Found existing labels in {labels_path}")
        if _confirm("[input] Would you like to remove and re-generate labels? (or use existing)", default=False):
            shutil.rmtree(labels_path)
            needs_labels = True
        else:
            needs_labels = False
    
    model_name = str(get_config("detect", "model_name", DEFAULT_MODEL_NAME))
    model_path = attempt_download_asset(dependencies / "models" / model_name)
    print(f"[ok] model {model_path} ")
    model = YOLO(Path(model_path).resolve())
    model_class_names = {int(key): str(value) for key, value in model.names.items()}
    
    # super lazy get config and split csv class ints
    face_classes = {int(part.strip()) for part in str(get_config("detect", "yolo_face_classes", DEFAULT_YOLO_FACE_CLASSES)).split(",") if part.strip()}
    match_classes =  {int(part.strip()) for part in  str(get_config("detect", "yolo_match_classes", DEFAULT_YOLO_MATCH_CLASSES)).split(",") if part.strip()}
    
    moved_count = 0
    missed_moves = []
    miss_count = 0
    match_count = 0
    error_count = 0
    error_results = []
    all_labels: dict[int, int] = {}
    
    if needs_labels:
        print(f"[info] Generating labels in {labels_path} needs_labels={needs_labels}")

        # labels_path.mkdir(parents=True, exist_ok=True)
        half = False
        batch = 1
        
        import torch
        print(f"[check] Torch available: {torch.cuda.is_available()} {torch.__version__}")
        # if torch.cuda.is_available():
        #     half = True
        #     batch = 2
        print(f"[debug] half={half} (FP16 inference)")
        print(f"[debug] batch={batch}")
        results = model(
            source=str(sources_path),
            conf=args.conf_float,
            project=str(sources_path / output_folder),
            name=".",
            save=False,
            save_txt=True,
            save_conf=True,
            save_crop=True,
            vid_stride=1200,
            batch=batch,
            half=half,
            stream=True,
            exist_ok=True,
            verbose=True,
        )
        for result in results:
            try:
                detected_classes = set()
                detected_labels: dict[int, int] = {}
                
                boxes = result.boxes  # Boxes object for bounding box outputs
                # speed = result.speed
                # (Path(result.save_dir) / "speeds").mkdir(exist_ok=True)
                # speeds_file = (Path(result.save_dir) / "speeds" / (Path(result.path).stem + ".json"))
                # print(f"> saving speeds {speeds_file}")
                # speeds_file.write_text(json.dumps(speed))
        
                for i, c in enumerate(boxes.cls):
                    class_int = int(c)
                    detected_classes.add(class_int)
                    detected_labels[class_int] = detected_labels.get(class_int, 0)
                    detected_labels[class_int] += 1
                    # add to results running tally too
                    all_labels[class_int] = all_labels.get(class_int, 0)
                    all_labels[class_int] += 1
                    
                    if class_int in face_classes and args.faces:
                        orig_file = Path(result.path)
                        if orig_file.is_file() and orig_file.suffix.lstrip('.').lower() in yolo_supported_images:
                            face_dir = sources_path / "Faces"
                            face_dir.mkdir(exist_ok=True)
                            file = face_dir / f"{orig_file.stem}_face{i}{orig_file.suffix}"
                            print(f"> Saving Faces {file}")
                            save_one_box(
                                boxes[i].xyxy,
                                result.orig_img.copy(),
                                file=file,
                                gain=1.2, # 120% of original bb
                                square=True,
                                BGR=True,
                                save=True
                            )
                            

                print(f"> detected_classes {detected_classes} in face_classes {face_classes} and match_classes {match_classes}")
                if bool(face_classes & detected_classes) and bool(match_classes & detected_classes):
                    match_count += 1
                    print(f"> OK!")
                    print("")
                    continue
                
                if not detected_labels:
                    missed_file = Path(sources_path) / MISSES_DIR_NAME / Path(result.path).name
                    missed_file.parent.mkdir(exist_ok=True, parents=True)
                    miss_count += 1
                    print(f"> NO labels detected {detected_labels}. Move to {missed_file}")
                    shutil.move(result.path, missed_file)
                    print("")
                    continue
                
                top_classes = [model_class_names[max(detected_labels, key=detected_labels.get)]] # pyright: ignore[reportCallIssue, reportArgumentType]
                top_class_name = top_classes[0]
                
                move_file = Path(sources_path) / Path(top_class_name) / Path(result.path).name # pyright: ignore[reportArgumentType]
                move_file.parent.mkdir(exist_ok=True, parents=True)
                moved_count += 1
                print(f"> MOVE to most common label ({move_file})")
                shutil.move(result.path, move_file)
                if move_file.exists():
                    print(f"> confirmed move file {move_file}")
                if Path(result.path).exists():
                    print(f"!!FILE WAS NOT MOVED!! {str(Path(result.path))}")
                    missed_moves.append((result.path, move_file))
                print("")
                continue
            except Exception as exc:
                print(f"[error] FAIL to process {result.path}: {exc}")
                error_count += 1
                error_results.append(result)
                print("")
                continue
    print("")
    if missed_moves and len(missed_moves) > 0:
        for m in missed_moves:
            result_path, move_file = m
            print(f"> MOVE retry ({move_file})")
            shutil.move(result_path, move_file)
        
    print(f"{moved_count} organized into {len(all_labels)} labels")
    print(f"{miss_count} without labels (moved to {sources_path / MISSES_DIR_NAME})")
    print(f"{error_count} skipped due to errors")

def _confirm(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        reply = input(f"\n\t{prompt}{suffix} ").strip().lower()
        print("")
        if not reply:
            return default
        if reply in {"y", "yes"}:
            return True
        if reply in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")

def _input_prompt(prompt: str, default: str | None = None) -> str | None:
    suffix = f" [{default}]" if default is not None else ""
    reply = input(f"\n\t{prompt}{suffix} ").strip()
    print("")
    if not reply:
        return default
    return reply

if __name__ == "__main__":
    main()
