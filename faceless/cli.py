#!/usr/bin/env python3
import argparse
import glob
import shutil
import sys
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.plotting import save_one_box

from faceless.config import __version__, get_config, DEPENDENCIES_DIR, DEFAULT_YOLO_MATCH_CLASSES, DEFAULT_YOLO_FACE_CLASSES, DEFAULT_MODEL_NAME

MISSES_DIR_NAME = "Unknown"
yolo_supported_images = ['jpeg', 'webp', 'jpg', 'png', 'dng', 'tiff', 'tif', 'jpeg2000', 'heif', 'avif', 'heic', 'jp2', 'bmp', 'mpo']
yolo_supported_videos = ['wmv', 'mov', 'avi', 'm4v', 'ts', 'mkv', 'gif', 'mp4', 'mpeg', 'asf', 'mpg', 'webm']

from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "default": "grey78"
})
console = Console(theme=custom_theme)

def print_app_header():
    import platform
    import torch
    from importlib.metadata import version
    ultralytics_name = version('ultralytics')
    numpy_name = version('numpy')
    insightface_name = version('insightface')
    onnxruntime_name = version('onnxruntime-gpu')
    torch_name = torch.__version__
    cuda_name = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "None"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-Only"
    
    console.rule(f"Faceless v.{__version__}", style="bold steel_blue1")
    console.print(f"analyzes images and video with Ultralytics YOLO and automatically labels and organizes media.", style="dim", justify="center")
    console.print("-", style="dim", justify="center")
    console.print(f"python {platform.python_version()}; {platform.system()}-{platform.machine()}-{platform.architecture()}; torch {torch_name}; CUDA {cuda_name} on {gpu_name}", style="dim", justify="center")
    console.print("-", style="dim", justify="center")
    console.print(f"numpy {numpy_name}; ultralytics {ultralytics_name}; insightface {insightface_name}; onnxruntime {onnxruntime_name};", style="dim", justify="center")
    console.rule(f"", style="bold steel_blue1")

def main() -> None:
    print_app_header()
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
        "-SkipFaces",
        "--skip-faces",
        action="store_true",
        dest="skip_faces",
        help="Skip saving detected faces to a 'Faces' directory"
    )

    args = parser.parse_args(sys.argv[1:])
    
    sources_path = Path(args.path)
    labels_path = sources_path / output_folder / 'labels'
    
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
        model_name = str(get_config("detect", "model_name", DEFAULT_MODEL_NAME))
        model_path = dependencies / 'models' / model_name
        print(f"[label] get model {model_path}")
        model_path = attempt_download_asset(model_path)
        print(f"[label] set-up prediction model {model_name} for {sources_path} with model {model_path}")
        model = YOLO(Path(model_path).resolve())
        model_class_names = {int(key): str(value) for key, value in model.names.items()}

        # labels_path.mkdir(parents=True, exist_ok=True)
        half = False
        batch = 1
        
        import torch
        import os
        print(f"[label] Generating labels in {labels_path} needs_labels={needs_labels}")
        print(f"[label] torch available: {torch.cuda.is_available()} {torch.__version__}")
        print(f"[label] half={half} (FP16 inference)")
        print(f"[label] batch={batch}")
        if not os.access(sources_path, os.R_OK):
            print(f"[label] Warning! {sources_path} is locked or inaccessible.")
        print(f"[label] sourcing media from {str(sources_path)}..")
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
                
                if not os.access(Path(result.path), os.R_OK):
                    print(f"[label] Warning! {sources_path} is locked or inaccessible.")
                
                for i, c in enumerate(boxes.cls):
                    class_int = int(c)
                    detected_classes.add(class_int)
                    detected_labels[class_int] = detected_labels.get(class_int, 0)
                    detected_labels[class_int] += 1
                    # add to results running tally too
                    all_labels[class_int] = all_labels.get(class_int, 0)
                    all_labels[class_int] += 1
                    
                    if class_int in face_classes and not args.skip_faces:
                        orig_file = Path(result.path)
                        # if its an image we can save Face immediately
                        if orig_file.is_file() and orig_file.suffix.lstrip('.').lower() in yolo_supported_images:
                            face_dir = sources_path / "Faces"
                            face_dir.mkdir(exist_ok=True)
                            face_file = face_dir / f"{orig_file.stem}_face{i}{orig_file.suffix}"
                            
                            # TODO: use built in boxes[c].orig_shape (orig_shape	tuple[int, int]	Original image shape in (height, width) format.) read S:\Spaces\Image-Classification\Faceless\.agents\references\ultralytics-docs\ultralytics_boxes.md for details
                            from PIL import Image
                            import hashlib
                            imagedata = Image.open(orig_file)
                            w, h = imagedata.size
                            md5_name = hashlib.md5(imagedata.tobytes()).hexdigest()
                            filename = f"{md5_name}_{w}x{h}{face_file.suffix}"
                            print(f"> Saving Faces {Path(face_dir) / filename}")
                            save_one_box(
                                boxes[i].xyxy,
                                result.orig_img.copy(),
                                file=Path(face_dir) / filename,
                                gain=1.25, # 125% of original bb
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
                    missed_file = sources_path / MISSES_DIR_NAME / Path(result.path).name
                    missed_file.parent.mkdir(exist_ok=True, parents=True)
                    miss_count += 1
                    print(f"> NO labels detected {detected_labels}. Move to {missed_file}")
                    shutil.move(result.path, missed_file)
                    print("")
                    continue
                
                top_classes = [model_class_names[max(detected_labels, key=detected_labels.get)]] # pyright: ignore[reportCallIssue, reportArgumentType]
                top_class_name = top_classes[0]
                
                move_file = sources_path / Path(top_class_name) / Path(result.path).name # pyright: ignore[reportArgumentType]
                move_file.parent.mkdir(exist_ok=True, parents=True)
                moved_count += 1
                print(f"> MOVE to most common label ({move_file})")
                shutil.move(result.path, move_file)
                if move_file.exists():
                    print(f"[ok] {move_file}")
                if Path(result.path).exists():
                    print(f"> [warn] !!FILE WAS NOT MOVED!! {result.path}")
                    missed_moves.append((result.path, move_file))
                print("")
                continue
            except Exception as exc:
                print(f"> [error] FAIL to process {result.path}: {exc}")
                import traceback
                traceback.print_exc()
                error_count += 1
                error_results.append(result)
                print("")
                continue
    print("")
    
    if not args.skip_faces:
        from faceless.insightface import cluster_faces
        cluster_faces(sources_path / 'Faces')

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
