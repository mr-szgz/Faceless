from pathlib import Path

from ultralytics import YOLO
from faceless.config import get_config, DEPENDENCIES_DIR, DEFAULT_YOLO_MATCH_CLASSES, DEFAULT_YOLO_FACE_CLASSES, DEFAULT_MODEL_NAME
from ultralytics.utils.downloads import attempt_download_asset
# from ultralytics.utils.plotting import save_one_box


def predict(sources_path: str | Path, output_folder: str | Path, move_dir: str | Path):
    dependencies = Path(str(get_config("project", "dependencies", DEPENDENCIES_DIR)))
    print(f"[info] Configured dependencies {dependencies}")
    sources_path = Path(sources_path)
    output_folder = Path(output_folder)
    move_dir = Path(move_dir)
        
    # Load model
    model_path = attempt_download_asset(dependencies / "models" / "corzent/yolo11x_watermark_detection")
    model = YOLO(model_path)

    results = model(
        source=str(sources_path),
        conf=0.3,
        project=str(sources_path / output_folder),
        name=".",
        save=False,
        save_txt=True,
        save_conf=True,
        save_crop=True,
        vid_stride=600,
        batch=1,
        # half=True,
        stream=True,
        exist_ok=True,
        verbose=True,
    )

    for result in results:
        for i, c in enumerate(result.boxes.cls):
            class_int = int(c)
            class_name = model.names.get(class_int)
            print(f"{result.path} {class_name} ({class_int})")
