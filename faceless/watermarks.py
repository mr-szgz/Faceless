from pathlib import Path

from ultralytics import YOLO
from faceless.config import get_config, DEPENDENCIES_DIR, DEFAULT_YOLO_MATCH_CLASSES, DEFAULT_YOLO_FACE_CLASSES, DEFAULT_MODEL_NAME
# from ultralytics.utils.downloads import attempt_download_asset
# from ultralytics.utils.plotting import save_one_box
from huggingface_hub import hf_hub_download


def predict(sources_path: str | Path, project_path: str | Path, move_dir: str | Path):
    dependencies = Path(str(get_config("project", "dependencies", DEPENDENCIES_DIR)))
    print(f"[info] Configured dependencies {dependencies}")
    sources_path = Path(sources_path)
    project_path = Path(project_path)
    (dependencies / "models" / "yolo11x_watermark_detection").mkdir(exist_ok=True)
    
    move_dir = Path(move_dir)
    model_path = hf_hub_download(
        repo_id="corzent/yolo11x_watermark_detection", 
        repo_type="model", 
        filename="best.pt", 
        local_dir=(dependencies / "models" / "yolo11x_watermark_detection")
    )

    # Load model
    # model_path = attempt_download_asset(dependencies / "models" / "corzent/yolo11x_watermark_detection")
    model = YOLO(model_path)

    results = model(
        source=str(sources_path),
        conf=0.3,
        project=str(project_path),
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
        # TODO: use result
        orig_file = result.path
        if Path(orig_file).exists() and Path(orig_file).is_file():
            pass
        for i, c in enumerate(result.boxes.cls):
            class_int = int(c)
            class_name = model.names.get(class_int)
            print(f"{result.path} {class_name} ({class_int})")
