from __future__ import annotations

from pathlib import Path
import shutil
from typing import TypedDict
from tqdm import tqdm
import cv2
import numpy as np
from insightface.app import FaceAnalysis

IMAGE_SUFFIXES = {
    ".jpeg",
    ".webp",
    ".jpg",
    ".png",
    ".dng",
    ".tiff",
    ".tif",
    ".jpeg2000",
    ".heif",
    ".avif",
    ".heic",
    ".jp2",
    ".bmp",
    ".mpo",
}

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

det_size = 640
try:
    app.prepare(ctx_id=0)
except Exception:
    # CPU-only fallback when CUDA initialization is unavailable.
    print(f"> [warn] CUDA FAILED CPU-ONLY MODE")
    app.prepare(ctx_id=-1)

class FaceCluster(TypedDict):
    embeddings: list[np.ndarray]
    paths: list[Path]
    centroid: np.ndarray

def cluster_faces(faces_path: str | Path) -> None:
    faces_path = Path(faces_path)
    threshold = 0.25
    clusters: list[FaceCluster] = []
    scanned_images = 0
    images_with_faces = 0
    unreadable_images = 0
    
    progress = tqdm(sorted(faces_path.rglob("*")), desc="Clustering faces")
    for image_path in progress:
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        
        progress.set_postfix_str(f"scanned={scanned_images}; clusters={len(clusters)}; unreadable={unreadable_images}")

        scanned_images += 1
        progress.set_description_str(f"Face > {image_path}")
        if not image_path.exists():
            raise RuntimeError(f"Image doesnt exist {image_path}")
        
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        min_dim = min(h, w)
        if min_dim < det_size:
            scale = det_size / min_dim
            image = cv2.resize(image, None, fx=scale, fy=scale) # pyright: ignore[reportArgumentType, reportCallIssue]
            h, w = image.shape[:2]
        pad = max(h, w)
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)) # pyright: ignore[reportArgumentType, reportCallIssue]
        faces = app.get(image)
        if not faces:
            app.prepare(ctx_id=0, det_size=(det_size * 2, det_size * 2))
            faces = app.get(image)
            app.prepare(ctx_id=0, det_size=(det_size, det_size))

        if not faces:
            unreadable_images += 1
            progress.set_description_str(f"[warn] no face embedding for {image_path}")
            continue

        images_with_faces += 1

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.normed_embedding.astype(np.float32)

        best_idx = -1
        best_score = -1.0

        for i, cluster in enumerate(clusters):
            score = float(np.dot(emb, cluster["centroid"]))
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= threshold:
            cluster = clusters[best_idx]
            cluster["embeddings"].append(emb)
            cluster["paths"].append(image_path)
            cluster["centroid"] = np.mean(np.stack(cluster["embeddings"]), axis=0)
            cluster["centroid"] /= np.linalg.norm(cluster["centroid"])
        else:
            clusters.append(
                {
                    "embeddings": [emb],
                    "paths": [image_path],
                    "centroid": emb.copy(),
                }
            )
    import hashlib
    for i, cluster in enumerate(clusters, start=1):
        cluster_id = hashlib.md5(cluster["centroid"].tobytes()).hexdigest()[:16]
        person_dir = faces_path / f"person_{cluster_id}"
        person_dir.mkdir(parents=True, exist_ok=True)
        for src in cluster["paths"]:
            print(f"> moving Face into {person_dir} / {src.name}")
            shutil.move(src, person_dir / src.name)
    print("")
    print(
        f"{images_with_faces}/{scanned_images} faces images; "
        f"{unreadable_images} unreadable; "
        f"created {len(clusters)} clusters."
    )
