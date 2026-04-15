"""
Single-file ArcFace export extracted from:
https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/81eaf3cafe58b3a20a4c4044c060a00d68247291/app/processors/face_swappers.py

Constraints per request:
- No model management (no load/unload/download/verify).
- One function per ArcFace capability.
- No other helper functions/classes defined.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage import transform as trans
from torchvision.transforms import v2


def arcface_recognize(
    ort_session,
    *,
    arcface_model_name: str,
    device: str,
    img: torch.Tensor,
    face_kps,
    similarity_type: str,
    arcface_dst,
):
    """
    Source reference:
    - https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/81eaf3cafe58b3a20a4c4044c060a00d68247291/app/processors/face_swappers.py#L99-L209

    Compute ArcFace embedding for standard ArcFace models using the same logic as FaceSwappers.recognize.

    Parameters
    ----------
    ort_session:
        Pre-loaded ONNX Runtime session for the ArcFace model (e.g., Inswapper128ArcFace, SimSwapArcFace).
        Must support .get_inputs(), .get_outputs(), .io_binding(), .run_with_iobinding().
    arcface_model_name:
        Used only to select the normalization branch:
        - "Inswapper128ArcFace"
        - "SimSwapArcFace"
        - anything else -> default branch
    device:
        Execution device string used by ORT IOBinding (e.g., "cpu", "cuda", "dml").
    img:
        Torch tensor image in CHW format (C,H,W). dtype may be uint8 or float.
    face_kps:
        5-point face landmarks (shape (5,2)).
    similarity_type:
        "Optimal", "Pearl", or anything else (default branch).
        NOTE: This exported function does NOT depend on app.processors.utils.faceutil;
        so the "Optimal" branch is not implemented and will raise.
    arcface_dst:
        Destination landmarks array used for alignment (shape (5,2)).

    Returns
    -------
    embedding: np.ndarray
        Flattened embedding vector.
    cropped_image: torch.Tensor
        Cropped/aligned face image as HWC tensor (112,112,3) in torch.
    """
    if similarity_type == "Optimal":
        raise NotImplementedError(
            'The "Optimal" branch depends on faceutil.warp_face_by_face_landmark_5 '
            "(app.processors.utils.faceutil), which is intentionally excluded."
        )

    if similarity_type == "Pearl":
        dst = np.array(arcface_dst, copy=True)
        dst[:, 0] += 8.0

        tform = trans.SimilarityTransform()
        tform.estimate(face_kps, dst)

        img = v2.functional.affine(
            img,
            tform.rotation * 57.2958,
            (tform.translation[0], tform.translation[1]),
            tform.scale,
            0,
            center=(0, 0),
        )
        img = v2.functional.crop(img, 0, 0, 128, 128)
        img = v2.Resize((112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(img)
    else:
        tform = trans.SimilarityTransform()
        tform.estimate(face_kps, arcface_dst)

        img = v2.functional.affine(
            img,
            tform.rotation * 57.2958,
            (tform.translation[0], tform.translation[1]),
            tform.scale,
            0,
            center=(0, 0),
        )
        img = v2.functional.crop(img, 0, 0, 112, 112)

    # Model-specific normalization (matches source)
    if arcface_model_name == "Inswapper128ArcFace":
        cropped_image = img.permute(1, 2, 0).clone()
        if img.dtype == torch.uint8:
            img = img.to(torch.float32)
        img = torch.sub(img, 127.5)
        img = torch.div(img, 127.5)
    elif arcface_model_name == "SimSwapArcFace":
        cropped_image = img.permute(1, 2, 0).clone()
        if img.dtype == torch.uint8:
            img = torch.div(img.to(torch.float32), 255.0)
        img = v2.functional.normalize(
            img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=False
        )
    else:
        cropped_image = img.permute(1, 2, 0).clone()
        if img.dtype == torch.uint8:
            img = img.to(torch.float32)
        img = torch.div(img, 127.5)
        img = torch.sub(img, 1)

    img = torch.unsqueeze(img, 0).contiguous()

    input_name = ort_session.get_inputs()[0].name
    output_names = [o.name for o in ort_session.get_outputs()]
    io_binding = ort_session.io_binding()

    io_binding.bind_input(
        name=input_name,
        device_type=device,
        device_id=0,
        element_type=np.float32,
        shape=tuple(img.size()),
        buffer_ptr=img.data_ptr(),
    )
    for name in output_names:
        io_binding.bind_output(name, device)

    ort_session.run_with_iobinding(io_binding)

    embedding = np.array(io_binding.copy_outputs_to_cpu()).flatten()
    return embedding, cropped_image


def cscs_preprocess_image(
    *,
    img: torch.Tensor,
    face_kps,
    FFHQ_kps,
):
    """
    Source reference:
    - https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/81eaf3cafe58b3a20a4c4044c060a00d68247291/app/processors/face_swappers.py#L211-L257

    CSCS ArcFace preprocessing (matches FaceSwappers.preprocess_image_cscs):
    - similarity transform from face_kps -> FFHQ_kps
    - affine + crop 512x512
    - resize to 112x112
    - normalize to mean/std (0.5,0.5,0.5)

    Returns
    -------
    input_tensor: torch.Tensor
        Shape (1,3,112,112), float32-like tensor suitable for ORT binding.
    cropped_image: torch.Tensor
        Shape (112,112,3) HWC, cloned.
    """
    tform = trans.SimilarityTransform()
    tform.estimate(face_kps, FFHQ_kps)

    temp = v2.functional.affine(
        img,
        tform.rotation * 57.2958,
        (tform.translation[0], tform.translation[1]),
        tform.scale,
        0,
        center=(0, 0),
    )
    temp = v2.functional.crop(temp, 0, 0, 512, 512)

    image = v2.Resize((112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)(temp)

    cropped_image = image.permute(1, 2, 0).clone()
    if image.dtype == torch.uint8:
        image = torch.div(image.to(torch.float32), 255.0)

    image = v2.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)

    return torch.unsqueeze(image, 0).contiguous(), cropped_image


def cscs_recognize_id_adapter(
    ort_session_id,
    *,
    device: str,
    img: torch.Tensor,
    face_kps,
    FFHQ_kps,
):
    """
    Source reference:
    - https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/81eaf3cafe58b3a20a4c4044c060a00d68247291/app/processors/face_swappers.py#L299-L354

    CSCS ID adapter embedding (matches FaceSwappers.recognize_cscs_id_adapter).

    If face_kps is not None, img is treated as the ORIGINAL image (CHW) and will be preprocessed.
    If face_kps is None, img is assumed to already be preprocessed as (1,3,112,112).

    Returns
    -------
    embedding_id: np.ndarray
        Flattened, L2-normalized embedding.
    """
    if face_kps is not None:
        # Inline dependency on cscs_preprocess_image (still top-level only, no nested helpers)
        img, _ = cscs_preprocess_image(img=img, face_kps=face_kps, FFHQ_kps=FFHQ_kps)

    io_binding = ort_session_id.io_binding()
    io_binding.bind_input(
        name="input",
        device_type=device,
        device_id=0,
        element_type=np.float32,
        shape=tuple(img.size()),
        buffer_ptr=img.data_ptr(),
    )
    io_binding.bind_output(name="output", device_type=device)

    ort_session_id.run_with_iobinding(io_binding)

    output = io_binding.copy_outputs_to_cpu()[0]
    embedding_id = torch.from_numpy(output).to("cpu")
    embedding_id = torch.nn.functional.normalize(embedding_id, dim=-1, p=2)
    return embedding_id.numpy().flatten()


def cscs_recognize(
    ort_session_arcface,
    ort_session_id,
    *,
    device: str,
    img: torch.Tensor,
    face_kps,
    FFHQ_kps,
):
    """
    Source reference:
    - https://github.com/VisoMasterFusion/VisoMaster-Fusion/blob/81eaf3cafe58b3a20a4c4044c060a00d68247291/app/processors/face_swappers.py#L259-L297

    CSCS ArcFace embedding (matches FaceSwappers.recognize_cscs):
    - preprocess with FFHQ landmarks
    - run CSCSArcFace model
    - L2 normalize
    - add ID adapter embedding

    Returns
    -------
    embedding: np.ndarray
        Combined embedding (arcface + id_adapter), flattened.
    cropped_image: torch.Tensor
        Preprocessed cropped face image (112,112,3) HWC.
    """
    # Inline dependency on cscs_preprocess_image (still top-level only, no nested helpers)
    img_pre, cropped_image = cscs_preprocess_image(img=img, face_kps=face_kps, FFHQ_kps=FFHQ_kps)

    io_binding = ort_session_arcface.io_binding()
    io_binding.bind_input(
        name="input",
        device_type=device,
        device_id=0,
        element_type=np.float32,
        shape=tuple(img_pre.size()),
        buffer_ptr=img_pre.data_ptr(),
    )
    io_binding.bind_output(name="output", device_type=device)

    ort_session_arcface.run_with_iobinding(io_binding)

    output = io_binding.copy_outputs_to_cpu()[0]
    embedding = torch.from_numpy(output).to("cpu")
    embedding = torch.nn.functional.normalize(embedding, dim=-1, p=2)
    embedding = embedding.numpy().flatten()

    # Inline dependency on cscs_recognize_id_adapter (still top-level only, no nested helpers)
    embedding_id = cscs_recognize_id_adapter(
        ort_session_id,
        device=device,
        img=img_pre,
        face_kps=None,
        FFHQ_kps=FFHQ_kps,
    )

    embedding = embedding + embedding_id
    return embedding, cropped_image
