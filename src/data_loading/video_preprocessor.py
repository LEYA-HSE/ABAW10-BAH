# video_preprocessor.py
# coding: utf-8
from __future__ import annotations
import os
import cv2
import torch
import numpy as np
from typing import Optional, Tuple, Sequence, Literal
from ultralytics import YOLO

# Lazy YOLO initialization.
_YOLO: Optional[YOLO] = None

def _lazy_yolo(weights_path: str) -> YOLO:
    global _YOLO
    if _YOLO is None:
        _YOLO = YOLO(weights_path)
    return _YOLO

# Utilities.

def select_uniform_frames(frames: Sequence[int], N: int) -> list[int]:
    N = int(N)
    if N <= 0 or len(frames) <= N:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, num=N, dtype=int)
    return [frames[i] for i in idx]


def _to_pixel_values(image_rgb: np.ndarray, image_processor, device: str) -> Optional[torch.Tensor]:
    if image_rgb is None or image_rgb.size == 0 or image_rgb.ndim != 3:
        return None
    if hasattr(image_processor, "to_pixel_values"):
        pv = image_processor.to_pixel_values(image_rgb)
        return pv.to(device) if isinstance(pv, torch.Tensor) else pv
    inputs = image_processor(images=image_rgb, return_tensors="pt")
    pv = inputs["pixel_values"]
    return pv.to(device) if isinstance(pv, torch.Tensor) else pv


def _ultra_device_arg(device: str):
    """Ultralytics expects cuda index (0/1/..) or 'cpu'/None."""
    if str(device).lower().startswith("cuda"):
        return 0
    return "cpu"


def _largest_box_xyxy(results) -> Optional[tuple[int, int, int, int]]:
    """Return bbox (x1,y1,x2,y2) of the largest detected object, or None."""
    if not results:
        return None
    r0 = results[0]
    if not hasattr(r0, "boxes") or r0.boxes is None or len(r0.boxes) == 0:
        return None
    xyxy = r0.boxes.xyxy  # [N,4]
    if xyxy is None or xyxy.numel() == 0:
        return None
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = int(torch.argmax(areas).item())
    x1, y1, x2, y2 = xyxy[idx].int().cpu().tolist()
    return x1, y1, x2, y2


def _run_yolo(
    model: YOLO,
    im_rgb: np.ndarray,
    *,
    mode: Literal["stable", "fast"],
    device_arg,
    imgsz: int,
    conf: float,
    iou: float,
    augment: bool,
):
    """
    YOLO runner without fallbacks:
      - mode="stable" ? track(persist=True); if tracking unavailable/error ? raise
      - mode="fast"   ? predict(); if error ? raise
    """
    if mode == "stable":
        if not hasattr(model, "track"):
            raise RuntimeError(
                "YOLO.track is unavailable for mode='stable' (upgrade ultralytics or use mode='fast')."
            )
        return model.track(
            im_rgb,
            persist=True,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            augment=augment,
            device=device_arg,
            verbose=False,
        )

    if mode != "fast":
        raise ValueError(
            f"Unknown YOLO mode: {mode!r} (expected 'stable' or 'fast')."
        )
    return model.predict(
        im_rgb,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device_arg,
        verbose=False,
    )


def _reset_yolo_tracker(model: YOLO) -> None:
    """
    Reset tracker state between videos (if present).
    If tracker structure is missing, do nothing.
    """
    pred = getattr(model, "predictor", None)
    if pred is None:
        return
    trackers = getattr(pred, "trackers", None)
    tracker = getattr(pred, "tracker", None)
    if isinstance(trackers, (list, tuple)) and trackers and hasattr(trackers[0], "reset"):
        trackers[0].reset()
    elif tracker is not None and hasattr(tracker, "reset"):
        tracker.reset()

# Main face extractor.
@torch.no_grad()
def get_face_pixel_values(
    video_path: str,
    segment_length: int,
    image_processor,                # CLIPProcessor | AutoImageProcessor
    *,
    device: str = "cuda",
    yolo_weights: str = "src/data_loading/yolov8n-face.pt",
    mode: Literal["stable", "fast"] = "stable",   # strict mode selection
    yolo_conf: float = 0.01,
    yolo_iou: float = 0.5,
    yolo_imgsz: int = 640,
    yolo_augment: bool = False,
    debug_dir: Optional[str] = None,
    debug_every: int = 1,
    debug_max: int = 0,
) -> Tuple[str, Optional[torch.Tensor]]:
    """
    Returns: (video_name, face_pixel_values [T,3,H,W] | None)
    Logic:
      1) sample segment_length frames uniformly
      2) for each frame, take the largest face bbox:
         - mode='stable': YOLO.track(persist=True)
         - mode='fast':   YOLO.predict
      3) if no bbox, fall back to the full frame
      4) reset tracker between videos (stable mode)
    """
    model = _lazy_yolo(yolo_weights)
    device_arg = _ultra_device_arg(device)

    if mode == "stable":
        _reset_yolo_tracker(model)

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    if not cap.isOpened():
        logging.warning(f"[get_face_pixel_values] Failed to open video: {video_path}")
        return video_name, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames > 0:
        need = set(select_uniform_frames(range(total_frames), segment_length))
    else:
        logging.warning(f"[get_face_pixel_values] total_frames=0 for {video_path}. Falling back to first frames.")
        need = None

    batches = []
    dbg_saved = 0
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    t = 0
    try:
        while True:
            ok, im0 = cap.read()
            if not ok:
                break
            use_frame = (need is None and t < segment_length) or (need is not None and t in need)
            if use_frame:
                im_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

                results = _run_yolo(
                    model,
                    im_rgb,
                    mode=mode,
                    device_arg=device_arg,
                    imgsz=yolo_imgsz,
                    conf=yolo_conf,
                    iou=yolo_iou,
                    augment=yolo_augment,
                )

                pv = None
                box = _largest_box_xyxy(results)
                if box is not None:
                    x1, y1, x2, y2 = box
                    # clamp bbox to image bounds
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    x2 = min(x2, im_rgb.shape[1])
                    y2 = min(y2, im_rgb.shape[0])
                    if x2 > x1 and y2 > y1:
                        roi = im_rgb[y1:y2, x1:x2]
                        pv = _to_pixel_values(roi, image_processor, device)

                # optional debug save
                if debug_dir and (debug_max <= 0 or dbg_saved < debug_max):
                    if debug_every <= 1 or (dbg_saved % debug_every == 0):
                        dbg_img = im_rgb.copy()
                        if box is not None:
                            x1, y1, x2, y2 = box
                            cv2.rectangle(dbg_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            tag = "face"
                        else:
                            tag = "no_face"
                            cv2.putText(
                                dbg_img,
                                "NO_FACE",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (255, 0, 0),
                                2,
                                cv2.LINE_AA,
                            )
                        fname = f"{os.path.splitext(video_name)[0]}_t{t:06d}_{tag}.jpg"
                        out_path = os.path.join(debug_dir, fname)
                        cv2.imwrite(out_path, cv2.cvtColor(dbg_img, cv2.COLOR_RGB2BGR))
                        dbg_saved += 1

                # business-logic fallback: no bbox ? full frame
                if pv is None:
                    pv = _to_pixel_values(im_rgb, image_processor, device)

                if pv is not None:
                    batches.append(pv)  # [1,3,H,W]
            t += 1
    finally:
        cap.release()

    face_tensor = torch.cat(batches, dim=0) if batches else None  # [T,3,H,W] or None
    return video_name, face_tensor
