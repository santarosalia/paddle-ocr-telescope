"""Detect → crop → Text Telescope SR → recognize (per text box)."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

ImageInput = Union[str, np.ndarray, Image.Image]

# lang → PaddleOCR TextRecognition model_name
REC_MODEL_BY_LANG = {
    "korean": "korean_PP-OCRv5_mobile_rec",
    "en": "en_PP-OCRv5_mobile_rec",
    "ch": "PP-OCRv5_server_rec",
    "chinese": "PP-OCRv5_server_rec",
    "japan": "japan_PP-OCRv5_mobile_rec",
}


@dataclass
class BoxResult:
    index: int
    poly: list[list[float]]
    lr_crop: np.ndarray
    sr_crop: Optional[np.ndarray] = None
    lr_text: str = ""
    lr_score: float = 0.0
    sr_text: Optional[str] = None
    sr_score: Optional[float] = None
    sr_elapsed_sec: Optional[float] = None
    error: Optional[str] = None


@dataclass
class DetSrRecResult:
    image: np.ndarray
    annotated: np.ndarray
    boxes: list[BoxResult] = field(default_factory=list)
    elapsed_sec: float = 0.0
    det_count: int = 0


def _to_rgb_array(image: ImageInput) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, str):
        return np.array(Image.open(image).convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr.copy()


def _payload(result: Any) -> dict:
    payload: Any = result
    if hasattr(payload, "json"):
        payload = payload.json
    if isinstance(payload, dict) and "res" in payload:
        payload = payload["res"]
    if hasattr(payload, "keys") and not isinstance(payload, dict):
        payload = dict(payload)
    return payload if isinstance(payload, dict) else {}


def create_text_detector(
    *,
    unclip_ratio: float = 1.5,
    box_thresh: Optional[float] = None,
    thresh: Optional[float] = None,
    model_name: Optional[str] = None,
):
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    from paddleocr import TextDetection

    kwargs: dict[str, Any] = {
        "unclip_ratio": unclip_ratio,
        "enable_mkldnn": False,
    }
    if box_thresh is not None:
        kwargs["box_thresh"] = box_thresh
    if thresh is not None:
        kwargs["thresh"] = thresh
    if model_name:
        kwargs["model_name"] = model_name
    return TextDetection(**kwargs)


def create_text_recognizer(lang: str = "korean", model_name: Optional[str] = None):
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    from paddleocr import TextRecognition

    name = model_name or REC_MODEL_BY_LANG.get(lang, REC_MODEL_BY_LANG["korean"])
    return TextRecognition(model_name=name, enable_mkldnn=False)


def detect_polys(detector: Any, image_rgb: np.ndarray) -> list[np.ndarray]:
    pages = list(detector.predict(image_rgb))
    if not pages:
        return []
    payload = _payload(pages[0])
    polys = payload.get("dt_polys") or payload.get("boxes") or []
    out: list[np.ndarray] = []
    for poly in polys:
        arr = np.asarray(poly, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 4:
            out.append(arr[:4])
    return out


def recognize_patch(recognizer: Any, crop_rgb: np.ndarray) -> tuple[str, float]:
    if crop_rgb.size == 0 or min(crop_rgb.shape[:2]) < 2:
        return "", 0.0
    pages = list(recognizer.predict(crop_rgb))
    if not pages:
        return "", 0.0
    payload = _payload(pages[0])
    if "rec_text" in payload:
        return str(payload.get("rec_text") or ""), float(payload.get("rec_score") or 0.0)
    texts = payload.get("rec_texts") or []
    scores = payload.get("rec_scores") or []
    if texts:
        return str(texts[0]), float(scores[0]) if scores else 0.0
    return "", 0.0


def crop_quad(image_rgb: np.ndarray, points: np.ndarray, pad_ratio: float = 0.05) -> np.ndarray:
    """Perspective-crop a 4-point text box (PaddleOCR-style), with optional expand."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2).copy()
    if pad_ratio > 0:
        center = pts.mean(axis=0)
        pts = center + (pts - center) * (1.0 + pad_ratio)

    width = int(
        max(
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[2] - pts[3]),
        )
    )
    height = int(
        max(
            np.linalg.norm(pts[0] - pts[3]),
            np.linalg.norm(pts[1] - pts[2]),
        )
    )
    width = max(width, 1)
    height = max(height, 1)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(
        image_rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.shape[0] >= crop.shape[1] * 1.5:
        crop = np.rot90(crop)
    return crop


def annotate_boxes(
    image_rgb: np.ndarray,
    boxes: list[BoxResult],
    *,
    use_sr_label: bool = True,
) -> np.ndarray:
    """Draw numbered boxes + short recognized text on a copy of the image."""
    canvas = image_rgb.copy()
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box in boxes:
        poly = [(float(x), float(y)) for x, y in box.poly]
        color = (0, 180, 80) if (box.sr_text is not None and not box.error) else (220, 60, 60)
        draw.line(poly + [poly[0]], fill=color, width=2)
        label_text = box.sr_text if use_sr_label and box.sr_text is not None else box.lr_text
        label = f"#{box.index} {label_text[:18]}"
        x, y = poly[0]
        draw.rectangle([x, max(0, y - 14), x + 8 * len(label), y], fill=color)
        draw.text((x + 2, max(0, y - 13)), label, fill=(255, 255, 255), font=font)
    return np.array(pil)


def run_det_sr_rec(
    image: ImageInput,
    detector: Any,
    recognizer: Any,
    telescope: Optional[Any],
    *,
    pad_ratio: float = 0.05,
    skip_sr: bool = False,
    max_boxes: Optional[int] = None,
) -> DetSrRecResult:
    """Full per-box pipeline: detect → crop → (optional SR) → recognize."""
    rgb = _to_rgb_array(image)
    t0 = time.perf_counter()
    polys = detect_polys(detector, rgb)
    if max_boxes is not None:
        polys = polys[: max(0, int(max_boxes))]

    boxes: list[BoxResult] = []
    for idx, poly in enumerate(polys, start=1):
        try:
            lr_crop = crop_quad(rgb, poly, pad_ratio=pad_ratio)
            lr_text, lr_score = recognize_patch(recognizer, lr_crop)
            item = BoxResult(
                index=idx,
                poly=poly.tolist(),
                lr_crop=lr_crop,
                lr_text=lr_text,
                lr_score=lr_score,
            )
            if not skip_sr and telescope is not None:
                sr_out = telescope.predict(Image.fromarray(lr_crop))
                item.sr_crop = sr_out.sr
                item.sr_elapsed_sec = sr_out.elapsed_sec
                sr_text, sr_score = recognize_patch(recognizer, sr_out.sr)
                item.sr_text = sr_text
                item.sr_score = sr_score
            boxes.append(item)
        except Exception as exc:  # noqa: BLE001 — keep going on other boxes
            logger.exception("box %s failed", idx)
            boxes.append(
                BoxResult(
                    index=idx,
                    poly=np.asarray(poly, dtype=float).tolist(),
                    lr_crop=np.zeros((1, 1, 3), dtype=np.uint8),
                    error=str(exc),
                )
            )

    elapsed = time.perf_counter() - t0
    annotated = annotate_boxes(rgb, boxes, use_sr_label=not skip_sr)
    return DetSrRecResult(
        image=rgb,
        annotated=annotated,
        boxes=boxes,
        elapsed_sec=elapsed,
        det_count=len(polys),
    )
