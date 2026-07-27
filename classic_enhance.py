"""Classical OpenCV preprocess for receipt / document OCR.

Default pipeline (`receipt`):
  original → ×3 upscale → CLAHE → Unsharp Mask → mild denoise
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import cv2
import numpy as np
from PIL import Image

ImageInput = Union[str, Path, np.ndarray, Image.Image]
ClassicMethod = Literal["clahe", "unsharp", "adaptive", "otsu", "receipt"]

CLASSIC_METHODS: tuple[ClassicMethod, ...] = (
    "receipt",
    "clahe",
    "unsharp",
    "adaptive",
    "otsu",
)

UPSCALE_FACTOR = 3


@dataclass
class ClassicResult:
    original: np.ndarray
    enhanced: np.ndarray
    elapsed_sec: float
    method: str
    scale: float = 1.0


def _to_rgb(image: ImageInput) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, (str, Path)):
        return np.array(Image.open(image).convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr.copy()


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _upscale(rgb: np.ndarray, scale: int = UPSCALE_FACTOR) -> np.ndarray:
    if scale == 1:
        return rgb
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def _apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def _apply_unsharp(gray: np.ndarray, amount: float = 1.5, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _apply_mild_denoise(gray: np.ndarray) -> np.ndarray:
    """Weak denoise that preserves stroke edges (bilateral, small sigma)."""
    return cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)


def _apply_adaptive(gray: np.ndarray, block: int = 31, c: int = 10) -> np.ndarray:
    block = max(3, block | 1)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        c,
    )


def _apply_otsu(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _apply_receipt(rgb: np.ndarray, clahe_clip: float = 2.0) -> tuple[np.ndarray, float]:
    """×3 upscale → CLAHE → Unsharp → mild denoise."""
    up = _upscale(rgb, UPSCALE_FACTOR)
    gray = _to_gray(up)
    gray = _apply_clahe(gray, clip_limit=clahe_clip, tile=8)
    gray = _apply_unsharp(gray, amount=1.5, sigma=1.0)
    gray = _apply_mild_denoise(gray)
    return _gray_to_rgb(gray), float(UPSCALE_FACTOR)


def enhance_array(
    rgb: np.ndarray,
    method: ClassicMethod = "receipt",
    *,
    clahe_clip: float = 2.0,
    adaptive_block: int = 31,
    adaptive_c: int = 10,
) -> tuple[np.ndarray, float]:
    """Return (enhanced_rgb, scale_vs_original)."""
    if method == "receipt":
        return _apply_receipt(rgb, clahe_clip=clahe_clip)

    gray = _to_gray(rgb)
    if method == "clahe":
        out = _apply_clahe(gray, clip_limit=clahe_clip)
    elif method == "unsharp":
        out = _apply_unsharp(gray)
    elif method == "adaptive":
        out = _apply_adaptive(gray, block=adaptive_block, c=adaptive_c)
    elif method == "otsu":
        out = _apply_otsu(gray)
    else:
        raise ValueError(f"Unknown classic method: {method}")
    return _gray_to_rgb(out), 1.0


class ClassicEnhancer:
    """Stateless classical preprocessor."""

    def __init__(
        self,
        method: ClassicMethod = "receipt",
        *,
        clahe_clip: float = 2.0,
        adaptive_block: int = 31,
        adaptive_c: int = 10,
    ) -> None:
        self.method = method
        self.clahe_clip = clahe_clip
        self.adaptive_block = adaptive_block
        self.adaptive_c = adaptive_c

    def predict(self, image: ImageInput) -> ClassicResult:
        original = _to_rgb(image)
        t0 = time.perf_counter()
        enhanced, scale = enhance_array(
            original,
            self.method,
            clahe_clip=self.clahe_clip,
            adaptive_block=self.adaptive_block,
            adaptive_c=self.adaptive_c,
        )
        elapsed = time.perf_counter() - t0
        return ClassicResult(
            original=original,
            enhanced=enhanced,
            elapsed_sec=elapsed,
            method=self.method,
            scale=scale,
        )


_cached: dict[tuple, ClassicEnhancer] = {}


def get_classic(
    method: ClassicMethod = "receipt",
    *,
    clahe_clip: float = 2.0,
    adaptive_block: int = 31,
    adaptive_c: int = 10,
    force_reload: bool = False,
) -> ClassicEnhancer:
    key = (method, clahe_clip, adaptive_block, adaptive_c)
    if force_reload or key not in _cached:
        _cached[key] = ClassicEnhancer(
            method=method,
            clahe_clip=clahe_clip,
            adaptive_block=adaptive_block,
            adaptive_c=adaptive_c,
        )
    return _cached[key]
