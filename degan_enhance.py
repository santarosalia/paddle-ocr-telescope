"""DE-GAN document enhancement wrapper (deblur / binarize for OCR preprocessing)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, np.ndarray, Image.Image]
DeGANTask = Literal["deblur", "binarize", "unwatermark"]

ROOT = Path(__file__).resolve().parent
DEFAULT_DEGAN_ROOT = ROOT / "vendor" / "DE-GAN"

WEIGHT_FILES: dict[DeGANTask, tuple[str, int]] = {
    "deblur": ("deblur_weights.h5", 1024),
    "binarize": ("binarization_generator_weights.h5", 1024),
    "unwatermark": ("watermark_rem_weights.h5", 512),
}


@dataclass
class DeGANResult:
    original: np.ndarray
    enhanced: np.ndarray
    elapsed_sec: float
    task: str


def _to_pil_rgb(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        if arr.shape[2] == 4:
            return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _split_patches(padded: np.ndarray) -> np.ndarray:
    h, w = padded.shape[:2]
    tile = 256
    patches = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            patches.append(padded[y : y + tile, x : x + tile, :])
    return np.array(patches)


def _merge_patches(patches: np.ndarray, h: int, w: int) -> np.ndarray:
    tile = 256
    out = np.zeros((h, w, 1), dtype=patches.dtype)
    idx = 0
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            out[y : y + tile, x : x + tile, :] = patches[idx]
            idx += 1
    return out


def _build_generator(biggest_layer: int):
    from tensorflow.keras.layers import (
        Concatenate,
        Conv2D,
        Dropout,
        Input,
        MaxPooling2D,
        UpSampling2D,
    )
    from tensorflow.keras.models import Model

    inputs = Input((256, 256, 1))
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(biggest_layer // 2, 3, activation="relu", padding="same")(pool3)
    conv4 = Conv2D(biggest_layer // 2, 3, activation="relu", padding="same")(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(biggest_layer, 3, activation="relu", padding="same")(pool4)
    conv5 = Conv2D(biggest_layer, 3, activation="relu", padding="same")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate(axis=-1)([drop4, up6])
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(merge6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(conv6)

    up7 = Conv2D(256, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=-1)([conv3, up7])
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(merge7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(conv7)

    up8 = Conv2D(128, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=-1)([conv2, up8])
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(merge8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(conv8)

    up9 = Conv2D(64, 2, activation="relu", padding="same")(UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=-1)([conv1, up9])
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(merge9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(conv9)
    conv9 = Conv2D(2, 3, activation="relu", padding="same")(conv9)
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    return Model(inputs=inputs, outputs=conv10)


def _weights_ready(degan_root: Path, task: DeGANTask) -> bool:
    filename, _ = WEIGHT_FILES[task]
    return (degan_root / "weights" / filename).is_file()


class DeGANEnhancer:
    """DE-GAN document enhancement (256px tiled inference)."""

    def __init__(
        self,
        degan_root: Union[str, Path] = DEFAULT_DEGAN_ROOT,
        task: DeGANTask = "deblur",
        binarize_thresh: float = 0.95,
    ) -> None:
        self.degan_root = Path(degan_root)
        self.task = task
        self.binarize_thresh = binarize_thresh
        self._generator = None

    def _load(self) -> None:
        if self._generator is not None:
            return
        filename, biggest_layer = WEIGHT_FILES[self.task]
        weights = self.degan_root / "weights" / filename
        if not weights.is_file():
            raise FileNotFoundError(
                f"DE-GAN weights not found: {weights}\nRun: python scripts/setup_degan.py"
            )
        self._generator = _build_generator(biggest_layer)
        self._generator.load_weights(str(weights))
        logger.info("Loaded DE-GAN task=%s from %s", self.task, weights)

    def predict(self, image: ImageInput) -> DeGANResult:
        self._load()
        assert self._generator is not None

        pil = _to_pil_rgb(image)
        original = np.array(pil)
        gray = np.array(pil.convert("L"), dtype=np.float32) / 255.0

        h = ((gray.shape[0] // 256) + 1) * 256
        w = ((gray.shape[1] // 256) + 1) * 256
        padded = np.ones((h, w), dtype=np.float32)
        padded[: gray.shape[0], : gray.shape[1]] = gray
        patches = _split_patches(padded.reshape(h, w, 1))

        t0 = time.perf_counter()
        preds = [
            self._generator.predict(patches[i].reshape(1, 256, 256, 1), verbose=0)
            for i in range(patches.shape[0])
        ]
        enhanced = _merge_patches(np.array(preds), h, w)
        enhanced = enhanced[: gray.shape[0], : gray.shape[1], 0]

        if self.task == "binarize":
            enhanced = (enhanced > self.binarize_thresh).astype(np.float32)

        elapsed = time.perf_counter() - t0
        gray_u8 = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

        return DeGANResult(original=original, enhanced=rgb, elapsed_sec=elapsed, task=self.task)


_cached: dict[tuple[str, str], DeGANEnhancer] = {}


def get_degan(
    degan_root: Union[str, Path] = DEFAULT_DEGAN_ROOT,
    task: DeGANTask = "deblur",
    force_reload: bool = False,
) -> DeGANEnhancer:
    key = (str(Path(degan_root).resolve()), task)
    if force_reload or key not in _cached:
        _cached[key] = DeGANEnhancer(degan_root=degan_root, task=task)
    return _cached[key]


def degan_ready(degan_root: Union[str, Path] = DEFAULT_DEGAN_ROOT, task: DeGANTask = "deblur") -> bool:
    return _weights_ready(Path(degan_root), task)
