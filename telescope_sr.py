"""PaddleOCR Text Telescope (scene text SR) inference wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, np.ndarray, Image.Image]


@dataclass
class TelescopeResult:
    """Before/after tensors as RGB uint8 images (H, W, 3)."""

    original: np.ndarray
    lr: np.ndarray
    sr: np.ndarray
    elapsed_sec: float


def _to_pil_rgb(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        elif arr.shape[2] == 3:
            # assume BGR from OpenCV unless already RGB-looking; caller should pass RGB/PIL
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(arr.astype(np.uint8))
    raise TypeError(f"Unsupported image type: {type(image)}")


def _tensor_to_rgb(chw: np.ndarray) -> np.ndarray:
    """(C,H,W) float [0,1] -> (H,W,3) uint8 RGB."""
    img = np.clip(chw * 255.0, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    return img


def _find_model_files(model_dir: Path) -> tuple[Path, Path]:
    """Return (pdmodel_or_json, pdiparams)."""
    for stem in ("inference", "model"):
        params = model_dir / f"{stem}.pdiparams"
        if not params.exists():
            continue
        for ext in (".json", ".pdmodel"):
            prog = model_dir / f"{stem}{ext}"
            if prog.exists():
                return prog, params
    raise FileNotFoundError(
        f"Inference model not found under {model_dir}. "
        "Run: python scripts/setup_model.py"
    )


class TelescopeSR:
    """Text Telescope super-resolution (Paddle Inference).

    Designed for word/line crops. Default shape is 3x32x128
    (input is bicubic-downsampled to 16x64, then SR to 32x128).
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        image_shape: tuple[int, int, int] = (3, 32, 128),
        use_gpu: bool = False,
        cpu_threads: int = 4,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.image_shape = image_shape
        self.use_gpu = use_gpu
        self.cpu_threads = cpu_threads
        self._predictor = None
        self._input_handle = None
        self._output_handles = None
        self._load()

    def _load(self) -> None:
        from paddle import inference

        prog, params = _find_model_files(self.model_dir)
        config = inference.Config(str(prog), str(params))

        if self.use_gpu:
            config.enable_use_gpu(200, 0)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(self.cpu_threads)
            # MKLDNN can break some older SR graphs on macOS/arm
            try:
                config.disable_mkldnn()
            except Exception:
                pass

        config.switch_ir_optim(True)
        config.enable_memory_optim()
        config.disable_glog_info()
        config.switch_use_feed_fetch_ops(False)

        self._predictor = inference.create_predictor(config)
        input_names = self._predictor.get_input_names()
        self._input_handle = self._predictor.get_input_handle(input_names[0])
        self._output_handles = [
            self._predictor.get_output_handle(name)
            for name in self._predictor.get_output_names()
        ]
        logger.info("Loaded Telescope model from %s", self.model_dir)

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        _, img_h, img_w = self.image_shape
        # Match PaddleOCR predict_sr: downsample by 2 (config down_sample_scale=2)
        lr = img.resize((img_w // 2, img_h // 2), Image.BICUBIC)
        arr = np.array(lr).astype("float32").transpose(2, 0, 1) / 255.0
        return arr[np.newaxis, ...]

    def predict(self, image: ImageInput) -> TelescopeResult:
        import time

        pil = _to_pil_rgb(image)
        original = np.array(pil)
        batch = self._preprocess(pil)

        t0 = time.perf_counter()
        self._input_handle.copy_from_cpu(batch)
        self._predictor.run()
        outputs = [h.copy_to_cpu() for h in self._output_handles]
        elapsed = time.perf_counter() - t0

        # Official predict_sr: outputs[0]=lr, outputs[1]=sr
        if len(outputs) >= 2:
            lr_batch, sr_batch = outputs[0], outputs[1]
        else:
            # fallback: only SR returned
            sr_batch = outputs[0]
            lr_batch = batch

        lr_img = _tensor_to_rgb(lr_batch[0])
        sr_img = _tensor_to_rgb(sr_batch[0])
        return TelescopeResult(
            original=original, lr=lr_img, sr=sr_img, elapsed_sec=elapsed
        )

    def predict_batch(self, images: list[ImageInput]) -> list[TelescopeResult]:
        return [self.predict(img) for img in images]


_cached: Optional[TelescopeSR] = None


def get_telescope(
    model_dir: Union[str, Path],
    use_gpu: bool = False,
    force_reload: bool = False,
) -> TelescopeSR:
    global _cached
    model_dir = Path(model_dir)
    if (
        force_reload
        or _cached is None
        or Path(_cached.model_dir) != model_dir
        or _cached.use_gpu != use_gpu
    ):
        _cached = TelescopeSR(model_dir=model_dir, use_gpu=use_gpu)
    return _cached
