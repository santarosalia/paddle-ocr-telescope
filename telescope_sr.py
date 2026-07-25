"""PaddleOCR Text Telescope (scene text SR) inference wrapper."""

from __future__ import annotations

import logging
import time
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
    tile_count: int = 1


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
    """(C,H,W) float -> (H,W,3) uint8 RGB. Handles tanh [-1,1] or [0,1]."""
    if float(chw.min()) < -0.05:
        chw = (chw + 1.0) * 0.5
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


def _pad_to_cover(
    img: np.ndarray, tile_h: int, tile_w: int, stride_h: int, stride_w: int
) -> tuple[np.ndarray, int, int]:
    """Pad so sliding window with stride covers the full image."""
    h, w = img.shape[:2]
    out_h = h if h <= tile_h else tile_h + ((h - tile_h + stride_h - 1) // stride_h) * stride_h
    out_w = w if w <= tile_w else tile_w + ((w - tile_w + stride_w - 1) // stride_w) * stride_w
    out_h = max(out_h, tile_h)
    out_w = max(out_w, tile_w)
    if out_h == h and out_w == w:
        return img, h, w
    padded = np.pad(
        img,
        ((0, out_h - h), (0, out_w - w), (0, 0)),
        mode="edge",
    )
    return padded, h, w


class TelescopeSR:
    """Text Telescope super-resolution (Paddle Inference).

    Fixed patch: LR 16×64 → SR 32×128 (×2).
    ``predict_full`` tiles a whole LR image (e.g. receipt) and stitches ×2 SR.
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

    @property
    def lr_tile_hw(self) -> tuple[int, int]:
        _, img_h, img_w = self.image_shape
        return img_h // 2, img_w // 2

    @property
    def sr_tile_hw(self) -> tuple[int, int]:
        _, img_h, img_w = self.image_shape
        return img_h, img_w

    def _load(self) -> None:
        from paddle import inference

        prog, params = _find_model_files(self.model_dir)
        config = inference.Config(str(prog), str(params))

        if self.use_gpu:
            config.enable_use_gpu(200, 0)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(self.cpu_threads)
            # MKLDNN / oneDNN can break some SR graphs on macOS/arm
            for disable in ("disable_mkldnn", "disable_onednn"):
                fn = getattr(config, disable, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass

        config.switch_ir_optim(True)
        # Paddle 3.x + macOS: enable_memory_optim(True) raises
        # "Not find predictor_id ... memory_optimize_pass"
        config.enable_memory_optim(False)
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

    def _patch_to_nchw(self, patch_rgb: np.ndarray) -> np.ndarray:
        tile_h, tile_w = self.lr_tile_hw
        if patch_rgb.shape[0] != tile_h or patch_rgb.shape[1] != tile_w:
            patch_rgb = np.array(
                Image.fromarray(patch_rgb).resize((tile_w, tile_h), Image.BICUBIC)
            )
        return patch_rgb.astype("float32").transpose(2, 0, 1) / 255.0

    def _run_batch(self, batch_nchw: np.ndarray) -> list[np.ndarray]:
        """NCHW float32 -> list of HWC uint8 SR tiles."""
        self._input_handle.copy_from_cpu(batch_nchw)
        self._predictor.run()
        sr_batch = self._output_handles[0].copy_to_cpu()
        return [_tensor_to_rgb(sr_batch[i]) for i in range(sr_batch.shape[0])]

    def predict(self, image: ImageInput) -> TelescopeResult:
        """Single patch: resize upload to LR tile, return one SR tile (demo/crop)."""
        pil = _to_pil_rgb(image)
        original = np.array(pil)
        tile_h, tile_w = self.lr_tile_hw
        lr_img = np.array(pil.resize((tile_w, tile_h), Image.BICUBIC), dtype=np.uint8)

        t0 = time.perf_counter()
        batch = self._patch_to_nchw(lr_img)[np.newaxis, ...]
        sr_img = self._run_batch(batch)[0]
        elapsed = time.perf_counter() - t0

        return TelescopeResult(
            original=original, lr=lr_img, sr=sr_img, elapsed_sec=elapsed, tile_count=1
        )

    def predict_full(
        self,
        image: ImageInput,
        overlap: int = 8,
        batch_size: int = 8,
        max_side: Optional[int] = None,
    ) -> TelescopeResult:
        """Treat upload as LR; tile whole image and stitch ×2 SR (receipts, etc.)."""
        pil = _to_pil_rgb(image)
        if max_side is not None:
            w, h = pil.size
            long_side = max(w, h)
            if long_side > max_side:
                scale = max_side / long_side
                pil = pil.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.BICUBIC,
                )

        lr = np.array(pil, dtype=np.uint8)
        original = lr.copy()
        tile_h, tile_w = self.lr_tile_hw
        sr_h, sr_w = self.sr_tile_hw
        scale = sr_h // tile_h  # 2

        overlap = int(np.clip(overlap, 0, min(tile_h, tile_w) - 1))
        stride_h = max(1, tile_h - overlap)
        stride_w = max(1, tile_w - overlap)

        padded, orig_h, orig_w = _pad_to_cover(lr, tile_h, tile_w, stride_h, stride_w)
        ph, pw = padded.shape[:2]

        ys = list(range(0, ph - tile_h + 1, stride_h))
        xs = list(range(0, pw - tile_w + 1, stride_w))
        if not ys:
            ys = [0]
        if not xs:
            xs = [0]
        # ensure bottom-right coverage
        if ys[-1] != ph - tile_h:
            ys.append(ph - tile_h)
        if xs[-1] != pw - tile_w:
            xs.append(pw - tile_w)

        coords = [(y, x) for y in ys for x in xs]
        out_h, out_w = ph * scale, pw * scale
        acc = np.zeros((out_h, out_w, 3), dtype=np.float64)
        weight = np.zeros((out_h, out_w, 1), dtype=np.float64)

        # raised-cosine-ish weights to hide seams
        wy = np.hanning(tile_h * scale) if tile_h * scale > 1 else np.ones(1)
        wx = np.hanning(tile_w * scale) if tile_w * scale > 1 else np.ones(1)
        if float(wy.sum()) == 0:
            wy = np.ones_like(wy)
        if float(wx.sum()) == 0:
            wx = np.ones_like(wx)
        tile_weight = np.outer(wy, wx).astype(np.float64)[..., np.newaxis]
        tile_weight = np.maximum(tile_weight, 1e-3)

        t0 = time.perf_counter()
        for start in range(0, len(coords), batch_size):
            chunk = coords[start : start + batch_size]
            patches = []
            for y, x in chunk:
                patches.append(self._patch_to_nchw(padded[y : y + tile_h, x : x + tile_w]))
            batch = np.stack(patches, axis=0)
            sr_tiles = self._run_batch(batch)
            for (y, x), sr_tile in zip(chunk, sr_tiles):
                oy, ox = y * scale, x * scale
                acc[oy : oy + sr_h, ox : ox + sr_w] += sr_tile.astype(np.float64) * tile_weight
                weight[oy : oy + sr_h, ox : ox + sr_w] += tile_weight

        sr_full = (acc / np.maximum(weight, 1e-6)).clip(0, 255).astype(np.uint8)
        sr_full = sr_full[: orig_h * scale, : orig_w * scale]
        elapsed = time.perf_counter() - t0

        logger.info(
            "Full-image SR: %dx%d → %dx%d (%d tiles, %.1f ms)",
            orig_w,
            orig_h,
            sr_full.shape[1],
            sr_full.shape[0],
            len(coords),
            elapsed * 1000,
        )
        return TelescopeResult(
            original=original,
            lr=lr,
            sr=sr_full,
            elapsed_sec=elapsed,
            tile_count=len(coords),
        )

    def predict_batch(self, images: list[ImageInput]) -> list[TelescopeResult]:
        return [self.predict_full(img) for img in images]


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
