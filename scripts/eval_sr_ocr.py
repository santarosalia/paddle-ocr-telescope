#!/usr/bin/env python3
"""Compare PaddleOCR recognition confidence on LR vs Text Telescope SR images.

Usage:
  python scripts/eval_sr_ocr.py --image receipt.jpg
  python scripts/eval_sr_ocr.py --image-dir ./samples --lang korean
  python scripts/eval_sr_ocr.py --image receipt.jpg --skip-sr   # OCR baseline only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = ROOT / "models" / "sr_telescope"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class RecLine:
    text: str
    score: float


@dataclass
class ScoreSummary:
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    min: float = 0.0
    max: float = 0.0
    low_count: int = 0
    low_ratio: float = 0.0


@dataclass
class OcrEval:
    lines: list[RecLine] = field(default_factory=list)
    summary: ScoreSummary = field(default_factory=ScoreSummary)


@dataclass
class ImageEvalResult:
    path: str
    lr_size: tuple[int, int]
    sr_size: Optional[tuple[int, int]] = None
    lr: Optional[OcrEval] = None
    sr: Optional[OcrEval] = None
    delta_mean: Optional[float] = None
    delta_median: Optional[float] = None
    sr_elapsed_sec: Optional[float] = None
    sr_image: Optional[np.ndarray] = None
    error: Optional[str] = None


def _model_ready(model_dir: Path) -> bool:
    return model_dir.is_dir() and any(model_dir.rglob("*.pdiparams"))


def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(
            p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
        )
        if not files:
            raise SystemExit(f"No images found under {path}")
        return files
    raise SystemExit(f"Not found: {path}")


def _to_scores(lines: list[RecLine]) -> np.ndarray:
    if not lines:
        return np.array([], dtype=float)
    return np.array([line.score for line in lines], dtype=float)


def summarize_scores(scores: np.ndarray, low_threshold: float) -> ScoreSummary:
    if scores.size == 0:
        return ScoreSummary()
    low = int(np.sum(scores < low_threshold))
    return ScoreSummary(
        count=int(scores.size),
        mean=float(scores.mean()),
        median=float(np.median(scores)),
        min=float(scores.min()),
        max=float(scores.max()),
        low_count=low,
        low_ratio=float(low / scores.size),
    )


def extract_rec_lines(ocr_page: Any) -> list[RecLine]:
    """Normalize PaddleOCR 2.x / 3.x outputs to rec lines."""
    payload: Any = ocr_page
    if hasattr(payload, "json"):
        payload = payload.json
    if isinstance(payload, dict) and "res" in payload:
        payload = payload["res"]

    # PaddleOCR 3.x dict / OCRResult
    if isinstance(payload, dict):
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        return [
            RecLine(text=str(text), score=float(score))
            for text, score in zip(texts, scores, strict=False)
        ]

    # PaddleOCR 2.x: [[box, (text, score)], ...]
    if isinstance(payload, list):
        lines: list[RecLine] = []
        for item in payload:
            if not item or len(item) < 2:
                continue
            rec = item[1]
            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                lines.append(RecLine(text=str(rec[0]), score=float(rec[1])))
        return lines

    return []


def create_paddle_ocr(lang: str, rec_score_thresh: float):
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")

    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=rec_score_thresh,
        enable_mkldnn=False,
    )


def run_ocr(image: Image.Image, ocr: Any) -> list[RecLine]:
    arr = np.array(image.convert("RGB"))
    pages = ocr.predict(arr)
    if not pages:
        return []
    return extract_rec_lines(pages[0])


def evaluate_ocr(lines: list[RecLine], low_threshold: float) -> OcrEval:
    scores = _to_scores(lines)
    return OcrEval(lines=lines, summary=summarize_scores(scores, low_threshold))


def evaluate_pil(
    pil: Image.Image,
    ocr: Any,
    telescope: Optional[Any],
    *,
    path: str = "<image>",
    overlap: int,
    batch_size: int,
    max_side: Optional[int],
    low_threshold: float,
    skip_sr: bool,
) -> ImageEvalResult:
    """Run LR (and optional SR) OCR confidence eval on an in-memory RGB image."""
    pil = pil.convert("RGB")
    lr_size = pil.size

    try:
        lr_lines = run_ocr(pil, ocr)
        lr_eval = evaluate_ocr(lr_lines, low_threshold)

        if skip_sr or telescope is None:
            return ImageEvalResult(path=path, lr_size=lr_size, lr=lr_eval)

        sr_result = telescope.predict_full(
            pil,
            overlap=overlap,
            batch_size=batch_size,
            max_side=max_side,
        )
        sr_pil = Image.fromarray(sr_result.sr)
        sr_lines = run_ocr(sr_pil, ocr)
        sr_eval = evaluate_ocr(sr_lines, low_threshold)

        delta_mean = None
        delta_median = None
        if lr_eval.summary.count and sr_eval.summary.count:
            delta_mean = sr_eval.summary.mean - lr_eval.summary.mean
            delta_median = sr_eval.summary.median - lr_eval.summary.median

        return ImageEvalResult(
            path=path,
            lr_size=lr_size,
            sr_size=(sr_result.sr.shape[1], sr_result.sr.shape[0]),
            lr=lr_eval,
            sr=sr_eval,
            delta_mean=delta_mean,
            delta_median=delta_median,
            sr_elapsed_sec=sr_result.elapsed_sec,
            sr_image=sr_result.sr,
        )
    except Exception as exc:  # noqa: BLE001 — collect per-image errors in batch runs
        return ImageEvalResult(path=path, lr_size=lr_size, error=str(exc))


def evaluate_image(
    image_path: Path,
    ocr: Any,
    telescope: Optional[Any],
    *,
    overlap: int,
    batch_size: int,
    max_side: Optional[int],
    low_threshold: float,
    skip_sr: bool,
) -> ImageEvalResult:
    pil = Image.open(image_path).convert("RGB")
    return evaluate_pil(
        pil,
        ocr,
        telescope,
        path=str(image_path),
        overlap=overlap,
        batch_size=batch_size,
        max_side=max_side,
        low_threshold=low_threshold,
        skip_sr=skip_sr,
    )

def _aggregate(results: list[ImageEvalResult], low_threshold: float) -> dict[str, Any]:
    lr_scores = []
    sr_scores = []
    for item in results:
        if item.error or not item.lr:
            continue
        lr_scores.extend([line.score for line in item.lr.lines])
        if item.sr:
            sr_scores.extend([line.score for line in item.sr.lines])

    agg: dict[str, Any] = {
        "images": len(results),
        "ok": sum(1 for r in results if not r.error),
        "errors": sum(1 for r in results if r.error),
        "low_threshold": low_threshold,
    }
    if lr_scores:
        agg["lr"] = asdict(summarize_scores(np.array(lr_scores), low_threshold))
    if sr_scores:
        agg["sr"] = asdict(summarize_scores(np.array(sr_scores), low_threshold))
    if lr_scores and sr_scores:
        lr_arr = np.array(lr_scores)
        sr_arr = np.array(sr_scores)
        agg["delta_mean"] = float(sr_arr.mean() - lr_arr.mean())
        agg["delta_median"] = float(np.median(sr_arr) - np.median(lr_arr))
    return agg


def _print_image_report(item: ImageEvalResult, low_threshold: float) -> None:
    print(f"\n=== {item.path} ===")
    if item.error:
        print(f"  ERROR: {item.error}")
        return

    print(f"  LR size: {item.lr_size[0]}×{item.lr_size[1]}")
    if item.sr_size:
        print(
            f"  SR size: {item.sr_size[0]}×{item.sr_size[1]} "
            f"({item.sr_elapsed_sec * 1000:.0f} ms SR)"
        )

    if item.lr:
        s = item.lr.summary
        print(
            f"  LR OCR: {s.count} lines, mean={s.mean:.4f}, median={s.median:.4f}, "
            f"low(<{low_threshold})={s.low_count} ({s.low_ratio:.1%})"
        )
    if item.sr:
        s = item.sr.summary
        print(
            f"  SR OCR: {s.count} lines, mean={s.mean:.4f}, median={s.median:.4f}, "
            f"low(<{low_threshold})={s.low_count} ({s.low_ratio:.1%})"
        )
    if item.delta_mean is not None:
        sign = "+" if item.delta_mean >= 0 else ""
        print(f"  Δ mean={sign}{item.delta_mean:.4f}, Δ median={sign}{item.delta_median:.4f}")

    if item.lr and item.lr.lines:
        print("  LR lines:")
        for line in item.lr.lines:
            print(f"    [{line.score:.4f}] {line.text}")
    if item.sr and item.sr.lines:
        print("  SR lines:")
        for line in item.sr.lines:
            print(f"    [{line.score:.4f}] {line.text}")


def _dataclass_to_json(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        data = asdict(obj)
        data.pop("sr_image", None)
        return {k: _dataclass_to_json(v) for k, v in data.items()}
    if isinstance(obj, list):
        return [_dataclass_to_json(v) for v in obj]
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PaddleOCR rec confidence on LR vs Text Telescope SR."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Single image path")
    src.add_argument("--image-dir", type=Path, help="Directory of images")

    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--lang", default="korean", help="PaddleOCR language (default: korean)")
    parser.add_argument("--low-threshold", type=float, default=0.8, help="Low-confidence cutoff")
    parser.add_argument(
        "--rec-score-thresh",
        type=float,
        default=0.0,
        help="PaddleOCR text_rec_score_thresh (0 = keep all lines for scoring)",
    )
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=None, help="Resize long side before SR")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--skip-sr", action="store_true", help="Run OCR on LR only")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument(
        "--save-sr-dir",
        type=Path,
        help="Save SR images as <stem>_sr.png under this directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = _collect_images(args.image or args.image_dir)

    if args.skip_sr:
        telescope = None
    elif not _model_ready(args.model_dir):
        raise SystemExit(
            f"Telescope model not found under {args.model_dir}. "
            "Run: python scripts/setup_model.py\n"
            "Or pass --skip-sr to measure LR OCR only."
        )
    else:
        from telescope_sr import get_telescope

        telescope = get_telescope(args.model_dir, use_gpu=args.use_gpu)

    print(f"Loading PaddleOCR (lang={args.lang})…")
    ocr = create_paddle_ocr(args.lang, args.rec_score_thresh)

    results: list[ImageEvalResult] = []
    for image_path in images:
        item = evaluate_image(
            image_path,
            ocr,
            telescope,
            overlap=args.overlap,
            batch_size=args.batch_size,
            max_side=args.max_side,
            low_threshold=args.low_threshold,
            skip_sr=args.skip_sr,
        )
        results.append(item)
        _print_image_report(item, args.low_threshold)

        if args.save_sr_dir and item.sr_image is not None and not item.error:
            args.save_sr_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.save_sr_dir / f"{image_path.stem}_sr.png"
            Image.fromarray(item.sr_image).save(out_path)
            print(f"  saved SR -> {out_path}")

    aggregate = _aggregate(results, args.low_threshold)
    print("\n=== Aggregate ===")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))

    if args.output:
        payload = {
            "aggregate": aggregate,
            "results": [_dataclass_to_json(r) for r in results],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote report -> {args.output}")


if __name__ == "__main__":
    main()
