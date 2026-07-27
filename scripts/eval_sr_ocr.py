#!/usr/bin/env python3
"""Compare PaddleOCR rec confidence: LR vs classic preprocess.

Usage:
  python scripts/eval_sr_ocr.py --image receipt.jpg
  python scripts/eval_sr_ocr.py --image-dir ./samples --method clahe
  python scripts/eval_sr_ocr.py --image receipt.jpg --skip-pp  # LR only
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

from classic_enhance import CLASSIC_METHODS, ClassicMethod, get_classic

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
    lr: Optional[OcrEval] = None
    pp: Optional[OcrEval] = None
    pp_elapsed_sec: Optional[float] = None
    pp_method: Optional[str] = None
    delta_mean_pp: Optional[float] = None
    delta_median_pp: Optional[float] = None
    best_by_mean: Optional[str] = None
    pp_image: Optional[np.ndarray] = None
    error: Optional[str] = None


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
    payload: Any = ocr_page
    if hasattr(payload, "json"):
        payload = payload.json
    if isinstance(payload, dict) and "res" in payload:
        payload = payload["res"]

    if isinstance(payload, dict):
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        return [
            RecLine(text=str(text), score=float(score))
            for text, score in zip(texts, scores, strict=False)
        ]

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


def _delta(lr: OcrEval, other: OcrEval) -> tuple[Optional[float], Optional[float]]:
    if not lr.summary.count or not other.summary.count:
        return None, None
    return (
        other.summary.mean - lr.summary.mean,
        other.summary.median - lr.summary.median,
    )


def _pick_best_by_mean(lr: OcrEval, variants: dict[str, OcrEval]) -> Optional[str]:
    candidates: dict[str, float] = {"lr": lr.summary.mean}
    for name, ev in variants.items():
        if ev.summary.count:
            candidates[name] = ev.summary.mean
    if not candidates:
        return None
    return max(candidates, key=candidates.get)


def evaluate_pil(
    pil: Image.Image,
    ocr: Any,
    enhancer: Optional[Any] = None,
    *,
    path: str = "<image>",
    low_threshold: float = 0.8,
    pp_method: str = "receipt",
) -> ImageEvalResult:
    """Run LR (+ optional classic preprocess) OCR confidence eval."""
    pil = pil.convert("RGB")
    lr_size = pil.size

    try:
        lr_eval = evaluate_ocr(run_ocr(pil, ocr), low_threshold)

        pp_eval = None
        pp_elapsed = None
        pp_image = None

        if enhancer is not None:
            pp_result = enhancer.predict(pil)
            pp_image = pp_result.enhanced
            pp_elapsed = pp_result.elapsed_sec
            pp_eval = evaluate_ocr(
                run_ocr(Image.fromarray(pp_result.enhanced), ocr), low_threshold
            )

        delta_mean, delta_median = (None, None)
        if pp_eval is not None:
            delta_mean, delta_median = _delta(lr_eval, pp_eval)

        variants: dict[str, OcrEval] = {}
        if pp_eval is not None:
            variants["pp"] = pp_eval

        return ImageEvalResult(
            path=path,
            lr_size=lr_size,
            lr=lr_eval,
            pp=pp_eval,
            pp_elapsed_sec=pp_elapsed,
            pp_method=pp_method if enhancer is not None else None,
            delta_mean_pp=delta_mean,
            delta_median_pp=delta_median,
            best_by_mean=_pick_best_by_mean(lr_eval, variants),
            pp_image=pp_image,
        )
    except Exception as exc:  # noqa: BLE001
        return ImageEvalResult(path=path, lr_size=lr_size, error=str(exc))


def evaluate_image(
    image_path: Path,
    ocr: Any,
    enhancer: Optional[Any] = None,
    *,
    low_threshold: float,
    pp_method: str = "receipt",
) -> ImageEvalResult:
    pil = Image.open(image_path).convert("RGB")
    return evaluate_pil(
        pil,
        ocr,
        enhancer,
        path=str(image_path),
        low_threshold=low_threshold,
        pp_method=pp_method,
    )


def _collect_variant_scores(results: list[ImageEvalResult], attr: str) -> list[float]:
    scores: list[float] = []
    for item in results:
        if item.error:
            continue
        ev = getattr(item, attr, None)
        if ev:
            scores.extend([line.score for line in ev.lines])
    return scores


def _aggregate(results: list[ImageEvalResult], low_threshold: float) -> dict[str, Any]:
    lr_scores = _collect_variant_scores(results, "lr")
    pp_scores = _collect_variant_scores(results, "pp")

    agg: dict[str, Any] = {
        "images": len(results),
        "ok": sum(1 for r in results if not r.error),
        "errors": sum(1 for r in results if r.error),
        "low_threshold": low_threshold,
    }
    if lr_scores:
        agg["lr"] = asdict(summarize_scores(np.array(lr_scores), low_threshold))
    if pp_scores:
        agg["pp"] = asdict(summarize_scores(np.array(pp_scores), low_threshold))

    lr_arr = np.array(lr_scores) if lr_scores else None
    if lr_arr is not None and lr_arr.size and pp_scores:
        pp_arr = np.array(pp_scores)
        agg["delta_mean_pp"] = float(pp_arr.mean() - lr_arr.mean())
        agg["delta_median_pp"] = float(np.median(pp_arr) - np.median(lr_arr))

    wins = {"lr": 0, "pp": 0}
    for item in results:
        if item.best_by_mean in wins:
            wins[item.best_by_mean] += 1
    if any(wins.values()):
        agg["best_by_mean_wins"] = wins

    return agg


def _print_eval(label: str, ev: OcrEval, low_threshold: float) -> None:
    s = ev.summary
    print(
        f"  {label}: {s.count} lines, mean={s.mean:.4f}, median={s.median:.4f}, "
        f"low(<{low_threshold})={s.low_count} ({s.low_ratio:.1%})"
    )


def _print_lines(label: str, ev: OcrEval) -> None:
    if not ev.lines:
        return
    print(f"  {label} lines:")
    for line in ev.lines:
        print(f"    [{line.score:.4f}] {line.text}")


def _print_image_report(item: ImageEvalResult, low_threshold: float) -> None:
    print(f"\n=== {item.path} ===")
    if item.error:
        print(f"  ERROR: {item.error}")
        return

    print(f"  LR size: {item.lr_size[0]}×{item.lr_size[1]}")
    if item.pp_elapsed_sec is not None:
        print(
            f"  classic ({item.pp_method}): "
            f"{item.pp_image.shape[1]}×{item.pp_image.shape[0]} "
            f"({item.pp_elapsed_sec * 1000:.0f} ms)"
            if item.pp_image is not None
            else f"  classic ({item.pp_method}): ({item.pp_elapsed_sec * 1000:.0f} ms)"
        )

    if item.lr:
        _print_eval("LR OCR", item.lr, low_threshold)
    if item.pp:
        _print_eval("classic OCR", item.pp, low_threshold)

    if item.delta_mean_pp is not None:
        sign = "+" if item.delta_mean_pp >= 0 else ""
        print(
            f"  Δ classic mean={sign}{item.delta_mean_pp:.4f}, "
            f"Δ median={sign}{item.delta_median_pp:.4f}"
        )

    if item.best_by_mean:
        print(f"  best_by_mean: {item.best_by_mean}")

    if item.lr:
        _print_lines("LR", item.lr)
    if item.pp:
        _print_lines("classic", item.pp)


def _dataclass_to_json(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        data = asdict(obj)
        data.pop("pp_image", None)
        return {k: _dataclass_to_json(v) for k, v in data.items()}
    if isinstance(obj, list):
        return [_dataclass_to_json(v) for v in obj]
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PaddleOCR rec confidence: LR vs classic preprocess."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Single image path")
    src.add_argument("--image-dir", type=Path, help="Directory of images")

    parser.add_argument(
        "--method",
        choices=list(CLASSIC_METHODS),
        default="receipt",
        help="Classic preprocess method (default: receipt)",
    )
    parser.add_argument("--lang", default="korean")
    parser.add_argument("--low-threshold", type=float, default=0.8)
    parser.add_argument("--rec-score-thresh", type=float, default=0.0)
    parser.add_argument("--skip-pp", action="store_true", help="Skip classic preprocess")
    parser.add_argument("--output", type=Path, help="Write JSON report")
    parser.add_argument("--save-pp-dir", type=Path, help="Save preprocessed PNGs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = _collect_images(args.image or args.image_dir)

    method: ClassicMethod = args.method
    if args.skip_pp:
        enhancer = None
        print("Mode: LR OCR baseline only")
    else:
        enhancer = get_classic(method=method)

    print(f"Loading PaddleOCR (lang={args.lang})…")
    ocr = create_paddle_ocr(args.lang, args.rec_score_thresh)

    results: list[ImageEvalResult] = []
    for image_path in images:
        item = evaluate_image(
            image_path,
            ocr,
            enhancer,
            low_threshold=args.low_threshold,
            pp_method=method,
        )
        results.append(item)
        _print_image_report(item, args.low_threshold)

        if args.save_pp_dir and item.pp_image is not None and not item.error:
            args.save_pp_dir.mkdir(parents=True, exist_ok=True)
            out = args.save_pp_dir / f"{image_path.stem}_pp_{method}.png"
            Image.fromarray(item.pp_image).save(out)
            print(f"  saved classic -> {out}")

    aggregate = _aggregate(results, args.low_threshold)
    print("\n=== Aggregate ===")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))

    if args.output:
        payload = {"aggregate": aggregate, "results": [_dataclass_to_json(r) for r in results]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote report -> {args.output}")


if __name__ == "__main__":
    main()
