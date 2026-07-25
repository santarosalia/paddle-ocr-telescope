#!/usr/bin/env python3
"""Compare PaddleOCR rec confidence: LR vs Text Telescope SR vs DE-GAN.

Usage:
  python scripts/eval_sr_ocr.py --image receipt.jpg
  python scripts/eval_sr_ocr.py --image-dir ./samples --lang korean
  python scripts/eval_sr_ocr.py --image receipt.jpg --skip-sr      # LR + DE-GAN
  python scripts/eval_sr_ocr.py --image receipt.jpg --skip-degan # LR + Telescope
  python scripts/eval_sr_ocr.py --image receipt.jpg --skip-sr --skip-degan  # LR only
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

from degan_enhance import DeGANTask, degan_ready, get_degan
from telescope_sr import get_telescope

DEFAULT_MODEL = ROOT / "models" / "sr_telescope"
DEFAULT_DEGAN_ROOT = ROOT / "vendor" / "DE-GAN"
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
    telescope: Optional[OcrEval] = None
    degan: Optional[OcrEval] = None
    telescope_size: Optional[tuple[int, int]] = None
    telescope_elapsed_sec: Optional[float] = None
    degan_elapsed_sec: Optional[float] = None
    degan_task: Optional[str] = None
    delta_mean_telescope: Optional[float] = None
    delta_median_telescope: Optional[float] = None
    delta_mean_degan: Optional[float] = None
    delta_median_degan: Optional[float] = None
    best_by_mean: Optional[str] = None
    telescope_image: Optional[np.ndarray] = None
    degan_image: Optional[np.ndarray] = None
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


def evaluate_image(
    image_path: Path,
    ocr: Any,
    telescope: Optional[Any],
    degan: Optional[Any],
    *,
    overlap: int,
    batch_size: int,
    max_side: Optional[int],
    low_threshold: float,
    degan_task: str,
) -> ImageEvalResult:
    pil = Image.open(image_path).convert("RGB")
    lr_size = pil.size

    try:
        lr_lines = run_ocr(pil, ocr)
        lr_eval = evaluate_ocr(lr_lines, low_threshold)

        telescope_eval = None
        degan_eval = None
        telescope_size = None
        telescope_elapsed = None
        degan_elapsed = None
        telescope_image = None
        degan_image = None

        if telescope is not None:
            sr_result = telescope.predict_full(
                pil,
                overlap=overlap,
                batch_size=batch_size,
                max_side=max_side,
            )
            telescope_image = sr_result.sr
            telescope_size = (sr_result.sr.shape[1], sr_result.sr.shape[0])
            telescope_elapsed = sr_result.elapsed_sec
            telescope_eval = evaluate_ocr(run_ocr(Image.fromarray(sr_result.sr), ocr), low_threshold)

        if degan is not None:
            degan_result = degan.predict(pil)
            degan_image = degan_result.enhanced
            degan_elapsed = degan_result.elapsed_sec
            degan_eval = evaluate_ocr(run_ocr(Image.fromarray(degan_result.enhanced), ocr), low_threshold)

        d_mean_t, d_median_t = (None, None)
        d_mean_d, d_median_d = (None, None)
        if telescope_eval is not None:
            d_mean_t, d_median_t = _delta(lr_eval, telescope_eval)
        if degan_eval is not None:
            d_mean_d, d_median_d = _delta(lr_eval, degan_eval)

        variants = {}
        if telescope_eval is not None:
            variants["telescope"] = telescope_eval
        if degan_eval is not None:
            variants["degan"] = degan_eval

        return ImageEvalResult(
            path=str(image_path),
            lr_size=lr_size,
            lr=lr_eval,
            telescope=telescope_eval,
            degan=degan_eval,
            telescope_size=telescope_size,
            telescope_elapsed_sec=telescope_elapsed,
            degan_elapsed_sec=degan_elapsed,
            degan_task=degan_task if degan is not None else None,
            delta_mean_telescope=d_mean_t,
            delta_median_telescope=d_median_t,
            delta_mean_degan=d_mean_d,
            delta_median_degan=d_median_d,
            best_by_mean=_pick_best_by_mean(lr_eval, variants),
            telescope_image=telescope_image,
            degan_image=degan_image,
        )
    except Exception as exc:  # noqa: BLE001
        return ImageEvalResult(path=str(image_path), lr_size=lr_size, error=str(exc))


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
    telescope_scores = _collect_variant_scores(results, "telescope")
    degan_scores = _collect_variant_scores(results, "degan")

    agg: dict[str, Any] = {
        "images": len(results),
        "ok": sum(1 for r in results if not r.error),
        "errors": sum(1 for r in results if r.error),
        "low_threshold": low_threshold,
    }
    if lr_scores:
        agg["lr"] = asdict(summarize_scores(np.array(lr_scores), low_threshold))
    if telescope_scores:
        agg["telescope"] = asdict(summarize_scores(np.array(telescope_scores), low_threshold))
    if degan_scores:
        agg["degan"] = asdict(summarize_scores(np.array(degan_scores), low_threshold))

    lr_arr = np.array(lr_scores) if lr_scores else None
    if lr_arr is not None and lr_arr.size:
        if telescope_scores:
            tel_arr = np.array(telescope_scores)
            agg["delta_mean_telescope"] = float(tel_arr.mean() - lr_arr.mean())
            agg["delta_median_telescope"] = float(np.median(tel_arr) - np.median(lr_arr))
        if degan_scores:
            deg_arr = np.array(degan_scores)
            agg["delta_mean_degan"] = float(deg_arr.mean() - lr_arr.mean())
            agg["delta_median_degan"] = float(np.median(deg_arr) - np.median(lr_arr))

    wins = {"lr": 0, "telescope": 0, "degan": 0}
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
    if item.telescope_size:
        print(
            f"  Telescope size: {item.telescope_size[0]}×{item.telescope_size[1]} "
            f"({item.telescope_elapsed_sec * 1000:.0f} ms)"
        )
    if item.degan_elapsed_sec is not None:
        print(
            f"  DE-GAN ({item.degan_task}): same size as LR "
            f"({item.degan_elapsed_sec * 1000:.0f} ms)"
        )

    if item.lr:
        _print_eval("LR OCR", item.lr, low_threshold)
    if item.telescope:
        _print_eval("Telescope OCR", item.telescope, low_threshold)
    if item.degan:
        _print_eval("DE-GAN OCR", item.degan, low_threshold)

    for label, d_mean, d_median in (
        ("Telescope", item.delta_mean_telescope, item.delta_median_telescope),
        ("DE-GAN", item.delta_mean_degan, item.delta_median_degan),
    ):
        if d_mean is not None:
            sign = "+" if d_mean >= 0 else ""
            print(f"  Δ {label} mean={sign}{d_mean:.4f}, Δ median={sign}{d_median:.4f}")

    if item.best_by_mean:
        print(f"  best_by_mean: {item.best_by_mean}")

    if item.lr:
        _print_lines("LR", item.lr)
    if item.telescope:
        _print_lines("Telescope", item.telescope)
    if item.degan:
        _print_lines("DE-GAN", item.degan)


def _dataclass_to_json(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        data = asdict(obj)
        data.pop("telescope_image", None)
        data.pop("degan_image", None)
        return {k: _dataclass_to_json(v) for k, v in data.items()}
    if isinstance(obj, list):
        return [_dataclass_to_json(v) for v in obj]
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PaddleOCR rec confidence: LR vs Telescope SR vs DE-GAN."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Single image path")
    src.add_argument("--image-dir", type=Path, help="Directory of images")

    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--degan-root", type=Path, default=DEFAULT_DEGAN_ROOT)
    parser.add_argument(
        "--degan-task",
        choices=["deblur", "binarize", "unwatermark"],
        default="deblur",
        help="DE-GAN enhancement mode (default: deblur for receipts)",
    )
    parser.add_argument("--lang", default="korean")
    parser.add_argument("--low-threshold", type=float, default=0.8)
    parser.add_argument("--rec-score-thresh", type=float, default=0.0)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=None)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--skip-sr", action="store_true", help="Skip Text Telescope SR")
    parser.add_argument("--skip-degan", action="store_true", help="Skip DE-GAN enhancement")
    parser.add_argument("--output", type=Path, help="Write JSON report")
    parser.add_argument("--save-telescope-dir", type=Path, help="Save Telescope SR PNGs")
    parser.add_argument("--save-degan-dir", type=Path, help="Save DE-GAN enhanced PNGs")
    # backward compat
    parser.add_argument("--save-sr-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = _collect_images(args.image or args.image_dir)
    save_telescope_dir = args.save_telescope_dir or args.save_sr_dir

    if args.skip_sr:
        telescope = None
    elif not _model_ready(args.model_dir):
        raise SystemExit(
            f"Telescope model not found under {args.model_dir}.\n"
            "Run: python scripts/setup_model.py\n"
            "Or pass --skip-sr."
        )
    else:
        telescope = get_telescope(args.model_dir, use_gpu=args.use_gpu)

    degan_task: DeGANTask = args.degan_task
    if args.skip_degan:
        degan = None
    elif not degan_ready(args.degan_root, degan_task):
        raise SystemExit(
            f"DE-GAN weights not found under {args.degan_root}/weights.\n"
            "Run: python scripts/setup_degan.py\n"
            "Or pass --skip-degan."
        )
    else:
        degan = get_degan(args.degan_root, task=degan_task)

    if telescope is None and degan is None:
        print("Mode: LR OCR baseline only")

    print(f"Loading PaddleOCR (lang={args.lang})…")
    ocr = create_paddle_ocr(args.lang, args.rec_score_thresh)

    results: list[ImageEvalResult] = []
    for image_path in images:
        item = evaluate_image(
            image_path,
            ocr,
            telescope,
            degan,
            overlap=args.overlap,
            batch_size=args.batch_size,
            max_side=args.max_side,
            low_threshold=args.low_threshold,
            degan_task=degan_task,
        )
        results.append(item)
        _print_image_report(item, args.low_threshold)

        if save_telescope_dir and item.telescope_image is not None and not item.error:
            save_telescope_dir.mkdir(parents=True, exist_ok=True)
            out = save_telescope_dir / f"{image_path.stem}_telescope.png"
            Image.fromarray(item.telescope_image).save(out)
            print(f"  saved Telescope -> {out}")

        if args.save_degan_dir and item.degan_image is not None and not item.error:
            args.save_degan_dir.mkdir(parents=True, exist_ok=True)
            out = args.save_degan_dir / f"{image_path.stem}_degan_{degan_task}.png"
            Image.fromarray(item.degan_image).save(out)
            print(f"  saved DE-GAN -> {out}")

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
