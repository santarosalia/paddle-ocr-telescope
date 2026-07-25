#!/usr/bin/env python3
"""Download Text Telescope weights and export a Paddle Inference model.

Steps:
  1. Shallow-clone PaddleOCR (release/2.7 — SR still first-class there)
  2. Download official train checkpoint
  3. Export inference model to ./models/sr_telescope
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "PaddleOCR"
MODELS = ROOT / "models" / "sr_telescope"
WEIGHTS_DIR = ROOT / "models" / "weights"
WEIGHTS_URL = "https://paddleocr.bj.bcebos.com/contribution/sr_telescope_train.tar"
PADDLEOCR_REPO = "https://github.com/PaddlePaddle/PaddleOCR.git"
PADDLEOCR_REF = "release/2.7"


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] already downloaded: {dest}")
        return dest

    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return dest


def ensure_paddleocr() -> Path:
    if (VENDOR / "tools" / "export_model.py").exists():
        print(f"[skip] PaddleOCR already present: {VENDOR}")
        return VENDOR

    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    if VENDOR.exists():
        shutil.rmtree(VENDOR)

    print(f"[clone] {PADDLEOCR_REPO}@{PADDLEOCR_REF}")
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            PADDLEOCR_REF,
            PADDLEOCR_REPO,
            str(VENDOR),
        ]
    )
    return VENDOR


def extract_weights(archive: Path) -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    # look for existing best_accuracy*
    existing = list(WEIGHTS_DIR.rglob("best_accuracy.pdparams"))
    if existing:
        ckpt = existing[0].with_suffix("")  # drop .pdparams
        # load_model expects path without extension
        print(f"[skip] weights found: {ckpt}.pdparams")
        return ckpt

    print(f"[extract] {archive}")
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(WEIGHTS_DIR)

    found = list(WEIGHTS_DIR.rglob("best_accuracy.pdparams"))
    if not found:
        # some archives use different names
        found = list(WEIGHTS_DIR.rglob("*.pdparams"))
    if not found:
        raise FileNotFoundError("No .pdparams found after extracting weights")

    params = found[0]
    return params.with_suffix("")


def ensure_export_deps() -> None:
    """Minimal deps for our slim exporter (no imgaug / visualdl)."""
    pkgs = [
        "shapely>=2.0.0",
        "pyyaml>=6.0",
        "pyclipper>=1.3.0",
        "scikit-image>=0.22.0",
        "lmdb>=1.4.0",
        "cython>=3.0.0",
    ]
    print("[deps] ensuring export dependencies…")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *pkgs]
    )


def export_inference(
    paddleocr_dir: Path,
    pretrained: Path,
    out_dir: Path,
    force: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "inference.pdiparams"
    alt = out_dir / "model.pdiparams"
    if not force and (marker.exists() or alt.exists()):
        print(f"[skip] inference model already exists: {out_dir} (use --force to re-export)")
        return

    cfg = paddleocr_dir / "configs" / "sr" / "sr_telescope.yml"
    if not cfg.exists():
        raise FileNotFoundError(f"Missing config: {cfg}")

    ensure_export_deps()

    for p in out_dir.glob("inference.*"):
        p.unlink()
    for p in out_dir.glob("model.*"):
        p.unlink()

    export_script = ROOT / "scripts" / "export_telescope.py"
    cmd = [
        sys.executable,
        str(export_script),
        "--paddleocr-dir",
        str(paddleocr_dir),
        "--config",
        str(cfg),
        "-o",
        f"Global.pretrained_model={pretrained}",
        "-o",
        f"Global.save_inference_dir={out_dir}",
        "-o",
        "Global.use_gpu=False",
    ]
    print("[export]", " ".join(cmd))
    subprocess.check_call(cmd)

    if not marker.exists() and not alt.exists():
        nested = list(out_dir.rglob("*.pdiparams"))
        if not nested:
            raise RuntimeError(f"Export finished but no pdiparams under {out_dir}")
        print(f"[ok] exported: {nested[0].parent}")
    else:
        print(f"[ok] exported inference model -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup Text Telescope inference model")
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Assume vendor/PaddleOCR already exists",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export inference model even if it already exists",
    )
    parser.add_argument(
        "--weights-url",
        default=WEIGHTS_URL,
        help="Train checkpoint tarball URL",
    )
    args = parser.parse_args()

    if args.skip_clone:
        if not (VENDOR / "tools" / "export_model.py").exists():
            raise SystemExit("vendor/PaddleOCR missing; omit --skip-clone")
        paddleocr_dir = VENDOR
    else:
        paddleocr_dir = ensure_paddleocr()

    archive = download(args.weights_url, WEIGHTS_DIR / "sr_telescope_train.tar")
    pretrained = extract_weights(archive)
    export_inference(paddleocr_dir, pretrained, MODELS, force=args.force)
    print("\nDone. Run Streamlit with:\n  streamlit run app.py")


if __name__ == "__main__":
    main()
