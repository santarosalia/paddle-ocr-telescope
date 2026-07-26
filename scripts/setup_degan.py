#!/usr/bin/env python3
"""Clone DE-GAN and download official enhancement weights."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEGAN_ROOT = ROOT / "vendor" / "DE-GAN"
WEIGHTS_DIR = DEGAN_ROOT / "weights"
DEGAN_REPO = "https://github.com/dali92002/DE-GAN.git"
GDRIVE_ID = "1J_t-TzR2rxp94SzfPoeuJniSFLfY3HM-"
REQUIRED_WEIGHTS = (
    "deblur_weights.h5",
    "binarization_generator_weights.h5",
    "watermark_rem_weights.h5",
)


def ensure_repo() -> Path:
    marker = DEGAN_ROOT / "enhance.py"
    if marker.exists():
        print(f"[skip] DE-GAN already present: {DEGAN_ROOT}")
        return DEGAN_ROOT

    DEGAN_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if DEGAN_ROOT.exists():
        shutil.rmtree(DEGAN_ROOT)

    print(f"[clone] {DEGAN_REPO}")
    subprocess.check_call(["git", "clone", "--depth", "1", DEGAN_REPO, str(DEGAN_ROOT)])
    return DEGAN_ROOT


def ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("[deps] installing gdown…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])


def download_weights(force: bool = False) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if not force and all((WEIGHTS_DIR / name).is_file() for name in REQUIRED_WEIGHTS):
        print(f"[skip] DE-GAN weights already present: {WEIGHTS_DIR}")
        return

    ensure_gdown()
    import gdown

    archive = WEIGHTS_DIR / "degan_weights.zip"
    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
    print(f"[download] {url}")
    gdown.download(url, str(archive), quiet=False)

    print(f"[extract] {archive}")
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            name = Path(member).name
            target = WEIGHTS_DIR / name
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())

    missing = [name for name in REQUIRED_WEIGHTS if not (WEIGHTS_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing weights after extract: {missing}")

    archive.unlink(missing_ok=True)
    print(f"[ok] DE-GAN weights -> {WEIGHTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup DE-GAN vendor repo and weights")
    parser.add_argument("--skip-clone", action="store_true", help="Assume vendor/DE-GAN exists")
    parser.add_argument("--force-download", action="store_true", help="Re-download weights")
    args = parser.parse_args()

    if args.skip_clone:
        if not (DEGAN_ROOT / "enhance.py").exists():
            raise SystemExit("vendor/DE-GAN missing; omit --skip-clone")
    else:
        ensure_repo()

    download_weights(force=args.force_download)
    print("\nDone. Example:")
    print("  python scripts/eval_sr_ocr.py --image receipt.jpg --lang korean")


if __name__ == "__main__":
    main()
