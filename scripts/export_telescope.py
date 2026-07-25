#!/usr/bin/env python3
"""Export Text Telescope train checkpoint to Paddle Inference format.

Avoids PaddleOCR's heavy (and NumPy2-incompatible) training deps like imgaug
by stubbing them before import and loading YAML config directly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
from pathlib import Path


def _stub_heavy_deps() -> None:
    """Prevent import of packages broken on NumPy 2 / unused for export."""
    if "imgaug" not in sys.modules:
        stub = types.ModuleType("imgaug")
        stub.augmenters = types.ModuleType("imgaug.augmenters")
        sys.modules["imgaug"] = stub
        sys.modules["imgaug.augmenters"] = stub.augmenters


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid -o item (want key=value): {item}")
        key, value = item.split("=", 1)
        # YAML-parse value so True/False/None/numbers work
        import yaml

        parsed = yaml.safe_load(value)
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paddleocr-dir",
        type=Path,
        required=True,
        help="Path to cloned PaddleOCR repo",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="sr_telescope.yml (default: <paddleocr>/configs/sr/sr_telescope.yml)",
    )
    parser.add_argument(
        "-o",
        dest="overrides",
        action="append",
        default=[],
        help="Config override, e.g. Global.pretrained_model=/path/best_accuracy",
    )
    args = parser.parse_args()

    paddleocr_dir = args.paddleocr_dir.resolve()
    cfg_path = (
        args.config
        if args.config
        else paddleocr_dir / "configs" / "sr" / "sr_telescope.yml"
    )
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    sys.path.insert(0, str(paddleocr_dir))
    _stub_heavy_deps()

    import paddle
    from paddle.jit import to_static

    from ppocr.modeling.architectures import build_model
    from ppocr.utils.save_load import load_model

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("export_telescope")

    config = _load_yaml(cfg_path)
    _apply_overrides(config, args.overrides)

    if config["Architecture"]["model_type"] != "sr":
        raise SystemExit("This exporter only supports Architecture.model_type=sr")

    config["Architecture"]["Transform"]["infer_mode"] = True
    model = build_model(config["Architecture"])
    load_model(config, model, model_type=config["Architecture"]["model_type"])
    model.eval()

    class _SrExportWrapper(paddle.nn.Layer):
        """Export SR only — lr_img aliases the input and breaks PIR name analysis."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, dict):
                return out["sr_img"]
            return out

    wrapped = _SrExportWrapper(model)
    wrapped.eval()

    save_dir = Path(config["Global"]["save_inference_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / "inference")

    # Match tools/export_model.py for model_type == "sr"
    input_spec = [
        paddle.static.InputSpec(shape=[None, 3, 16, 64], dtype="float32", name="image")
    ]
    static_model = to_static(wrapped, input_spec=input_spec)
    paddle.jit.save(static_model, save_path)
    logger.info("inference model saved to %s", save_path)


if __name__ == "__main__":
    main()
