"""Streamlit demo: Text Telescope before / after comparison."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from telescope_sr import TelescopeSR, get_telescope

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "sr_telescope"


def _model_ready(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    return any(model_dir.rglob("*.pdiparams"))


def _upscale_for_display(img: np.ndarray, min_h: int = 96) -> np.ndarray:
    """Nearest-neighbor upscale tiny SR crops so Streamlit shows them clearly."""
    h, w = img.shape[:2]
    if h >= min_h:
        return img
    scale = max(1, int(np.ceil(min_h / h)))
    return np.array(
        Image.fromarray(img).resize((w * scale, h * scale), Image.NEAREST)
    )


st.set_page_config(
    page_title="PaddleOCR Text Telescope",
    page_icon="🔭",
    layout="wide",
)

st.title("Text Telescope 전후 비교")
st.caption(
    "Scene Text Telescope (CVPR 2021) — 저해상도 텍스트 크롭을 초해상도(SR)로 복원합니다. "
    "단어/행 단위 크롭 이미지에 최적화되어 있습니다."
)

with st.sidebar:
    st.header("설정")
    model_dir = Path(
        st.text_input("모델 경로", value=str(DEFAULT_MODEL))
    )
    use_gpu = st.checkbox("GPU 사용", value=False)
    show_lr = st.checkbox("모델 입력(LR)도 표시", value=True)
    enlarge = st.checkbox("작은 결과 확대 표시", value=True)
    st.divider()
    st.markdown(
        "**모델 준비**\n\n"
        "```bash\npip install -r requirements.txt\n"
        "python scripts/setup_model.py\n```"
    )

if not _model_ready(model_dir):
    st.warning(
        f"추론 모델이 없습니다: `{model_dir}`\n\n"
        "터미널에서 `python scripts/setup_model.py` 를 먼저 실행하세요."
    )
    st.stop()

uploaded = st.file_uploader(
    "이미지 업로드 (텍스트 크롭 / 단어 영역 권장)",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded is None:
    st.info("이미지를 업로드하면 Telescope 적용 전·후를 나란히 보여줍니다.")
    st.stop()

pil = Image.open(uploaded).convert("RGB")

col_meta1, col_meta2 = st.columns(2)
with col_meta1:
    st.write(f"원본 크기: **{pil.size[0]} × {pil.size[1]}**")
with col_meta2:
    st.write(f"파일: `{uploaded.name}`")

run = st.button("Telescope 실행", type="primary", use_container_width=True)

if not run:
    st.subheader("업로드 원본")
    st.image(pil, use_container_width=True)
    st.stop()

with st.spinner("Text Telescope 추론 중…"):
    try:
        engine: TelescopeSR = get_telescope(model_dir, use_gpu=use_gpu)
        result = engine.predict(pil)
    except Exception as exc:  # noqa: BLE001 — show in UI
        st.error(f"추론 실패: {exc}")
        st.exception(exc)
        st.stop()

st.success(f"완료 ({result.elapsed_sec * 1000:.1f} ms)")

disp_orig = _upscale_for_display(result.original) if enlarge else result.original
disp_lr = _upscale_for_display(result.lr) if enlarge else result.lr
disp_sr = _upscale_for_display(result.sr) if enlarge else result.sr

if show_lr:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("원본")
        st.image(disp_orig, use_container_width=True)
        st.caption(f"{result.original.shape[1]}×{result.original.shape[0]}")
    with c2:
        st.subheader("Before (LR)")
        st.image(disp_lr, use_container_width=True)
        st.caption(
            f"{result.lr.shape[1]}×{result.lr.shape[0]} — 모델 입력 (bicubic ↓2)"
        )
    with c3:
        st.subheader("After (SR)")
        st.image(disp_sr, use_container_width=True)
        st.caption(f"{result.sr.shape[1]}×{result.sr.shape[0]} — Telescope 출력")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Before (원본)")
        st.image(disp_orig, use_container_width=True)
        st.caption(f"{result.original.shape[1]}×{result.original.shape[0]}")
    with c2:
        st.subheader("After (Telescope SR)")
        st.image(disp_sr, use_container_width=True)
        st.caption(f"{result.sr.shape[1]}×{result.sr.shape[0]}")

st.divider()
st.subheader("Before / After 나란히")
# side-by-side strip for quick visual diff (LR upscaled to SR size)
lr_matched = np.array(
    Image.fromarray(result.lr).resize(
        (result.sr.shape[1], result.sr.shape[0]), Image.NEAREST
    )
)
strip = np.concatenate([lr_matched, result.sr], axis=1)
if enlarge:
    strip = _upscale_for_display(strip, min_h=128)
st.image(
    strip,
    caption="Left: LR (before)  |  Right: SR (after)",
    use_container_width=True,
)

sr_pil = Image.fromarray(result.sr)
buf = BytesIO()
sr_pil.save(buf, format="PNG")
st.download_button(
    "SR 결과 PNG 다운로드",
    data=buf.getvalue(),
    file_name=f"telescope_sr_{Path(uploaded.name).stem}.png",
    mime="image/png",
)
