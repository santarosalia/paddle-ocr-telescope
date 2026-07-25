"""Streamlit demo: Text Telescope full-image (receipt) SR."""

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


st.set_page_config(
    page_title="PaddleOCR Text Telescope",
    page_icon="🔭",
    layout="wide",
)

st.title("Text Telescope — 영수증 전체 SR")
st.caption(
    "업로드 이미지를 저해상도(LR)로 보고, 타일 단위로 Text Telescope(×2)를 적용한 뒤 "
    "전체를 이어 붙입니다."
)

with st.sidebar:
    st.header("설정")
    model_dir = Path(st.text_input("모델 경로", value=str(DEFAULT_MODEL)))
    use_gpu = st.checkbox("GPU 사용", value=False)
    overlap = st.slider("타일 오버랩 (px)", min_value=0, max_value=12, value=8)
    batch_size = st.slider("배치 크기", min_value=1, max_value=16, value=8)
    limit_side = st.checkbox("긴 변 제한 (속도)", value=True)
    max_side = st.number_input(
        "긴 변 최대 px",
        min_value=256,
        max_value=4096,
        value=1280,
        step=64,
        disabled=not limit_side,
    )
    st.divider()
    st.markdown(
        "**모델 준비**\n\n"
        "```bash\npip install -r requirements.txt\n"
        "python scripts/setup_model.py\n```"
    )
    st.caption("타일 LR 16×64 → SR 32×128. 영수증처럼 텍스트가 많은 이미지에 적합합니다.")

if not _model_ready(model_dir):
    st.warning(
        f"추론 모델이 없습니다: `{model_dir}`\n\n"
        "터미널에서 `python scripts/setup_model.py` 를 먼저 실행하세요."
    )
    st.stop()

uploaded = st.file_uploader(
    "영수증 / 저해상도 이미지 업로드",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded is None:
    st.info("이미지를 업로드하면 전체 ×2 Telescope SR 전후를 비교합니다.")
    st.stop()

pil = Image.open(uploaded).convert("RGB")

col_meta1, col_meta2 = st.columns(2)
with col_meta1:
    st.write(f"원본(LR) 크기: **{pil.size[0]} × {pil.size[1]}**")
with col_meta2:
    st.write(f"파일: `{uploaded.name}`")

run = st.button("전체 이미지 Telescope 실행", type="primary", use_container_width=True)

if not run:
    st.subheader("업로드 원본 (LR)")
    st.image(pil, use_container_width=True)
    st.stop()

with st.spinner("전체 이미지 타일 SR 추론 중… (크기에 따라 시간이 걸릴 수 있습니다)"):
    try:
        engine: TelescopeSR = get_telescope(model_dir, use_gpu=use_gpu)
        result = engine.predict_full(
            pil,
            overlap=int(overlap),
            batch_size=int(batch_size),
            max_side=int(max_side) if limit_side else None,
        )
    except Exception as exc:  # noqa: BLE001 — show in UI
        st.error(f"추론 실패: {exc}")
        st.exception(exc)
        st.stop()

st.success(
    f"완료 — {result.tile_count}타일, {result.elapsed_sec * 1000:.0f} ms · "
    f"{result.lr.shape[1]}×{result.lr.shape[0]} → "
    f"{result.sr.shape[1]}×{result.sr.shape[0]}"
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Before (LR)")
    st.image(result.lr, use_container_width=True)
    st.caption(f"{result.lr.shape[1]}×{result.lr.shape[0]}")
with c2:
    st.subheader("After (SR ×2)")
    st.image(result.sr, use_container_width=True)
    st.caption(f"{result.sr.shape[1]}×{result.sr.shape[0]}")

st.divider()
st.subheader("Before / After 나란히")
lr_matched = np.array(
    Image.fromarray(result.lr).resize(
        (result.sr.shape[1], result.sr.shape[0]), Image.NEAREST
    )
)
# 너무 큰 비교 스트립은 표시용으로만 축소
strip = np.concatenate([lr_matched, result.sr], axis=1)
max_strip_w = 2400
if strip.shape[1] > max_strip_w:
    scale = max_strip_w / strip.shape[1]
    strip = np.array(
        Image.fromarray(strip).resize(
            (max_strip_w, max(1, int(strip.shape[0] * scale))),
            Image.BILINEAR,
        )
    )
st.image(
    strip,
    caption="Left: LR (nearest ×2)  |  Right: Telescope SR",
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
