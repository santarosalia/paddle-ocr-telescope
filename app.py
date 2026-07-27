"""Streamlit demo: classic image preprocess only."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from classic_enhance import CLASSIC_METHODS, ClassicMethod, get_classic

st.set_page_config(
    page_title="Classic Preprocess",
    page_icon="🧾",
    layout="wide",
)

st.title("Classic Preprocess")
st.caption("원본 → ×3 upscale → CLAHE → Unsharp → mild denoise")

with st.sidebar:
    st.header("전처리")
    pp_method: ClassicMethod = st.selectbox(  # type: ignore[assignment]
        "method",
        options=list(CLASSIC_METHODS),
        index=list(CLASSIC_METHODS).index("receipt"),
        help="receipt=×3→CLAHE→Unsharp→mild denoise (기본)",
    )
    st.divider()
    st.markdown("```bash\npip install -r requirements.txt\nstreamlit run app.py\n```")

uploaded = st.file_uploader(
    "이미지 업로드",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded is None:
    st.info("이미지를 업로드한 뒤 전처리를 실행하세요.")
    st.stop()

pil = Image.open(uploaded).convert("RGB")
st.write(f"원본 크기: **{pil.size[0]} × {pil.size[1]}** · `{uploaded.name}`")

run = st.button("전처리 실행", type="primary", use_container_width=True)
if not run:
    st.subheader("원본")
    st.image(pil, use_container_width=True)
    st.stop()

with st.spinner("전처리 중…"):
    try:
        result = get_classic(method=pp_method).predict(pil)
    except Exception as exc:  # noqa: BLE001
        st.error(f"전처리 실패: {exc}")
        st.exception(exc)
        st.stop()

st.success(
    f"완료 — {result.elapsed_sec * 1000:.0f} ms · "
    f"{result.original.shape[1]}×{result.original.shape[0]} → "
    f"{result.enhanced.shape[1]}×{result.enhanced.shape[0]}"
    + (f" (×{result.scale:g})" if result.scale != 1.0 else "")
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Before")
    st.image(result.original, use_container_width=True)
    st.caption(f"{result.original.shape[1]}×{result.original.shape[0]}")
with c2:
    st.subheader(f"After ({result.method})")
    st.image(result.enhanced, use_container_width=True)
    st.caption(f"{result.enhanced.shape[1]}×{result.enhanced.shape[0]}")

buf = BytesIO()
Image.fromarray(result.enhanced).save(buf, format="PNG")
st.download_button(
    "전처리 결과 PNG 다운로드",
    data=buf.getvalue(),
    file_name=f"pp_{pp_method}_{Path(uploaded.name).stem}.png",
    mime="image/png",
    use_container_width=True,
)
