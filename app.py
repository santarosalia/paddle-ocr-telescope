"""Streamlit demo: Text Telescope SR + OCR eval + det→SR→rec."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from telescope_sr import TelescopeSR, get_telescope
from text_sr_pipeline import (
    create_text_detector,
    create_text_recognizer,
    run_det_sr_rec,
)

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_sr_ocr import (  # noqa: E402
    create_paddle_ocr,
    evaluate_pil,
)

DEFAULT_MODEL = ROOT / "models" / "sr_telescope"


def _model_ready(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    return any(model_dir.rglob("*.pdiparams"))


@st.cache_resource(show_spinner="PaddleOCR 로딩 중…")
def _cached_ocr(lang: str, rec_score_thresh: float):
    return create_paddle_ocr(lang, rec_score_thresh)


@st.cache_resource(show_spinner="TextDetection 로딩 중…")
def _cached_detector(unclip_ratio: float, box_thresh: float, det_model: str):
    return create_text_detector(
        unclip_ratio=unclip_ratio,
        box_thresh=box_thresh if box_thresh > 0 else None,
        model_name=det_model or None,
    )


@st.cache_resource(show_spinner="TextRecognition 로딩 중…")
def _cached_recognizer(lang: str, rec_model: str):
    return create_text_recognizer(lang=lang, model_name=rec_model or None)


def _summary_metrics(summary, low_threshold: float) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("라인 수", summary.count)
    c2.metric("mean", f"{summary.mean:.4f}")
    c3.metric("median", f"{summary.median:.4f}")
    c4.metric(
        f"low(<{low_threshold})",
        f"{summary.low_count} ({summary.low_ratio:.0%})",
    )


def _lines_table(lines) -> None:
    if not lines:
        st.caption("인식된 라인이 없습니다.")
        return
    st.dataframe(
        [
            {"score": round(line.score, 4), "text": line.text}
            for line in sorted(lines, key=lambda x: x.score)
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_sr_tab(
    pil: Image.Image,
    name: str,
    model_dir: Path,
    use_gpu: bool,
    overlap: int,
    batch_size: int,
    max_side,
) -> None:
    run = st.button(
        "전체 이미지 Telescope 실행",
        type="primary",
        use_container_width=True,
        key="run_full_sr",
    )
    if not run:
        st.subheader("업로드 원본 (LR)")
        st.image(pil, use_container_width=True)
        return

    with st.spinner("전체 이미지 타일 SR 추론 중…"):
        try:
            engine: TelescopeSR = get_telescope(model_dir, use_gpu=use_gpu)
            result = engine.predict_full(
                pil,
                overlap=overlap,
                batch_size=batch_size,
                max_side=max_side,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"추론 실패: {exc}")
            st.exception(exc)
            return

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

    buf = BytesIO()
    Image.fromarray(result.sr).save(buf, format="PNG")
    st.download_button(
        "SR 결과 PNG 다운로드",
        data=buf.getvalue(),
        file_name=f"telescope_sr_{Path(name).stem}.png",
        mime="image/png",
        key="dl_full_sr",
    )


def _render_eval_tab(
    pil: Image.Image,
    name: str,
    model_dir: Path,
    use_gpu: bool,
    overlap: int,
    batch_size: int,
    max_side,
    lang: str,
    low_threshold: float,
    rec_score_thresh: float,
    skip_sr: bool,
) -> None:
    run = st.button(
        "LR vs SR OCR 평가 실행",
        type="primary",
        use_container_width=True,
        key="run_eval",
    )
    if not run:
        st.subheader("업로드 원본 (LR)")
        st.image(pil, use_container_width=True)
        st.caption("전체 이미지에 OCR을 돌려 LR/SR 신뢰도를 비교합니다.")
        return

    if not skip_sr and not _model_ready(model_dir):
        st.error(f"Telescope 모델이 없습니다: `{model_dir}`")
        return

    with st.spinner("OCR 평가 중…"):
        try:
            ocr = _cached_ocr(lang, float(rec_score_thresh))
            telescope = None if skip_sr else get_telescope(model_dir, use_gpu=use_gpu)
            item = evaluate_pil(
                pil,
                ocr,
                telescope,
                path=name,
                overlap=overlap,
                batch_size=batch_size,
                max_side=max_side,
                low_threshold=float(low_threshold),
                skip_sr=skip_sr,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"평가 실패: {exc}")
            st.exception(exc)
            return

    if item.error:
        st.error(item.error)
        return

    if item.delta_mean is not None:
        sign = "+" if item.delta_mean >= 0 else ""
        st.success(
            f"Δ mean {sign}{item.delta_mean:.4f} · Δ median {sign}{item.delta_median:.4f}"
        )
    else:
        st.success("평가 완료")

    if item.lr:
        st.subheader("LR OCR")
        _summary_metrics(item.lr.summary, low_threshold)
        with st.expander("LR 라인", expanded=False):
            _lines_table(item.lr.lines)
    if item.sr:
        st.subheader("SR OCR")
        _summary_metrics(item.sr.summary, low_threshold)
        with st.expander("SR 라인", expanded=False):
            _lines_table(item.sr.lines)


def _render_box_tab(
    pil: Image.Image,
    model_dir: Path,
    use_gpu: bool,
    lang: str,
    unclip_ratio: float,
    box_thresh: float,
    pad_ratio: float,
    det_model: str,
    rec_model: str,
    skip_sr: bool,
    max_boxes: int,
) -> None:
    st.caption(
        "1) TextDetection → 2) 박스 크롭 → 3) Telescope SR(패치) → 4) TextRecognition. "
        "박스마다 LR/SR로 인식된 단어를 비교합니다."
    )
    run = st.button(
        "디텍션 → SR → 인식 실행",
        type="primary",
        use_container_width=True,
        key="run_box_pipeline",
    )
    if not run:
        st.subheader("업로드 원본")
        st.image(pil, use_container_width=True)
        return

    if not skip_sr and not _model_ready(model_dir):
        st.error(f"Telescope 모델이 없습니다: `{model_dir}`")
        return

    with st.spinner("박스별 det → SR → rec 실행 중…"):
        try:
            detector = _cached_detector(float(unclip_ratio), float(box_thresh), det_model)
            recognizer = _cached_recognizer(lang, rec_model)
            telescope = None if skip_sr else get_telescope(model_dir, use_gpu=use_gpu)
            result = run_det_sr_rec(
                pil,
                detector,
                recognizer,
                telescope,
                pad_ratio=float(pad_ratio),
                skip_sr=skip_sr,
                max_boxes=int(max_boxes) if max_boxes > 0 else None,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"파이프라인 실패: {exc}")
            st.exception(exc)
            return

    st.success(
        f"완료 — {result.det_count}박스, {result.elapsed_sec * 1000:.0f} ms"
    )
    st.subheader("박스 오버레이")
    st.image(result.annotated, use_container_width=True)

    rows = []
    for box in result.boxes:
        delta = None
        if box.sr_score is not None:
            delta = box.sr_score - box.lr_score
        rows.append(
            {
                "#": box.index,
                "LR text": box.lr_text,
                "LR score": round(box.lr_score, 4),
                "SR text": box.sr_text if box.sr_text is not None else "",
                "SR score": round(box.sr_score, 4) if box.sr_score is not None else None,
                "Δ score": round(delta, 4) if delta is not None else None,
                "error": box.error or "",
            }
        )
    st.subheader("박스별 인식 결과")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("박스별 크롭 비교")
    for box in result.boxes:
        title = f"#{box.index}  LR: {box.lr_text!r}"
        if box.sr_text is not None:
            title += f"  →  SR: {box.sr_text!r}"
        with st.expander(title, expanded=box.index <= 5):
            if box.error:
                st.error(box.error)
                continue
            c1, c2 = st.columns(2)
            with c1:
                st.caption(f"LR crop · score={box.lr_score:.4f}")
                st.image(box.lr_crop, use_container_width=True)
                st.code(box.lr_text or "(empty)")
            with c2:
                if box.sr_crop is not None:
                    st.caption(
                        f"SR crop · score={box.sr_score:.4f}"
                        + (
                            f" · {box.sr_elapsed_sec * 1000:.0f} ms"
                            if box.sr_elapsed_sec is not None
                            else ""
                        )
                    )
                    st.image(box.sr_crop, use_container_width=True)
                    st.code(box.sr_text or "(empty)")
                else:
                    st.info("SR 없음 (건너뜀)")


st.set_page_config(
    page_title="PaddleOCR Text Telescope",
    page_icon="🔭",
    layout="wide",
)

st.title("Text Telescope")
st.caption(
    "전체 SR · 전체 OCR 비교 · 박스별 det→크롭→SR→인식"
)

with st.sidebar:
    st.header("공통")
    model_dir = Path(st.text_input("모델 경로", value=str(DEFAULT_MODEL)))
    use_gpu = st.checkbox("GPU 사용", value=False)
    lang = st.text_input("lang", value="korean")
    skip_sr = st.checkbox("SR 건너뛰기", value=False)

    st.divider()
    st.header("전체 타일 SR")
    overlap = st.slider("타일 오버랩 (px)", 0, 12, 8)
    batch_size = st.slider("배치 크기", 1, 16, 8)
    limit_side = st.checkbox("긴 변 제한", value=True)
    max_side = st.number_input(
        "긴 변 최대 px", 256, 4096, 1280, 64, disabled=not limit_side
    )

    st.divider()
    st.header("디텍션 / 인식")
    unclip_ratio = st.slider("det unclip_ratio", 1.0, 3.0, 1.5, 0.1)
    box_thresh = st.slider("det box_thresh (0=기본)", 0.0, 1.0, 0.0, 0.05)
    pad_ratio = st.slider("크롭 여유 pad_ratio", 0.0, 0.3, 0.05, 0.01)
    max_boxes = st.number_input("최대 박스 수 (0=무제한)", 0, 500, 0, 10)
    det_model = st.text_input("det model_name (빈칸=기본)", value="")
    rec_model = st.text_input("rec model_name (빈칸=lang 매핑)", value="")

    st.divider()
    st.header("전체 OCR 평가")
    low_threshold = st.slider("low-confidence 기준", 0.0, 1.0, 0.8, 0.05)
    rec_score_thresh = st.number_input("rec score thresh", 0.0, 1.0, 0.0, 0.05)

    st.divider()
    st.markdown(
        "```bash\npip install -r requirements.txt\n"
        "python scripts/setup_model.py\n```"
    )

max_side_arg = int(max_side) if limit_side else None

uploaded = st.file_uploader(
    "영수증 / 저해상도 이미지 업로드",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded is None:
    st.info("이미지를 업로드한 뒤 탭에서 실행하세요.")
    st.stop()

pil = Image.open(uploaded).convert("RGB")
c_meta1, c_meta2 = st.columns(2)
with c_meta1:
    st.write(f"원본 크기: **{pil.size[0]} × {pil.size[1]}**")
with c_meta2:
    st.write(f"파일: `{uploaded.name}`")

tab_box, tab_sr, tab_eval = st.tabs(
    ["박스별 det→SR→rec", "전체 SR", "전체 OCR 신뢰도"]
)

with tab_box:
    _render_box_tab(
        pil,
        model_dir,
        use_gpu,
        lang,
        float(unclip_ratio),
        float(box_thresh),
        float(pad_ratio),
        det_model.strip(),
        rec_model.strip(),
        skip_sr,
        int(max_boxes),
    )

with tab_sr:
    if not _model_ready(model_dir):
        st.warning(f"추론 모델이 없습니다: `{model_dir}`")
    else:
        _render_sr_tab(
            pil,
            uploaded.name,
            model_dir,
            use_gpu,
            int(overlap),
            int(batch_size),
            max_side_arg,
        )

with tab_eval:
    _render_eval_tab(
        pil,
        uploaded.name,
        model_dir,
        use_gpu,
        int(overlap),
        int(batch_size),
        max_side_arg,
        lang,
        float(low_threshold),
        float(rec_score_thresh),
        skip_sr,
    )
