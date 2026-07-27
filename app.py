"""Streamlit demo: classic preprocess + OCR eval + det→rec."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from classic_enhance import CLASSIC_METHODS, ClassicMethod, get_classic
from text_sr_pipeline import (
    create_text_detector,
    create_text_recognizer,
    run_det_pp_rec,
)

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_sr_ocr import (  # noqa: E402
    create_paddle_ocr,
    evaluate_pil,
)


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


def _render_eval_tab(
    pil: Image.Image,
    name: str,
    lang: str,
    low_threshold: float,
    rec_score_thresh: float,
    skip_pp: bool,
    pp_method: ClassicMethod,
) -> None:
    run = st.button(
        "LR / 고전 전처리 OCR 평가 실행",
        type="primary",
        use_container_width=True,
        key="run_eval",
    )
    if not run:
        st.subheader("업로드 원본 (LR)")
        st.image(pil, use_container_width=True)
        st.caption(
            "전체 이미지에 OCR을 돌려 LR · 고전 전처리 후 rec confidence를 비교합니다."
        )
        return

    with st.spinner("OCR 평가 중…"):
        try:
            ocr = _cached_ocr(lang, float(rec_score_thresh))
            enhancer = None if skip_pp else get_classic(method=pp_method)
            item = evaluate_pil(
                pil,
                ocr,
                enhancer,
                path=name,
                low_threshold=float(low_threshold),
                pp_method=pp_method,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"평가 실패: {exc}")
            st.exception(exc)
            return

    if item.error:
        st.error(item.error)
        return

    if item.best_by_mean:
        labels = {"lr": "LR", "pp": f"classic ({pp_method})"}
        st.success(f"best_by_mean: **{labels.get(item.best_by_mean, item.best_by_mean)}**")

    if item.delta_mean_pp is not None:
        sign = "+" if item.delta_mean_pp >= 0 else ""
        st.caption(f"classic Δ mean {sign}{item.delta_mean_pp:.4f}")

    compare_rows = []
    if item.lr:
        compare_rows.append(
            {
                "variant": "LR",
                "lines": item.lr.summary.count,
                "mean": round(item.lr.summary.mean, 4),
                "median": round(item.lr.summary.median, 4),
                "low_ratio": f"{item.lr.summary.low_ratio:.0%}",
            }
        )
    if item.pp:
        compare_rows.append(
            {
                "variant": f"classic ({item.pp_method})",
                "lines": item.pp.summary.count,
                "mean": round(item.pp.summary.mean, 4),
                "median": round(item.pp.summary.median, 4),
                "low_ratio": f"{item.pp.summary.low_ratio:.0%}",
            }
        )
    if compare_rows:
        st.subheader("신뢰도 요약")
        st.dataframe(compare_rows, use_container_width=True, hide_index=True)

    if item.pp_image is not None:
        st.subheader("전처리 결과 미리보기")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("LR (원본)")
            st.image(pil, use_container_width=True)
        with c2:
            st.caption(
                f"classic ({item.pp_method})"
                + (
                    f" · {item.pp_image.shape[1]}×{item.pp_image.shape[0]}"
                    if item.pp_image is not None
                    else ""
                )
                + (
                    f" · {item.pp_elapsed_sec * 1000:.0f} ms"
                    if item.pp_elapsed_sec is not None
                    else ""
                )
            )
            st.image(item.pp_image, use_container_width=True)

        buf = BytesIO()
        Image.fromarray(item.pp_image).save(buf, format="PNG")
        st.download_button(
            "전처리 결과 PNG 다운로드",
            data=buf.getvalue(),
            file_name=f"pp_{pp_method}_{Path(name).stem}.png",
            mime="image/png",
            key="dl_pp",
        )

    if item.lr:
        st.subheader("LR OCR")
        _summary_metrics(item.lr.summary, low_threshold)
        with st.expander("LR 라인", expanded=False):
            _lines_table(item.lr.lines)
    if item.pp:
        st.subheader(f"classic OCR ({item.pp_method})")
        _summary_metrics(item.pp.summary, low_threshold)
        with st.expander("classic 라인", expanded=False):
            _lines_table(item.pp.lines)


def _render_box_tab(
    pil: Image.Image,
    lang: str,
    unclip_ratio: float,
    box_thresh: float,
    pad_ratio: float,
    det_model: str,
    rec_model: str,
    max_boxes: int,
    skip_pp: bool,
    pp_method: ClassicMethod,
) -> None:
    st.caption(
        "1) 전체 이미지 고전 전처리 → 2) TextDetection → 3) 박스 크롭 → 4) TextRecognition. "
        "같은 박스 좌표로 LR / 전처리 크롭을 비교합니다."
    )
    run = st.button(
        "고전 전처리 → det → rec 실행",
        type="primary",
        use_container_width=True,
        key="run_box_pipeline",
    )
    if not run:
        st.subheader("업로드 원본")
        st.image(pil, use_container_width=True)
        return

    with st.spinner("고전 전처리 → det → rec 실행 중…"):
        try:
            detector = _cached_detector(float(unclip_ratio), float(box_thresh), det_model)
            recognizer = _cached_recognizer(lang, rec_model)
            enhancer = None if skip_pp else get_classic(method=pp_method)
            result = run_det_pp_rec(
                pil,
                detector,
                recognizer,
                enhancer,
                pad_ratio=float(pad_ratio),
                skip_pp=skip_pp,
                max_boxes=int(max_boxes) if max_boxes > 0 else None,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"파이프라인 실패: {exc}")
            st.exception(exc)
            return

    timing = f"완료 — {result.det_count}박스, {result.elapsed_sec * 1000:.0f} ms"
    if result.pp_elapsed_sec is not None:
        timing += f" (전처리 {result.pp_elapsed_sec * 1000:.0f} ms)"
    st.success(timing)

    if result.pp_image is not None:
        st.subheader("전체 이미지 전처리")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("LR (원본)")
            st.image(result.image, use_container_width=True)
        with c2:
            st.caption(
                f"classic ({result.pp_method or pp_method})"
                + (f" · ×{result.pp_scale:g}" if result.pp_scale != 1.0 else "")
            )
            st.image(result.pp_image, use_container_width=True)

    st.subheader("박스 오버레이")
    st.image(result.annotated, use_container_width=True)

    rows = []
    for box in result.boxes:
        delta = None
        if box.pp_score is not None:
            delta = box.pp_score - box.lr_score
        rows.append(
            {
                "#": box.index,
                "LR text": box.lr_text,
                "LR score": round(box.lr_score, 4),
                "PP text": box.pp_text if box.pp_text is not None else "",
                "PP score": round(box.pp_score, 4) if box.pp_score is not None else None,
                "Δ score": round(delta, 4) if delta is not None else None,
                "error": box.error or "",
            }
        )
    st.subheader("박스별 인식 결과")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("박스별 크롭 비교")
    for box in result.boxes:
        title = f"#{box.index}  LR: {box.lr_text!r}"
        if box.pp_text is not None:
            title += f"  →  PP: {box.pp_text!r}"
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
                if box.pp_crop is not None:
                    st.caption(f"PP crop · score={box.pp_score:.4f}")
                    st.image(box.pp_crop, use_container_width=True)
                    st.code(box.pp_text or "(empty)")
                else:
                    st.info("전처리 없음 (건너뜀)")


st.set_page_config(
    page_title="Classic Preprocess OCR",
    page_icon="🧾",
    layout="wide",
)

st.title("Classic Preprocess OCR")
st.caption("LR / 고전 전처리 OCR 비교 · 전체 전처리 → det → 크롭 → 인식")

with st.sidebar:
    st.header("공통")
    lang = st.text_input("lang", value="korean")

    st.divider()
    st.header("디텍션 / 인식")
    unclip_ratio = st.slider("det unclip_ratio", 1.0, 3.0, 1.5, 0.1)
    box_thresh = st.slider("det box_thresh (0=기본)", 0.0, 1.0, 0.0, 0.05)
    pad_ratio = st.slider("크롭 여유 pad_ratio", 0.0, 0.3, 0.05, 0.01)
    max_boxes = st.number_input("최대 박스 수 (0=무제한)", 0, 500, 0, 10)
    det_model = st.text_input("det model_name (빈칸=기본)", value="")
    rec_model = st.text_input("rec model_name (빈칸=lang 매핑)", value="")

    st.divider()
    st.header("전처리 / OCR 평가")
    low_threshold = st.slider("low-confidence 기준", 0.0, 1.0, 0.8, 0.05)
    rec_score_thresh = st.number_input("rec score thresh", 0.0, 1.0, 0.0, 0.05)
    skip_pp = st.checkbox("전처리 건너뛰기", value=False)
    pp_method: ClassicMethod = st.selectbox(  # type: ignore[assignment]
        "전처리 method",
        options=list(CLASSIC_METHODS),
        index=list(CLASSIC_METHODS).index("receipt"),
        help="receipt=×3→CLAHE→Unsharp→mild denoise (기본), adaptive/otsu=이진화",
    )

    st.divider()
    st.markdown("```bash\npip install -r requirements.txt\nstreamlit run app.py\n```")

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

tab_eval, tab_box = st.tabs(["전체 OCR 신뢰도", "전처리 → det → rec"])

with tab_eval:
    _render_eval_tab(
        pil,
        uploaded.name,
        lang,
        float(low_threshold),
        float(rec_score_thresh),
        skip_pp,
        pp_method,
    )

with tab_box:
    _render_box_tab(
        pil,
        lang,
        float(unclip_ratio),
        float(box_thresh),
        float(pad_ratio),
        det_model.strip(),
        rec_model.strip(),
        int(max_boxes),
        skip_pp,
        pp_method,
    )
