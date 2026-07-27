# Classic Preprocess OCR Demo

영수증/문서 이미지에 **OpenCV 고전 전처리**를 적용한 뒤 PaddleOCR 신뢰도를 비교하는 Streamlit 데모입니다.

## 빠른 시작

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

이미지를 업로드한 뒤 **LR / 고전 전처리 OCR 평가 실행**을 누르면 전후 비교가 표시됩니다.

## 전처리 method

| method | 설명 |
|--------|------|
| `receipt` | ×3 upscale → CLAHE → Unsharp → mild denoise (기본) |
| `clahe` | 대비 보정 |
| `unsharp` | 언샤프 마스크 |
| `adaptive` | adaptive 이진화 |
| `otsu` | Otsu 이진화 |

## 구성

| 경로 | 설명 |
|------|------|
| `app.py` | Streamlit UI |
| `classic_enhance.py` | OpenCV 전처리 |
| `scripts/eval_sr_ocr.py` | CLI OCR 신뢰도 비교 (LR vs classic) |
| `text_sr_pipeline.py` | 전체 전처리 → det → crop → rec |
