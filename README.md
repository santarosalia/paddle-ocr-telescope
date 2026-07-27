# Classic Preprocess Demo

영수증/문서 이미지에 **OpenCV 고전 전처리**를 적용하는 Streamlit 데모입니다.

## 빠른 시작

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

이미지를 업로드한 뒤 **전처리 실행**을 누르면 전후 비교가 표시됩니다.

## 기본 파이프라인 (`receipt`)

```
원본 → ×3 upscale → CLAHE → Unsharp Mask → mild denoise
```

| method | 설명 |
|--------|------|
| `receipt` | 위 파이프라인 (기본) |
| `clahe` | 대비 보정만 |
| `unsharp` | 언샤프만 |
| `adaptive` | adaptive 이진화 |
| `otsu` | Otsu 이진화 |

## 구성

| 경로 | 설명 |
|------|------|
| `app.py` | Streamlit UI |
| `classic_enhance.py` | OpenCV 전처리 |
