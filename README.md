# PaddleOCR Text Telescope Demo

[Scene Text Telescope](https://www.paddleocr.ai/v2.10.0/en/algorithm/super_resolution/algorithm_sr_telescope.html) (CVPR 2021) 텍스트 초해상도 모델을 Streamlit으로 테스트합니다. 이미지를 올리면 **적용 전(LR) / 적용 후(SR)** 를 나란히 확인할 수 있습니다.

> 이 모델은 **단어·텍스트 행 크롭**에 맞춰져 있습니다. 전체 장면 사진보다는 OCR로 자른 텍스트 패치를 넣는 편이 효과가 좋습니다.

## 빠른 시작

Python 3.10+ (3.13 포함). macOS/arm64에서는 PyPI에 **paddlepaddle 3.x**만 제공됩니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 공식 가중치 다운로드 + inference 모델 export (최초 1회)
python scripts/setup_model.py

# UI 실행
streamlit run app.py
```

브라우저에서 이미지를 업로드한 뒤 **Telescope 실행**을 누르면 전후 비교가 표시됩니다.

## 구성

| 경로 | 설명 |
|------|------|
| `app.py` | Streamlit UI |
| `telescope_sr.py` | Paddle Inference 래퍼 |
| `scripts/setup_model.py` | PaddleOCR clone + 가중치 다운로드 + export |
| `models/sr_telescope/` | export된 inference 모델 (setup 후 생성) |

## 참고

- 학습/export는 PaddleOCR `release/2.7` 기준입니다.
- 기본 입력 shape: `3×32×128` (내부에서 ×0.5 bicubic downsample 후 SR).
- macOS / CPU 기본. GPU를 쓰려면 사이드바에서 GPU 옵션을 켜고 CUDA용 `paddlepaddle-gpu`를 설치하세요.
