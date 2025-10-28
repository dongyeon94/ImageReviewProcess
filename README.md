# Image Review Process

이미지 리뷰를 분석하는 AI 기반 프로젝트입니다.

## 프로젝트 개요

### 주요 기능

1. **감정 분석 (Sentiment Analysis)**
   - 리뷰 텍스트의 긍정/부정 감정 분석
   - 신뢰도 점수 제공

2. **이미지 분류 (Image Classification)**
   - 착용 리뷰 vs 상품 리뷰 분류
   - 착용 리뷰: 신체 일부 여부 판단 (전신, 상반신, 하반신 등)
   - 상품 리뷰: 리뷰 상품과 대상 상품 유사도 분석

3. **이미지 유사도 분석 (Image Similarity)**
   - 두 이미지 간의 유사도 계산

4. **토픽 추출 (Topic Extraction)**
   - 리뷰에서 주요 토픽/키워드 추출

## 프로젝트 구조

```
ImageReviewProcess/
├── src/                          # 소스 코드
│   ├── sentiment/               # 감정 분석 모듈
│   │   └── analyzer.py
│   ├── classification/          # 이미지 분류 모듈
│   │   └── image_classifier.py
│   ├── image/                   # 이미지 처리 모듈
│   │   └── similarity.py
│   ├── topic/                   # 토픽 추출 모듈
│   │   └── extractor.py
│   └── processor.py             # 통합 처리 모듈
├── data/                        # 데이터 디렉토리
├── models/                      # 학습된 모델 저장
├── results/                     # 분석 결과
├── tests/                       # 테스트 코드
├── config.py                    # 설정 파일
├── main.py                      # 메인 엔트리포인트
├── requirements.txt             # 의존성 패키지
├── .env.example                 # 환경 변수 예제
└── README.md                    # 프로젝트 설명

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 필요에 따라 수정
```

## 사용법

### 기본 사용

```python
from src.processor import ImageReviewProcessor

processor = ImageReviewProcessor()

result = processor.process(
    image_path='path/to/image.jpg',
    review_text='리뷰 텍스트',
    target_product='대상 상품명'  # 선택사항
)

print(result)
```

### 반환 값

```python
{
    'image_path': 'path/to/image.jpg',
    'review_text': '리뷰 텍스트',
    'sentiment': {
        'sentiment': 'positive',
        'confidence': 0.95,
        'scores': {'positive': 0.95, 'negative': 0.05}
    },
    'classification': {
        'category': 'wearing',
        'confidence': 0.87,
        'details': {...}
    },
    'topics': [
        {'topic': 'quality', 'confidence': 0.95},
        {'topic': 'fit', 'confidence': 0.87}
    ],
    'category_confidence': 0.87
}
```

## 모듈 설명

### 1. SentimentAnalyzer
리뷰 텍스트의 감정을 분석합니다.

### 2. ImageClassifier
이미지를 착용 리뷰/상품 리뷰로 분류하고 추가 분석을 수행합니다.

### 3. TopicExtractor
리뷰에서 주요 토픽을 추출합니다.

### 4. ImageSimilarityAnalyzer
두 이미지 간의 유사도를 계산합니다.

## 기술 스택

- **Python**: 3.8+
- **Image Processing**: OpenCV, Pillow
- **NLP**: Transformers, NLTK, Gensim
- **ML**: scikit-learn, PyTorch
- **Configuration**: python-dotenv, pydantic

## 라이선스

MIT License

## 개발자

Dongyeon Kim
