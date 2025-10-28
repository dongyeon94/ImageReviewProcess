"""프로젝트 설정 파일"""

import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.absolute()

# 데이터 디렉토리
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# 이미지 설정
IMAGE_CONFIG = {
    'max_size': (1024, 1024),
    'formats': ['jpg', 'jpeg', 'png', 'bmp', 'gif']
}

# 모델 설정
MODEL_CONFIG = {
    'sentiment_model': 'distilbert-base-uncased-finetuned-sst-2-english',
    'device': 'cuda' if os.environ.get('USE_GPU', '').lower() == 'true' else 'cpu',
}

# 분류 설정
CLASSIFICATION_CONFIG = {
    'wearing_categories': ['full_body', 'upper_body', 'lower_body', 'detail'],
    'product_categories': ['product_only', 'with_packaging']
}

# 임계값 설정
THRESHOLD_CONFIG = {
    'sentiment_confidence': 0.7,
    'classification_confidence': 0.7,
    'similarity_threshold': 0.75
}
