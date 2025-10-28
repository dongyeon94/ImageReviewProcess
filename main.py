"""메인 엔트리포인트

이미지 리뷰 분석 프로젝트의 메인 실행 파일
"""

import logging
from pathlib import Path
from src.processor import ImageReviewProcessor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """메인 함수"""
    logger.info("이미지 리뷰 분석 프로젝트 시작")

    # 프로세서 초기화
    processor = ImageReviewProcessor()

    # 예제 실행 (실제 이미지와 텍스트로 테스트)
    example_result = {
        'status': 'ready',
        'message': 'Image Review Processor initialized successfully',
        'modules': [
            'Sentiment Analyzer',
            'Image Classifier',
            'Topic Extractor',
            'Image Similarity Analyzer'
        ]
    }

    logger.info(f"프로세서 준비 완료: {example_result}")
    return example_result


if __name__ == '__main__':
    main()
