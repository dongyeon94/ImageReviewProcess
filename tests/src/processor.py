"""Main Image Review Processor

전체 이미지 리뷰 분석 처리를 조율하는 메인 모듈
"""

from sentiment.analyzer import SentimentAnalyzer
from classification.image_classifier import ImageClassifier
from topic.extractor import TopicExtractor
from image.similarity import ImageSimilarityAnalyzer


class ImageReviewProcessor:
    """이미지 리뷰 분석을 통합 처리"""

    def __init__(self):
        """프로세서 초기화"""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.image_classifier = ImageClassifier()
        self.topic_extractor = TopicExtractor()
        self.similarity_analyzer = ImageSimilarityAnalyzer()

    def process(self, image_path: str, review_text: str, target_product: str = None) -> dict:
        """
        리뷰 이미지와 텍스트를 종합 분석

        Args:
            image_path: 리뷰 이미지 경로
            review_text: 리뷰 텍스트
            target_product: 대상 상품 정보 (선택)

        Returns:
            {
                'sentiment': {...},
                'classification': {...},
                'topics': [...],
                'confidence': float
            }
        """
        result = {
            'image_path': image_path,
            'review_text': review_text,
            'sentiment': None,
            'classification': None,
            'topics': None,
            'category_confidence': None
        }

        # 1. 이미지 분류
        result['classification'] = self.image_classifier.classify(image_path)
        result['category_confidence'] = result['classification'].get('confidence', 0)

        # 2. 감정 분석
        result['sentiment'] = self.sentiment_analyzer.analyze(review_text)

        # 3. 토픽 추출
        result['topics'] = self.topic_extractor.extract(review_text)

        # 4. 상품 리뷰인 경우 추가 분석
        if result['classification'].get('category') == 'product' and target_product:
            result['product_analysis'] = self.image_classifier.analyze_product_review(
                image_path,
                target_product
            )

        return result
