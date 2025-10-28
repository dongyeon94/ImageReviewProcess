"""이미지 분류 테스트"""

import unittest
from src.classification.image_classifier import ImageClassifier


class TestImageClassifier(unittest.TestCase):
    """ImageClassifier 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.classifier = ImageClassifier()

    def test_classify_returns_dict(self):
        """분류 결과가 딕셔너리 반환하는지 테스트"""
        # 테스트 이미지 경로 (실제 이미지 필요)
        # result = self.classifier.classify('test_image.jpg')
        # self.assertIsInstance(result, dict)
        pass

    def test_wearing_review_analysis(self):
        """착용 리뷰 분석 테스트"""
        # result = self.classifier.analyze_wearing_review('test_image.jpg')
        # self.assertIn('has_body', result)
        # self.assertIn('body_parts', result)
        pass

    def test_product_review_analysis(self):
        """상품 리뷰 분석 테스트"""
        # result = self.classifier.analyze_product_review('test_image.jpg', 'target_product')
        # self.assertIn('similarity_score', result)
        # self.assertIn('is_same_product', result)
        pass


if __name__ == '__main__':
    unittest.main()
