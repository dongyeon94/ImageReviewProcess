"""통합 프로세서 테스트"""

import unittest
from src.processor import ImageReviewProcessor


class TestImageReviewProcessor(unittest.TestCase):
    """ImageReviewProcessor 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.processor = ImageReviewProcessor()

    def test_processor_initialization(self):
        """프로세서 초기화 테스트"""
        self.assertIsNotNone(self.processor.sentiment_analyzer)
        self.assertIsNotNone(self.processor.image_classifier)
        self.assertIsNotNone(self.processor.topic_extractor)
        self.assertIsNotNone(self.processor.similarity_analyzer)

    def test_process_returns_dict(self):
        """처리 결과가 딕셔너리 반환하는지 테스트"""
        # 테스트 이미지와 텍스트 필요
        # result = self.processor.process('test_image.jpg', 'test review text')
        # self.assertIsInstance(result, dict)
        # self.assertIn('sentiment', result)
        # self.assertIn('classification', result)
        # self.assertIn('topics', result)
        pass


if __name__ == '__main__':
    unittest.main()
