"""감정 분석 테스트"""

import unittest
from src.sentiment.analyzer import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """SentimentAnalyzer 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.analyzer = SentimentAnalyzer()

    def test_positive_sentiment(self):
        """긍정 감정 분석 테스트"""
        text = "상품이 정말 좋아요! 만족합니다."
        result = self.analyzer.analyze(text)
        self.assertIsNotNone(result)
        self.assertIn('sentiment', result)

    def test_negative_sentiment(self):
        """부정 감정 분석 테스트"""
        text = "정말 실망했어요. 품질이 안 좋네요."
        result = self.analyzer.analyze(text)
        self.assertIsNotNone(result)
        self.assertIn('sentiment', result)

    def test_confidence_score(self):
        """신뢰도 점수 테스트"""
        text = "좋습니다."
        result = self.analyzer.analyze(text)
        if result:
            self.assertIn('confidence', result)
            self.assertGreaterEqual(result['confidence'], 0)
            self.assertLessEqual(result['confidence'], 1)


if __name__ == '__main__':
    unittest.main()
