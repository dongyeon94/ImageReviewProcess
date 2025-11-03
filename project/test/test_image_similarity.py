import unittest
import sys
import os
import cv2
import numpy as np
import tempfile
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.similarity.ImageSimilarity import ImageSimilarity


class TestImageSimilarity(unittest.TestCase):
    """ImageSimilarity 클래스 테스트"""

    @classmethod
    def setUpClass(cls):
        """테스트 시작 전 샘플 이미지 생성"""
        # 임시 디렉토리 생성
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = cls.temp_dir.name

        # 고엣지 이미지 생성 (체스판 패턴)
        cls.high_edge_image_path = os.path.join(cls.temp_path, 'high_edge.png')
        high_edge_img = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                if (i // 10 + j // 10) % 2 == 0:
                    high_edge_img[i:i+10, j:j+10] = 255
        cv2.imwrite(cls.high_edge_image_path, high_edge_img)

        # 저엣지 이미지 생성 (단색)
        cls.low_edge_image_path = os.path.join(cls.temp_path, 'low_edge.png')
        low_edge_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.imwrite(cls.low_edge_image_path, low_edge_img)

        # 그라디언트 이미지 생성
        cls.gradient_image_path = os.path.join(cls.temp_path, 'gradient.png')
        gradient_img = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            gradient_img[i, :] = int(255 * (i / 100))
        cv2.imwrite(cls.gradient_image_path, gradient_img)

    @classmethod
    def tearDownClass(cls):
        """테스트 후 임시 파일 정리"""
        cls.temp_dir.cleanup()

    def test_initialization_with_gpu(self):
        """GPU 설정으로 초기화 테스트"""
        print("\n[TEST] ImageSimilarity 초기화 테스트 (GPU)")
        similarity = ImageSimilarity(use_gpu=True)
        self.assertIsNotNone(similarity)
        self.assertIsNotNone(similarity.device)
        print(f"✓ Device: {similarity.device}")

    def test_initialization_without_gpu(self):
        """CPU 설정으로 초기화 테스트"""
        print("\n[TEST] ImageSimilarity 초기화 테스트 (CPU)")
        similarity = ImageSimilarity(use_gpu=False)
        self.assertIsNotNone(similarity)
        self.assertEqual(str(similarity.device), 'cpu')
        print(f"✓ Device: {similarity.device}")

    def test_edge_detection_high_edge_image(self):
        """고엣지 이미지 테스트 (엣지가 많은 이미지)"""
        print("\n[TEST] 고엣지 이미지 엣지 감지")
        similarity = ImageSimilarity(use_gpu=False)
        result = similarity._EdgeDetection(self.high_edge_image_path, plot=False)

        # 고엣지 이미지는 True를 반환해야 함
        self.assertTrue(result)
        print(f"✓ 고엣지 이미지 결과: {result}")

    def test_edge_detection_low_edge_image(self):
        """저엣지 이미지 테스트 (엣지가 적은 이미지)"""
        print("\n[TEST] 저엣지 이미지 엣지 감지")
        similarity = ImageSimilarity(use_gpu=False)
        result = similarity._EdgeDetection(self.low_edge_image_path, plot=False)

        # 저엣지 이미지는 False를 반환해야 함
        self.assertFalse(result)
        print(f"✓ 저엣지 이미지 결과: {result}")

    def test_edge_detection_return_type(self):
        """반환값이 boolean 타입인지 확인"""
        print("\n[TEST] 반환값 타입 확인")
        similarity = ImageSimilarity(use_gpu=False)
        result = similarity._EdgeDetection(self.high_edge_image_path, plot=False)

        self.assertIsInstance(result, (bool, np.bool_))
        print(f"✓ 반환값 타입: {type(result)}")

    def test_edge_detection_with_invalid_path(self):
        """존재하지 않는 이미지 경로로 테스트"""
        print("\n[TEST] 유효하지 않은 경로 처리")
        similarity = ImageSimilarity(use_gpu=False)

        with self.assertRaises((cv2.error, FileNotFoundError, TypeError)):
            similarity._EdgeDetection('/invalid/path/to/image.png', plot=False)
        print("✓ 유효하지 않은 경로 예외 처리됨")

    def test_edge_detection_output_consistency(self):
        """같은 이미지 여러 번 실행 시 결과 일관성"""
        print("\n[TEST] 출력 일관성 확인")
        similarity = ImageSimilarity(use_gpu=False)
        result1 = similarity._EdgeDetection(self.gradient_image_path, plot=False)
        result2 = similarity._EdgeDetection(self.gradient_image_path, plot=False)

        self.assertEqual(result1, result2)
        print(f"✓ 첫 번째 실행: {result1}, 두 번째 실행: {result2}")

    def test_inheritance_from_validation(self):
        """ImageSimilarity가 ImageValidation을 올바르게 상속받는지 확인"""
        print("\n[TEST] 클래스 상속 확인")
        similarity = ImageSimilarity(use_gpu=False)

        # _EdgeDetection 메서드가 존재하는지 확인
        self.assertTrue(hasattr(similarity, '_EdgeDetection'))
        # imagePlot 메서드가 존재하는지 확인
        self.assertTrue(hasattr(similarity, 'imagePlot'))
        print("✓ ImageValidation 상속 완료")


if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)
