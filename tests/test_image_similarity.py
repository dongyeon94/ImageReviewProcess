"""이미지 유사도 분석 테스트"""

import unittest
import os
from src.image.similarity import ImageSimilarityAnalyzer


class TestImageSimilarityAnalyzer(unittest.TestCase):
    """ImageSimilarityAnalyzer 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.analyzer = ImageSimilarityAnalyzer(verbose=True)
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.original_path = os.path.join(self.data_path, 'original_image')
        self.review_path = os.path.join(self.data_path, 'review_image')

    def test_similarity_same_image(self):
        """동일한 이미지 유사도 테스트 (같은 이미지끼리 비교)"""
        image_path = os.path.join(self.original_path, 'product1_front.jpg')

        # 파일 존재 확인
        self.assertTrue(os.path.exists(image_path), f"이미지 파일이 없습니다: {image_path}")

        # 같은 이미지끼리 비교
        similarity = self.analyzer.calculate_similarity(image_path, image_path)

        # 같은 이미지는 1에 가까워야 함 (부동소수점 오차 허용)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.99)
        self.assertLessEqual(similarity, 1.001)
        print(f"✓ 동일 이미지 유사도: {similarity:.4f}")

    def test_similarity_different_products(self):
        """서로 다른 상품 이미지 유사도 테스트"""
        image1 = os.path.join(self.original_path, 'product1_front.jpg')
        image2 = os.path.join(self.original_path, 'product2_front.jpg')

        self.assertTrue(os.path.exists(image1), f"이미지 파일이 없습니다: {image1}")
        self.assertTrue(os.path.exists(image2), f"이미지 파일이 없습니다: {image2}")

        similarity = self.analyzer.calculate_similarity(image1, image2)

        # 다른 상품은 유사도가 낮아야 함
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        print(f"✓ 다른 상품 유사도: {similarity:.4f}")

    def test_similarity_same_product_different_angle(self):
        """같은 상품의 다른 각도 이미지 유사도 테스트"""
        image1 = os.path.join(self.original_path, 'product1_front.jpg')
        image2 = os.path.join(self.original_path, 'product1_back.jpg')

        self.assertTrue(os.path.exists(image1), f"이미지 파일이 없습니다: {image1}")
        self.assertTrue(os.path.exists(image2), f"이미지 파일이 없습니다: {image2}")

        similarity = self.analyzer.calculate_similarity(image1, image2)

        # 같은 상품이지만 다른 각도는 중간 정도의 유사도
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        print(f"✓ 같은 상품 다른 각도 유사도: {similarity:.4f}")

    def test_similarity_original_vs_review(self):  ############################
        """원본 이미지와 리뷰 이미지 유사도 테스트"""
        original = os.path.join(self.original_path, 'product2_front.jpg')
        review = os.path.join(self.review_path, 'product2_1.jpg')

        self.assertTrue(os.path.exists(original), f"이미지 파일이 없습니다: {original}")
        self.assertTrue(os.path.exists(review), f"이미지 파일이 없습니다: {review}")

        similarity = self.analyzer.calculate_similarity(original, review)

        # 유사도가 0-1 범위
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        print(f"✓ 원본 vs 리뷰 이미지 유사도: {similarity:.4f}")

    def test_similarity_all_review_images_for_product2(self):
        """Product1의 모든 리뷰 이미지와 원본 이미지의 유사도 테스트"""
        original = os.path.join(self.original_path, 'product2_front.jpg')

        self.assertTrue(os.path.exists(original), f"이미지 파일이 없습니다: {original}")

        # product1_1, product1_2, product1_3 리뷰 이미지
        review_images = ['product2_1.jpg', 'product2_2.jpg', 'product2_3.jpg']

        print(f"\nProduct2 리뷰 이미지 유사도:")
        for review_name in review_images:
            review_path = os.path.join(self.review_path, review_name)

            if os.path.exists(review_path):
                similarity = self.analyzer.calculate_similarity(original, review_path)
                self.assertGreaterEqual(similarity, 0.0)
                self.assertLessEqual(similarity, 1.0)
                print(f"  - {review_name}: {similarity:.4f}")

    def test_similarity_consistency(self):
        """유사도 계산의 일관성 테스트 (같은 비교 두 번)"""
        image1 = os.path.join(self.original_path, 'product1_front.jpg')
        image2 = os.path.join(self.original_path, 'product2_front.jpg')

        self.assertTrue(os.path.exists(image1), f"이미지 파일이 없습니다: {image1}")
        self.assertTrue(os.path.exists(image2), f"이미지 파일이 없습니다: {image2}")

        # 같은 비교를 두 번 실행
        similarity1 = self.analyzer.calculate_similarity(image1, image2)
        similarity2 = self.analyzer.calculate_similarity(image1, image2)

        # 결과가 일관성 있어야 함
        self.assertAlmostEqual(similarity1, similarity2, places=5)
        print(f"✓ 유사도 계산 일관성: {similarity1:.4f} == {similarity2:.4f}")

    def test_invalid_path(self):
        """유효하지 않은 경로 테스트"""
        image1 = os.path.join(self.original_path, 'product1_front.jpg')
        invalid_path = '/invalid/path/image.jpg'

        self.assertTrue(os.path.exists(image1), f"이미지 파일이 없습니다: {image1}")

        # 유효하지 않은 경로로 비교 시 예외 발생
        with self.assertRaises(Exception):
            self.analyzer.calculate_similarity(image1, invalid_path)

        print("✓ 유효하지 않은 경로 예외 처리 확인")

    def test_stage0_quality_validation(self):
        """Stage 0 이미지 품질 검증 테스트"""
        image_path = os.path.join(self.original_path, 'product1_front.jpg')

        self.assertTrue(os.path.exists(image_path), f"이미지 파일이 없습니다: {image_path}")

        # Stage 0 직접 테스트
        passed, quality_scores = self.analyzer.stage0_image_quality_validation(image_path)

        # 결과 검증
        self.assertIsInstance(passed, bool)
        self.assertIsInstance(quality_scores, dict)
        self.assertIn('laplacian_variance', quality_scores)
        self.assertIn('edge_density', quality_scores)
        self.assertIn('is_single_color', quality_scores)
        self.assertIn('color_diversity', quality_scores)

        print(f"\n✓ Stage 0 품질 검증 완료")
        print(f"  - 통과 여부: {passed}")
        print(f"  - Laplacian 분산: {quality_scores['laplacian_variance']:.2f}")
        print(f"  - 엣지 밀도: {quality_scores['edge_density']:.4f}")
        print(f"  - 단색 여부: {quality_scores['is_single_color']}")
        print(f"  - 색상 다양성: {quality_scores['color_diversity']:.4f}")
        if quality_scores['pass_reasons']:
            print(f"  - 통과 사유:")
            for reason in quality_scores['pass_reasons']:
                print(f"    ✓ {reason}")
        if quality_scores['fail_reasons']:
            print(f"  - 실패 사유:")
            for reason in quality_scores['fail_reasons']:
                print(f"    ✗ {reason}")

    def test_quality_validation_affects_similarity(self):
        """Stage 0 품질 검증이 최종 유사도에 영향을 미치는지 테스트"""
        image1 = os.path.join(self.original_path, 'product1_front.jpg')
        image2 = os.path.join(self.original_path, 'product2_front.jpg')

        self.assertTrue(os.path.exists(image1), f"이미지 파일이 없습니다: {image1}")
        self.assertTrue(os.path.exists(image2), f"이미지 파일이 없습니다: {image2}")

        # verbose=True로 상세 정보 출력
        similarity, scores = self.analyzer.calculate_similarity(image1, image2, verbose=False)

        # Stage 0 결과 확인
        self.assertIn('stage0_quality', scores)
        stage0_quality = scores['stage0_quality']

        self.assertIn('image1', stage0_quality)
        self.assertIn('image2', stage0_quality)

        print(f"\n✓ Stage 0이 포함된 최종 유사도: {similarity:.4f}")
        print(f"  - Pipeline Status:")
        for status in scores['pipeline_status']:
            print(f"    - {status}")


if __name__ == '__main__':
    unittest.main()
