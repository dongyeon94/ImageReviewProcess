"""Image Similarity Analysis - Multi-Stage Pipeline

다단계 이미지 유사도 분석 파이프라인:
1단계: pHash를 이용한 완전히 다른 이미지 필터링
2단계: Edge Detection + Shape Matching으로 윤곽선 비교
3단계: LBP + Gabor Filter로 질감/패턴 비교
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
from typing import Tuple, Dict
from scipy.spatial import distance
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
from PIL import Image
import imagehash


class MultiStagePipeline:
    """다단계 이미지 유사도 분석 파이프라인"""

    def __init__(self, use_gpu: bool = True):
        """
        파이프라인 초기화

        Args:
            use_gpu: GPU 사용 여부
        """
        self.device = torch.device(
            'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        )

        # 임계값 설정
        self.phash_threshold = 15  # pHash 해밍 거리 (0-64)
        self.shape_threshold = 0.3  # Shape Matching 임계값
        self.texture_threshold = 0.4  # Texture 유사도 임계값

        print("[INFO] 다단계 파이프라인 초기화 완료")

    # ==================== 0단계: 이미지 품질 검증 ====================

    def _calculate_laplacian_variance(self, image_path: str) -> float:
        """
        Laplacian 분산으로 이미지의 선명도 계산
        값이 낮을수록 블러/노이즈가 많은 상태

        Args:
            image_path: 이미지 경로

        Returns:
            Laplacian 분산 값
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        return laplacian_var

    def _calculate_edge_density(self, image_path: str) -> float:
        """
        이미지의 엣지 밀도 계산
        경계선이 많은 정도를 0-1 사이의 값으로 반환

        Args:
            image_path: 이미지 경로

        Returns:
            엣지 밀도 (0-1)
        """
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)

        # 엣지 픽셀 비율 계산
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = edge_pixels / total_pixels

        return edge_density

    def _detect_single_color(self, image_path: str, threshold: int = 50) -> Tuple[bool, float]:
        """
        이미지가 한 가지 색으로 지배되는지 감지
        색상 다양성을 엔트로피로 판단

        Args:
            image_path: 이미지 경로
            threshold: 히스토그램 피크 임계값 (색상이 이 비율 이상 차지하면 단색)

        Returns:
            (단색 여부, 색상 다양성 점수)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        # BGR을 HSV로 변환 (색상 분석에 더 적합)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # H(Hue) 채널 히스토그램 계산
        hist = cv2.calcHist([hsv], [0], None, [36], [0, 180])
        hist = hist.astype(float) / hist.sum()

        # 가장 높은 히스토그램 값 (dominant color ratio)
        dominant_color_ratio = np.max(hist)

        # 색상 엔트로피 계산 (다양성 지표)
        hist_safe = hist[hist > 0]
        color_entropy = -np.sum(hist_safe * np.log2(hist_safe + 1e-10))

        # 정규화된 엔트로피 (0-1)
        max_entropy = np.log2(len(hist))
        normalized_entropy = color_entropy / max_entropy if max_entropy > 0 else 0

        # 단색 판정: 주요 색상이 threshold% 이상 차지
        is_single_color = dominant_color_ratio * 100 > threshold

        return is_single_color, normalized_entropy

    def stage0_image_quality_validation(self, image_path: str) -> Tuple[bool, Dict]:
        """
        0단계: 이미지 품질 검증 (사전 필터링)
        - 이미지 선명도 검증
        - 경계선/엣지 존재 여부 검증
        - 단색 이미지 감지

        Args:
            image_path: 이미지 경로

        Returns:
            (통과 여부, 상세 분석 딕셔너리)
        """
        try:
            quality_scores = {
                'laplacian_variance': None,
                'edge_density': None,
                'is_single_color': None,
                'color_diversity': None,
                'pass_reasons': [],
                'fail_reasons': []
            }

            # 1. 이미지 선명도 검증
            laplacian_var = self._calculate_laplacian_variance(image_path)
            quality_scores['laplacian_variance'] = laplacian_var
            sharpness_threshold = 100  # 임계값 (하한)

            if laplacian_var < sharpness_threshold:
                quality_scores['fail_reasons'].append(
                    f"선명도 부족 (Laplacian: {laplacian_var:.2f} < {sharpness_threshold})"
                )
            else:
                quality_scores['pass_reasons'].append(
                    f"선명도 양호 (Laplacian: {laplacian_var:.2f})"
                )

            # 2. 경계선/엣지 밀도 검증
            edge_density = self._calculate_edge_density(image_path)
            quality_scores['edge_density'] = edge_density
            edge_threshold = 0.05  # 최소 5% 이상의 엣지 필요

            if edge_density < edge_threshold:
                quality_scores['fail_reasons'].append(
                    f"경계선 부족 (엣지 밀도: {edge_density:.4f} < {edge_threshold})"
                )
            else:
                quality_scores['pass_reasons'].append(
                    f"경계선 양호 (엣지 밀도: {edge_density:.4f})"
                )

            # 3. 단색 이미지 감지
            is_single_color, color_diversity = self._detect_single_color(image_path)
            quality_scores['is_single_color'] = is_single_color
            quality_scores['color_diversity'] = color_diversity
            color_threshold = 0.2  # 최소 색상 다양성

            if is_single_color or color_diversity < color_threshold:
                quality_scores['fail_reasons'].append(
                    f"색상 정보 부족 (단색: {is_single_color}, 엔트로피: {color_diversity:.4f})"
                )
            else:
                quality_scores['pass_reasons'].append(
                    f"색상 정보 충분 (엔트로피: {color_diversity:.4f})"
                )

            # 최종 판정: 3개 조건 모두 통과해야 함
            passed = len(quality_scores['fail_reasons']) == 0

            return passed, quality_scores

        except Exception as e:
            raise Exception(f"이미지 품질 검증 중 오류 발생: {str(e)}")

    # ==================== 1단계: pHash 기반 필터링 ====================

    def _calculate_phash(self, image_path: str, hash_size: int = 8) -> str:
        """
        pHash (Perceptual Hash) 계산

        Args:
            image_path: 이미지 경로
            hash_size: 해시 크기

        Returns:
            pHash 값
        """
        img = Image.open(image_path)
        return str(imagehash.phash(img, hash_size=hash_size))

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        두 해시 사이의 해밍 거리 계산

        Args:
            hash1: 첫 번째 해시
            hash2: 두 번째 해시

        Returns:
            해밍 거리
        """
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def stage1_phash_filtering(self, image1_path: str, image2_path: str) -> Tuple[bool, float]:
        """
        1단계: pHash를 이용한 완전히 다른 이미지 필터링

        Args:
            image1_path: 첫 번째 이미지 경로
            image2_path: 두 번째 이미지 경로

        Returns:
            (통과 여부, 유사도 점수)
        """
        hash1 = self._calculate_phash(image1_path)
        hash2 = self._calculate_phash(image2_path)

        hamming_dist = self._hamming_distance(hash1, hash2)

        # 해밍 거리 정규화 (0-64 범위를 0-1로)
        normalized_distance = hamming_dist / 64.0
        phash_similarity = 1.0 - normalized_distance

        # 임계값 이상이면 통과
        passed = hamming_dist <= self.phash_threshold

        return passed, phash_similarity

    # ==================== 2단계: Edge Detection + Shape Matching ====================

    def _extract_edges(self, image_path: str) -> np.ndarray:
        """
        Canny Edge Detection으로 윤곽선 추출

        Args:
            image_path: 이미지 경로

        Returns:
            엣지 맵
        """
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gaussian Blur로 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny Edge Detection
        edges = cv2.Canny(blurred, 100, 200)

        return edges

    def _match_shapes(self, edges1: np.ndarray, edges2: np.ndarray) -> float:
        """
        두 엣지 맵의 유사도 계산 (교집합 / 합집합)

        Args:
            edges1: 첫 번째 엣지 맵
            edges2: 두 번째 엣지 맵

        Returns:
            Shape 유사도 (0-1)
        """
        # 동일 크기로 리사이징
        h, w = edges1.shape
        edges2_resized = cv2.resize(edges2, (w, h))

        # 교집합과 합집합 계산
        intersection = cv2.bitwise_and(edges1, edges2_resized)
        union = cv2.bitwise_or(edges1, edges2_resized)

        intersection_count = np.count_nonzero(intersection)
        union_count = np.count_nonzero(union)

        if union_count == 0:
            return 1.0

        # Jaccard 유사도 (IoU)
        iou = intersection_count / union_count

        return iou

    def stage2_shape_matching(self, image1_path: str, image2_path: str) -> Tuple[bool, float]:
        """
        2단계: Edge Detection + Shape Matching으로 윤곽선 비교

        Args:
            image1_path: 첫 번째 이미지 경로
            image2_path: 두 번째 이미지 경로

        Returns:
            (통과 여부, 유사도 점수)
        """
        edges1 = self._extract_edges(image1_path)
        edges2 = self._extract_edges(image2_path)

        shape_similarity = self._match_shapes(edges1, edges2)

        # 임계값 이상이면 통과
        passed = shape_similarity >= self.shape_threshold

        return passed, shape_similarity

    # ==================== 3단계: LBP + Gabor Filter 기반 질감 분석 ====================

    def _extract_lbp_features(self, image_path: str, radius: int = 3, n_points: int = 8) -> np.ndarray:
        """
        LBP (Local Binary Pattern) 특징 추출

        Args:
            image_path: 이미지 경로
            radius: LBP 반경
            n_points: 샘플 포인트 수

        Returns:
            LBP 히스토그램
        """
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))

        # LBP 계산
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

        # 히스토그램 계산
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_points + 2,
            range=(0, n_points + 2)
        )

        # 정규화
        hist = hist.astype(float) / hist.sum()

        return hist

    def _create_gabor_filters(self, kernel_size: int = 21) -> list:
        """
        Gabor Filter 생성

        Args:
            kernel_size: 필터 커널 크기

        Returns:
            Gabor 필터 리스트
        """
        filters = []
        ksize = (kernel_size, kernel_size)
        sigma = kernel_size / 5
        theta_values = np.arange(0, np.pi, np.pi / 4)  # 4개 방향
        lambda_values = [kernel_size / 4, kernel_size / 2]  # 2개 주파수

        for theta in theta_values:
            for lambda_ in lambda_values:
                kernel = cv2.getGaborKernel(
                    ksize=ksize,
                    sigma=sigma,
                    theta=theta,
                    lambd=lambda_,
                    gamma=0.5,
                    psi=0
                )
                filters.append(kernel)

        return filters

    def _extract_gabor_features(self, image_path: str) -> np.ndarray:
        """
        Gabor Filter를 이용한 질감 특징 추출

        Args:
            image_path: 이미지 경로

        Returns:
            Gabor 응답 벡터
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))

        filters = self._create_gabor_filters()

        gabor_responses = []
        for filer in filters:
            response = cv2.filter2D(img, cv2.CV_32F, filer)
            # 응답의 통계적 특징 추출 (평균, 표준편차)
            gabor_responses.append(response.mean())
            gabor_responses.append(response.std())

        return np.array(gabor_responses)

    def _compare_texture_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        두 히스토그램 비교 (Chi-square)

        Args:
            hist1: 첫 번째 히스토그램
            hist2: 두 번째 히스토그램

        Returns:
            유사도 (0-1)
        """
        # Chi-square 거리 계산
        chi_square = 0.5 * np.sum(
            (hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-8)
        )

        # 거리를 유사도로 변환 (exponential decay)
        texture_similarity = np.exp(-chi_square)

        return texture_similarity

    def _compare_gabor_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        두 Gabor 특징 비교 (코사인 유사도)

        Args:
            features1: 첫 번째 Gabor 특징
            features2: 두 번째 Gabor 특징

        Returns:
            유사도 (0-1)
        """
        # 코사인 유사도
        similarity = distance.cosine(features1, features2)

        # 거리를 유사도로 변환
        gabor_similarity = 1.0 - similarity

        return max(0, gabor_similarity)

    def stage3_texture_analysis(self, image1_path: str, image2_path: str) -> Tuple[bool, Dict]:
        """
        3단계: LBP + Gabor Filter로 질감/패턴 비교

        Args:
            image1_path: 첫 번째 이미지 경로
            image2_path: 두 번째 이미지 경로

        Returns:
            (통과 여부, 상세 점수 딕셔너리)
        """
        # LBP 특징 추출 및 비교
        lbp_hist1 = self._extract_lbp_features(image1_path)
        lbp_hist2 = self._extract_lbp_features(image2_path)
        lbp_similarity = self._compare_texture_histograms(lbp_hist1, lbp_hist2)

        # Gabor Filter 특징 추출 및 비교
        gabor_features1 = self._extract_gabor_features(image1_path)
        gabor_features2 = self._extract_gabor_features(image2_path)
        gabor_similarity = self._compare_gabor_features(gabor_features1, gabor_features2)

        # 질감 유사도 (가중 평균)
        texture_similarity = 0.5 * lbp_similarity + 0.5 * gabor_similarity

        scores = {
            'lbp': lbp_similarity,
            'gabor': gabor_similarity,
            'texture': texture_similarity
        }

        passed = texture_similarity >= self.texture_threshold

        return passed, scores

    # ==================== 최종 파이프라인 ====================

    def calculate_similarity(self, image1_path: str, image2_path: str,
                           verbose: bool = False) -> Tuple[float, Dict]:
        """
        다단계 파이프라인을 이용한 최종 유사도 계산

        Args:
            image1_path: 첫 번째 이미지 경로
            image2_path: 두 번째 이미지 경로
            verbose: 상세 정보 출력 여부

        Returns:
            (최종 유사도, 상세 점수 딕셔너리)
        """
        try:
            if not os.path.exists(image1_path) or not os.path.exists(image2_path):
                raise FileNotFoundError("이미지 파일을 찾을 수 없습니다")

            scores = {
                'stage0_quality': None,
                'stage1_phash': None,
                'stage2_shape': None,
                'stage3_texture': None,
                'final_similarity': None,
                'pipeline_status': []
            }

            # ========== 0단계: 이미지 품질 검증 ==========
            stage0_passed_img1, quality_scores_img1 = self.stage0_image_quality_validation(image1_path)
            stage0_passed_img2, quality_scores_img2 = self.stage0_image_quality_validation(image2_path)

            scores['stage0_quality'] = {
                'image1': quality_scores_img1,
                'image2': quality_scores_img2
            }

            # 둘 다 통과해야 진행
            stage0_passed = stage0_passed_img1 and stage0_passed_img2

            scores['pipeline_status'].append(
                f"0단계 (품질검증): Image1 - {stage0_passed_img1}, Image2 - {stage0_passed_img2}"
            )

            if verbose:
                print(f"\n[0단계 - 이미지 품질 검증]")
                print(f"  Image1 통과: {stage0_passed_img1}")
                if quality_scores_img1['pass_reasons']:
                    for reason in quality_scores_img1['pass_reasons']:
                        print(f"    ✓ {reason}")
                if quality_scores_img1['fail_reasons']:
                    for reason in quality_scores_img1['fail_reasons']:
                        print(f"    ✗ {reason}")
                print(f"  Image2 통과: {stage0_passed_img2}")
                if quality_scores_img2['pass_reasons']:
                    for reason in quality_scores_img2['pass_reasons']:
                        print(f"    ✓ {reason}")
                if quality_scores_img2['fail_reasons']:
                    for reason in quality_scores_img2['fail_reasons']:
                        print(f"    ✗ {reason}")
                print("--------")
                print()


            # print(f"  최종 유사도: {final_similarity:.4f}")
            print(f"  판정: ✗ 비교 불가능, {stage0_passed}")
            print('------------')
            print('------------')
            print('------------')

            # Stage 0 실패 시 조기 종료
            # if not stage0_passed:
            #     final_similarity = 0.0
            #     scores['final_similarity'] = final_similarity

            #     if verbose:
            #         print(f"\n[파이프라인 중단]")
            #         print(f"  사유: 이미지 품질 검증 실패")
            #         print(f"  최종 유사도: {final_similarity:.4f}")
            #         print(f"  판정: ✗ 비교 불가능")

            #     return final_similarity, scores

            # ========== 1단계: pHash 필터링 ==========
            stage1_passed, phash_sim = self.stage1_phash_filtering(image1_path, image2_path)
            scores['stage1_phash'] = phash_sim
            scores['pipeline_status'].append(
                f"1단계 (pHash): 유사도: {phash_sim:.4f}, 단계 : {stage1_passed}"
            )

            if verbose:
                print(f"\n[1단계 - pHash 필터링]")
                print(f"  유사도: {phash_sim:.4f}")
                print(f"  임계값: {self.phash_threshold} (거리), 단계 : {stage1_passed}")
                print("--------")
                print(scores)
                print("--------")
                print()
                print()
                print()

            # ========== 2단계: Shape Matching ==========
            stage2_passed, shape_sim = self.stage2_shape_matching(image1_path, image2_path)
            scores['stage2_shape'] = shape_sim
            scores['pipeline_status'].append(
                f"2단계 (Shape): 유사도: {shape_sim:.4f}, 단계 : {stage2_passed}"
            )

            if verbose:
                print(f"\n[2단계 - Edge Detection + Shape Matching]")
                print(f"  유사도: {shape_sim:.4f}")
                print(f"  임계값: {self.shape_threshold}, 단계 : {stage2_passed}")
                print("--------")
                print(scores)
                print("--------")
                print()
                print()
                print()

            # ========== 3단계: Texture Analysis ==========
            stage3_passed, texture_scores = self.stage3_texture_analysis(image1_path, image2_path)
            scores['stage3_texture'] = texture_scores['texture']
            scores['pipeline_status'].append(
                f"3단계 (Texture): LBP: {texture_scores['lbp']:.4f}, Gabor: {texture_scores['gabor']:.4f}, 통합: {texture_scores['texture']:.4f}, 단계: {stage3_passed}"
            )

            if verbose:
                print(f"\n[3단계 - LBP + Gabor Filter 질감 분석]")
                print(f"  LBP 유사도: {texture_scores['lbp']:.4f}")
                print(f"  Gabor 유사도: {texture_scores['gabor']:.4f}")
                print(f"  통합 유사도: {texture_scores['texture']:.4f}")
                print(f"  임계값: {self.texture_threshold}, 단계: {stage3_passed}")
                print("--------")
                print(scores)
                print("--------")
                print()
                print()
                print()
                

            # ========== 최종 유사도 계산 ==========
            # 가중 평균 (패스/페일 상관없이 항상 계산)
            final_similarity = (
                phash_sim * 0.2 +
                shape_sim * 0.3 +
                texture_scores['texture'] * 0.5
            )

            final_similarity = np.clip(final_similarity, 0.0, 1.0)
            scores['final_similarity'] = final_similarity

            if verbose:
                print(f"\n[최종 결과]")
                print(f"  가중치: pHash(0.2) × {phash_sim:.4f} + Shape(0.3) × {shape_sim:.4f} + Texture(0.5) × {texture_scores['texture']:.4f}")
                print(f"  최종 유사도: {final_similarity:.4f}")
                print(f"  판정: {'✓ 같은 상품' if final_similarity > 0.6 else '✗ 다른 상품'}")

            return final_similarity, scores

        except Exception as e:
            raise Exception(f"이미지 비교 중 오류 발생: {str(e)}")


# 하위 호환성을 위한 래퍼 클래스
class ImageSimilarityAnalyzer(MultiStagePipeline):
    """기존 API와의 호환성을 위한 래퍼"""

    def __init__(self, use_gpu: bool = True, verbose: bool = True):
        """
        초기화

        Args:
            use_gpu: GPU 사용 여부
            verbose: 상세 정보 출력 여부
        """
        super().__init__(use_gpu=use_gpu)
        self.verbose = verbose

    def calculate_similarity(self, image1_path: str, image2_path: str, verbose: bool = None) -> float:
        """
        기존 API와의 호환성을 위해 float만 반환

        Args:
            image1_path: 첫 번째 이미지 경로
            image2_path: 두 번째 이미지 경로
            verbose: 상세 정보 출력 여부 (None이면 초기화 시 설정값 사용)

        Returns:
            유사도 점수 (0-1)
        """
        verbose_flag = verbose if verbose is not None else self.verbose
        similarity, _ = super().calculate_similarity(image1_path, image2_path, verbose=verbose_flag)
        return similarity
