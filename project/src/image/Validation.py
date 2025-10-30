import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn



class ImageValidation:
    """ 이미지 유사도 판별 파이프 라인"""

    def __init__(self, use_gpu: bool = True):
        self.device = torch.device(
            'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        )
        

    def _EdgeDetection(self, imagePath: str, plot: bool = False):
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Canny Edge
        edges = cv2.Canny(gray, 100, 200)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_ratio = edge_pixels / total_pixels

        # 2. Sobel Edge (gradient magnitude)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_mean_strength = np.mean(sobel_magnitude)

        # 3. Laplacian Edge (second derivative)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_strength = np.mean(np.abs(laplacian))

        # 4. Edge index 
        edge_index = (
            (min(edge_ratio / 0.25, 1.0)) * 0.4 + 
            (min(sobel_mean_strength / 100, 1.0)) * 0.3 + 
            (min(laplacian_strength / 80, 1.0)) * 0.3            
        )
        edge_index = min(edge_index, 1.0)

        print("-------------------")
        print("현재 파일 이름 : " + imagePath)
        print(f"엣지 픽셀 비율(Canny): {edge_ratio:.4f}")
        print(f"Sobel 엣지 평균 강도: {sobel_mean_strength:.2f}")
        print(f"Laplacian 엣지 평균 강도: {laplacian_strength:.2f}")
        print(f"경계면 지수 : {edge_index:.2f} ")
        print("-------------------")

        if plot:
             self.imagePlot(image, gray, edges, sobel_magnitude, laplacian)

        if edge_index < 0.3:
            return False
        else:
            return True

    def imagePlot(self, image, grayImage, edgesImage, sobel_magnitude, laplacian):
        # Visualization of intermediate results
        plt.figure(figsize=(15, 8))

        # Original image (convert BGR to RGB for correct color display)
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        # Grayscale image
        plt.subplot(2, 3, 2)
        plt.imshow(grayImage, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')

        # Canny edges
        plt.subplot(2, 3, 3)
        plt.imshow(edgesImage, cmap='gray')
        plt.title("Canny Edges")
        plt.axis('off')

        # Sobel magnitude normalized to 0-255
        sobel_norm = np.uint8(255 * (sobel_magnitude / np.max(sobel_magnitude))) if np.max(sobel_magnitude) > 0 else np.uint8(sobel_magnitude)
        plt.subplot(2, 3, 4)
        plt.imshow(sobel_norm, cmap='gray')
        plt.title("Sobel Magnitude")
        plt.axis('off')

        # Laplacian absolute value normalized to 0-255
        laplacian_abs = np.abs(laplacian)
        laplacian_norm = np.uint8(255 * (laplacian_abs / np.max(laplacian_abs))) if np.max(laplacian_abs) > 0 else np.uint8(laplacian_abs)
        plt.subplot(2, 3, 5)
        plt.imshow(laplacian_norm, cmap='gray')
        plt.title("Laplacian Abs Value")
        plt.axis('off')

        # Empty subplot for layout symmetry
        plt.subplot(2, 3, 6)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # 4. 엣지 강도 분포 히스토그램 (Sobel 기반)
        plt.figure(figsize=(6, 3))
        plt.hist(sobel_magnitude.ravel(), bins=50, color='gray')
        plt.title(f"{image} - Sobel Edge Magnitude Distribution")
        plt.xlabel("Gradient Magnitude")
        plt.ylabel("Pixel Count")
        plt.tight_layout()
        plt.show()
