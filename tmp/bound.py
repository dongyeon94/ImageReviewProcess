import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

print(os.listdir())

for i in os.listdir():
    if "jpg" in i or "png" in i:
        image = cv2.imread(i)
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

        print("-------------------")
        print("현재 파일 이름 : " + i)
        print(f"엣지 픽셀 비율(Canny): {edge_ratio:.4f}")
        print(f"Sobel 엣지 평균 강도: {sobel_mean_strength:.2f}")
        print(f"Laplacian 엣지 평균 강도: {laplacian_strength:.2f}")
        print("-------------------")

        # Visualization of intermediate results
        plt.figure(figsize=(15, 8))

        # Original image (convert BGR to RGB for correct color display)
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        # Grayscale image
        plt.subplot(2, 3, 2)
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')

        # Canny edges
        plt.subplot(2, 3, 3)
        plt.imshow(edges, cmap='gray')
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
        plt.title(f"{i} - Sobel Edge Magnitude Distribution")
        plt.xlabel("Gradient Magnitude")
        plt.ylabel("Pixel Count")
        plt.tight_layout()
        plt.show()