import cv2
import numpy as np

def process_image(input_image):
    """Full enhancement pipeline"""
    # Step 1: Denoise
    denoised = cv2.fastNlMeansDenoising(input_image, h=10)

    # Step 2: Gamma correction (brighten)
    table = np.array([(i / 255.0) ** 0.5 * 255 for i in range(256)], dtype=np.uint8)
    brightened = cv2.LUT(denoised, table)

    # Step 3: Histogram equalization (contrast)
    equalized = cv2.equalizeHist(brightened)

    # Step 4: Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(equalized, -1, kernel)

    return sharpened
