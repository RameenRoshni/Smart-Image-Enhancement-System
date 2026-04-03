import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_image(image_path):
    # Load RGB and Grayscale
    img_rgb = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    print("===== LAB 01: Image Acquisition =====")
    print(f"Resolution     : {img_gray.shape[1]} x {img_gray.shape[0]}")
    print(f"Data Type      : {img_gray.dtype}")
    print(f"Pixel Matrix (top-left 5x5):\n{img_gray[:5, :5]}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_rgb);      axes[0].set_title("RGB Image");       axes[0].axis('off')
    axes[1].imshow(img_gray, cmap='gray'); axes[1].set_title("Grayscale"); axes[1].axis('off')
    plt.suptitle("Lab 01 - Image Acquisition")
    plt.tight_layout()
    plt.savefig("images/output/lab01_output.png")
    plt.show()

    return img_gray
