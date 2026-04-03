import cv2
import matplotlib.pyplot as plt
import numpy as np

def sampling_quantization(img):
    print("\n===== LAB 02: Sampling & Quantization =====")
    scales = [0.25, 0.5, 1.0, 1.5, 2.0]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    # --- Sampling ---
    for i, scale in enumerate(scales):
        h, w = img.shape
        resized = cv2.resize(img, (int(w*scale), int(h*scale)))
        resized_back = cv2.resize(resized, (w, h))  # restore for comparison
        axes[0][i].imshow(resized_back, cmap='gray')
        axes[0][i].set_title(f"Scale {scale}x\n{resized.shape[1]}x{resized.shape[0]}")
        axes[0][i].axis('off')
        print(f"Scale {scale}x → Resolution: {resized.shape[1]}x{resized.shape[0]}")

    # --- Quantization (bit depth) ---
    bits = [8, 4, 2, 1]
    for i, bit in enumerate(bits):
        levels = 2 ** bit
        quantized = (img // (256 // levels)) * (256 // levels)
        axes[1][i].imshow(quantized, cmap='gray')
        axes[1][i].set_title(f"{bit}-bit ({levels} levels)")
        axes[1][i].axis('off')
    axes[1][4].axis('off')

    plt.suptitle("Lab 02 - Sampling & Quantization")
    plt.tight_layout()
    plt.savefig("images/output/lab02_output.png")
    plt.show()
