import cv2
import numpy as np
import matplotlib.pyplot as plt

def geometric_transformations(img):
    print("\n===== LAB 03: Geometric Transformations =====")
    h, w = img.shape
    cx, cy = w // 2, h // 2

    angles = [30, 45, 60, 90, 120, 150, 180]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Rotations
    for i, angle in enumerate(angles):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        axes[i // 4][i % 4].imshow(rotated, cmap='gray')
        axes[i // 4][i % 4].set_title(f"Rotation {angle}°")
        axes[i // 4][i % 4].axis('off')

    # Translation
    M_trans = np.float32([[1, 0, 50], [0, 1, 30]])
    translated = cv2.warpAffine(img, M_trans, (w, h))
    axes[1][3].imshow(translated, cmap='gray')
    axes[1][3].set_title("Translation (50,30)")
    axes[1][3].axis('off')

    plt.suptitle("Lab 03 - Geometric Transformations")
    plt.tight_layout()
    plt.savefig("images/output/lab03_output.png")
    plt.show()

    # Inverse (restore 90° rotation)
    M90 = cv2.getRotationMatrix2D((cx, cy), 90, 1.0)
    rotated90 = cv2.warpAffine(img, M90, (w, h))
    M_inv = cv2.getRotationMatrix2D((cx, cy), -90, 1.0)
    restored = cv2.warpAffine(rotated90, M_inv, (w, h))

    fig2, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img, cmap='gray');        ax[0].set_title("Original");    ax[0].axis('off')
    ax[1].imshow(rotated90, cmap='gray');  ax[1].set_title("Rotated 90°"); ax[1].axis('off')
    ax[2].imshow(restored, cmap='gray');   ax[2].set_title("Restored");    ax[2].axis('off')
    plt.suptitle("Lab 03 - Inverse Transformation")
    plt.savefig("images/output/lab03_inverse.png")
    plt.show()
