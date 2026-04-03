import cv2
import numpy as np
import matplotlib.pyplot as plt

def intensity_transformations(img):
    print("\n===== LAB 04: Intensity Transformations =====")

    # Negative
    negative = 255 - img

    # Log transformation
    img_float = np.float32(img) + 1
    c = 255 / np.log(1 + np.max(img_float))
    log_img = np.uint8(c * np.log(img_float))

    # Gamma corrections
    def gamma_correction(image, gamma):
        table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image, table)

    gamma_05 = gamma_correction(img, 0.5)
    gamma_15 = gamma_correction(img, 1.5)

    titles = ["Original", "Negative", "Log", "Gamma=0.5", "Gamma=1.5"]
    images = [img, negative, log_img, gamma_05, gamma_15]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle("Lab 04 - Intensity Transformations")
    plt.tight_layout()
    plt.savefig("images/output/lab04_output.png")
    plt.show()

    print("Best for Brightening      → Gamma = 0.5")
    print("Best for Highlighting Details → Log Transformation")

    return gamma_05  # best enhanced version
