import cv2
import matplotlib.pyplot as plt

def histogram_processing(img):
    print("\n===== LAB 05: Histogram Processing =====")

    equalized = cv2.equalizeHist(img)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0][0].imshow(img, cmap='gray');       axes[0][0].set_title("Original Image");    axes[0][0].axis('off')
    axes[0][1].imshow(equalized, cmap='gray'); axes[0][1].set_title("Equalized Image");   axes[0][1].axis('off')
    axes[1][0].hist(img.ravel(), 256, [0, 256], color='blue', alpha=0.7)
    axes[1][0].set_title("Original Histogram")
    axes[1][1].hist(equalized.ravel(), 256, [0, 256], color='green', alpha=0.7)
    axes[1][1].set_title("Equalized Histogram")

    plt.suptitle("Lab 05 - Histogram Processing")
    plt.tight_layout()
    plt.savefig("images/output/lab05_output.png")
    plt.show()

    return equalized
