import cv2
import matplotlib.pyplot as plt
import os

from functions.image_basics   import analyze_image
from functions.sampling     import sampling_quantization
from functions.transformations    import geometric_transformations
from functions.intensity     import intensity_transformations
from functions.histogram     import histogram_processing
from functions.pipeline      import process_image

# Setup
os.makedirs("images/output", exist_ok=True)
IMAGE_PATH = "images/input/sample.jpg"

# Run all labs
img_gray = analyze_image(IMAGE_PATH)
sampling_quantization(img_gray)
geometric_transformations(img_gray)
intensity_transformations(img_gray)
histogram_processing(img_gray)

# Final pipeline
enhanced = process_image(img_gray)
cv2.imwrite("images/output/final_enhanced.png", enhanced)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(img_gray, cmap='gray'); ax[0].set_title("Original");  ax[0].axis('off')
ax[1].imshow(enhanced, cmap='gray'); ax[1].set_title("Enhanced");  ax[1].axis('off')
plt.suptitle("Lab 06 - Final Enhanced Output")
plt.savefig("images/output/lab06_final.png")
plt.show()

print("\n✅ All tasks complete! Check images/output/ for results.")
