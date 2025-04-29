
!pip install opencv-python-headless scipy scikit-image scikit-learn

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.filters import gabor
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import files

# === Upload Two Eye Images ===
print("Upload the first eye image:")
uploaded1 = files.upload()
img1_path = next(iter(uploaded1))

print("Upload the second eye image:")
uploaded2 = files.upload()
img2_path = next(iter(uploaded2))

# === Image Preprocessing ===
def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (320, 240))  # Resize for standardization
    return img

# === Segment Iris Using Hough Transform ===
def segment_iris(img):
    blurred = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=20, maxRadius=60)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :1]:
            x, y, r = circle
            iris = img[y - r:y + r, x - r:x + r]
            return iris
    return None

# === Normalize Iris ===
def normalize_iris(iris_img):
    if iris_img is None:
        return None
    iris_resized = cv2.resize(iris_img, (64, 512))
    return iris_resized

# === Feature Extraction using Gabor Filter ===
def extract_features(normalized_img):
    real, imag = gabor(normalized_img, frequency=0.6)
    feature_vector = real.flatten()
    return feature_vector.reshape(1, -1)

# === Full Pipeline for Matching ===
def iris_pipeline(img_path):
    img = preprocess(img_path)
    iris = segment_iris(img)
    norm = normalize_iris(iris)
    if norm is not None:
        feat = extract_features(norm)
        return img, iris, norm, feat
    return None, None, None, None

# === Process Both Images ===
img1, iris1, norm1, feat1 = iris_pipeline(img1_path)
img2, iris2, norm2, feat2 = iris_pipeline(img2_path)

# === Match and Display ===
# === Match and Display ===
if feat1 is not None and feat2 is not None:
    similarity = cosine_similarity(feat1, feat2)[0][0]
    threshold = 0.90  # You can adjust this based on testing
    match_result = "✅ Match Found" if similarity >= threshold else "❌ No Match"

    print(f"\n🔍 Similarity Score: {similarity:.4f}")
    print(f"📌 Result: {match_result}")

    # Visualization
    fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    axs[0, 0].imshow(img1, cmap='gray')
    axs[0, 0].set_title("Eye Image 1")
    axs[0, 1].imshow(iris1, cmap='gray')
    axs[0, 1].set_title("Iris 1")
    axs[0, 2].imshow(norm1, cmap='gray')
    axs[0, 2].set_title("Normalized 1")

    axs[1, 0].imshow(img2, cmap='gray')
    axs[1, 0].set_title("Eye Image 2")
    axs[1, 1].imshow(iris2, cmap='gray')
    axs[1, 1].set_title("Iris 2")
    axs[1, 2].imshow(norm2, cmap='gray')
    axs[1, 2].set_title("Normalized 2")

    plt.suptitle(f"{match_result} | 🔗 Similarity Score: {similarity:.4f}", fontsize=16)
    plt.tight_layout()
    plt.show()
else:
    print("❌ Iris could not be detected in one or both images. Try with clearer eye photos.")


import matplotlib.pyplot as plt
from skimage.filters import gabor

# Assuming you want to use the normalized iris from the first image
# Change 'norm1' to 'norm2' if you want to use the second image's iris
norm_iris = norm1

# Apply Gabor filter to normalized iris
gabor_real, gabor_imag = gabor(norm_iris, frequency=0.6)

# Resize Gabor image for better visualization (scale up)
gabor_resized = cv2.resize(gabor_real, (512, 256), interpolation=cv2.INTER_CUBIC)
norm_resized = cv2.resize(norm_iris, (512, 256), interpolation=cv2.INTER_CUBIC)

# Show the normalized iris and Gabor-filtered image
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(norm_resized, cmap='gray')
plt.title("Normalized Iris (Resized)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gabor_resized, cmap='gray')
plt.title("Gabor Filtered Iris (Enhanced View)")
plt.axis('off')

plt.tight_layout()
plt.show()