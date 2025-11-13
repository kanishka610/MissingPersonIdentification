import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('n.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib display

# 1. Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (15, 15), 0)

# 2. Median Filter
median_blur = cv2.medianBlur(image, 15)

# 3. Bilateral Filter
bilateral_filter = cv2.bilateralFilter(image, 15, 75, 75)

# 4. Non-Local Means Denoising (Grayscale)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
non_local_means = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)

# Display the results
titles = ['Original', 'Gaussian Blur', 'Median Blur', 'Bilateral Filter', 'Non-Local Means']
images = [image_rgb, gaussian_blur, median_blur, bilateral_filter, np.uint8(non_local_means)]

plt.figure(figsize=(15, 8))

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    if i == 4:  # Non-Local Means image is grayscale
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
