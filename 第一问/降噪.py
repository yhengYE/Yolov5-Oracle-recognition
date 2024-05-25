from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  # 假设你是在Windows操作系统上
# Load the image from the file
img_path = 'data/w01870.jpg'
original_image = cv2.imread(img_path, 0)  # Load as grayscale

# Apply median filter
median_filtered = cv2.medianBlur(original_image, 5)

# Apply bilateral filter
bilateral_filtered = cv2.bilateralFilter(original_image, 9, 75, 75)

# Display the original and filtered images for comparison
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('原始图像', fontproperties=font)
plt.axis('off')

# Median filtered image
plt.subplot(1, 3, 2)
plt.imshow(median_filtered, cmap='gray')
plt.title('中值滤波', fontproperties=font)
plt.axis('off')

# Bilateral filtered image
plt.subplot(1, 3, 3)
plt.imshow(bilateral_filtered, cmap='gray')
plt.title('双边滤波', fontproperties=font)
plt.axis('off')

# Show the plot
plt.show()
