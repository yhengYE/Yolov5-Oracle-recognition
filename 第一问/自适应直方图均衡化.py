# Applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
# to the original image for contrast enhancement
import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
# 读取图像
original_image = cv2.imread('data/h02060.jpg', 0)  # 以灰度模式读取图像

equalized_median = cv2.equalizeHist(original_image)

# Create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_equalized = clahe.apply(original_image)

# Display the results using matplotlib
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('原始图像', fontproperties=font)
plt.axis('off')

# CLAHE equalized image
plt.subplot(1, 2, 2)
plt.imshow(clahe_equalized, cmap='gray')
plt.title('自适应直方图均衡化', fontproperties=font)
plt.axis('off')

# Show the plot
plt.show()
