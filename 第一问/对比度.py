import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
# 读取图像
original_image = cv2.imread('data/h02060.jpg', 0)  # 以灰度模式读取图像

equalized_median = cv2.equalizeHist(original_image)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('原始图像', fontproperties=font)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_median, cmap='gray')
plt.title('直方图均衡化',fontproperties=font)
plt.axis('off')

# 显示图像
plt.show()
