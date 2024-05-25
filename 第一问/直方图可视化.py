import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# 设置支持中文的字体
font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)  # 请根据你的系统路径选择合适的字体文件

# 加载图像
img_path = 'data/h02060.jpg'
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

# 创建CLAHE对象（对比度限制的自适应直方图均衡化）
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 应用CLAHE到原始图像
clahe_image = clahe.apply(image)

# 计算并绘制直方图
plt.figure(figsize=(12, 6))

# 设置背景颜色
plt.rcParams['axes.facecolor'] = '#f8f8f8'
plt.rcParams['savefig.facecolor'] = '#f8f8f8'

# 原始图像及其直方图
ax1 = plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('原始图像', fontproperties=font)
plt.axis('off')

ax2 = plt.subplot(2, 2, 2)
plt.hist(image.ravel(), 256, [0, 256], color='navy', alpha=0.75)
plt.title('原始图像直方图', fontproperties=font)
plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线并设置样式

# CLAHE处理后的图像及其直方图
ax3 = plt.subplot(2, 2, 3)
plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE图像', fontproperties=font)
plt.axis('off')

ax4 = plt.subplot(2, 2, 4)
plt.hist(clahe_image.ravel(), 256, [0, 256], color='crimson', alpha=0.75)
plt.title('CLAHE图像直方图', fontproperties=font)
plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线

plt.tight_layout(pad=3.0)  # 设置图形间的空白
plt.show()
