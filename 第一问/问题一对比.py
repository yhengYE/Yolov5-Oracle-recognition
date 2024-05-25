import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置支持中文的字体
font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=14)  # 请根据你的系统路径选择合适的字体文件

# 加载图像
img_path = 'data/h02060.jpg'
original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

# 应用双边滤波进行降噪
bilateral_filtered = cv2.bilateralFilter(original_image, 9, 75, 75)

# 创建CLAHE对象（对比度限制的自适应直方图均衡化）
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 在双边滤波后应用CLAHE
clahe_on_bilateral = clahe.apply(bilateral_filtered)

# 绘制和显示处理后的图像
plt.figure(figsize=(12, 6))

# 原始图像
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('原始图像', fontproperties=font)
plt.axis('off')

# 双边滤波后应用CLAHE的图像
plt.subplot(1, 2, 2)
plt.imshow(clahe_on_bilateral, cmap='gray')
plt.title('双边滤波后应用CLAHE', fontproperties=font)
plt.axis('off')

plt.tight_layout()
plt.show()
