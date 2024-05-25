import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import ast
import os

# 路径设置
img_dir_path = 'D:\develop\pylearn\mathorcup/3_Test/Figures'
image_name = '020111.jpg'  # Added missing quotes and corrected variable name
img_path = os.path.join(img_dir_path, image_name)

# 加载图片
img = Image.open(img_path)  # Corrected capitalization of "Image"

# 假设已经有了标记数据，这里使用硬编码的示例
marks = "[[204.0, 67.0, 239.0, 118.0, 1.0], [209.0, 127.0, 233.0, 175.0, 1.0], [90.0, 339.0, 137.0, 392.0, 1.0], [191.0, 183.0, 231.0, 297.0, 1.0], [180.0, 306.0, 230.0, 397.0, 1.0]]"  # Corrected list format and added missing bracket

# 解析标记数据
marks_list = ast.literal_eval(marks)  # Corrected variable name and added missing underscore

# 创建绘图
fig, ax = plt.subplots()
ax.imshow(img)

# 绘制每个边界框
for mark in marks_list:  # Corrected variable name and added missing underscore
    # 标记数据中已经是具体的像素坐标
    # 格式:[x_min,y_min,x_max,y_max,1.0]
    rect_x = mark[0]  # Corrected variable names and added missing underscore
    rect_y = mark[1]
    rect_width = mark[2] - mark[0]  # Corrected variable names and added missing underscore
    rect_height = mark[3] - mark[1]  # Corrected variable names and added missing underscore
    print("绘制矩形:", rect_x, rect_y, rect_width, rect_height)  # 打印坐标和大小
    # 绘制矩形
    rect = patches.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=1, edgecolor='r', facecolor='none')  # Corrected variable names and added missing underscore
    ax.add_patch(rect)  # Corrected method name and added missing underscore

# 显示图像
plt.show()
