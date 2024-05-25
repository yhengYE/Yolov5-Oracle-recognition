import cv2
from PIL import Image
import os

# 图片和标签的文件夹路径
images_folder = 'E:/money/mathercup/cesi'
labels_folder = 'yolov5-master/runs/detect/exp8/labels'
output_folder = '分割'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历图片文件夹中的所有文件
for image_filename in os.listdir(images_folder):
    if image_filename.endswith('.jpg') or image_filename.endswith('.png'):  # 支持jpg和png图片
        image_path = os.path.join(images_folder, image_filename)
        label_filename = image_filename.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(labels_folder, label_filename)
        
        # 检查对应的标签文件是否存在
        if os.path.exists(label_path):
            # 加载图片
            image = Image.open(image_path)
            img_width, img_height = image.size

            # 读取YOLO格式的标签文件
            with open(label_path, 'r') as file:
                lines = file.readlines()

            # 对于标签中的每个对象
            for i, line in enumerate(lines):
                parts = line.strip().split()
                x_center, y_center, width, height = map(float, parts[1:5])

                # YOLO格式转换为像素坐标
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height

                # 计算边界框的左上角和右下角坐标
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)

                # 裁剪图片
                cropped_image = image.crop((x_min, y_min, x_max, y_max))

                # 保存裁剪的图片
                cropped_image.save(os.path.join(output_folder, f'object_{i+1}_{image_filename}'))

print("所有检测到的对象已被裁剪并保存.")
