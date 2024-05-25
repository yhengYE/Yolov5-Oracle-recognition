import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import os
import torch.nn as nn
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

import matplotlib
# 设置matplotlib配置，使用支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统可以使用SimHei字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 定义数据变换，确保和训练时一致
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载类别
full_dataset = ImageFolder(root='D:\develop\pylearn\mathorcup/4_Recognize/训练集', transform=transform)

# 加载模型
model = models.resnet34(pretrained=False)  # False to not use pretrained weights initially
num_classes = 76  # 设置正确的类别数
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('resnet50.pth'))
model.eval()  # 设置模型为评估模式

# 设定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 文件夹路径
folder_path = '分割'

# 选择用于可视化的图片数量
num_images_to_display = 5
images_displayed = 0

# 读取图片并进行预测
for image_name in os.listdir(folder_path):
    if image_name.endswith('.jpg') and images_displayed < num_images_to_display:  # 确保处理.jpg文件
        image_path = os.path.join(folder_path, image_name)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
        image_tensor = image_tensor.to(device)

        # 预测
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = full_dataset.classes[predicted.item()]

        # 可视化
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title(f'Predicted class: {predicted_class}')
        plt.show()

        images_displayed += 1
