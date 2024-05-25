import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import models
import torch.nn as nn
import torch.optim as optim

# 设置数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),                      # 调整图像大小
    transforms.Grayscale(num_output_channels=3),        # 将图像从灰度转换为三通道
    transforms.ToTensor(),                              # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # 归一化
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
full_dataset = ImageFolder(root='甲骨文智能识别中原始拓片单字自动分割与识别研究/4_Recognize/训练集', transform=transform)

# 分割数据集为训练集和测试集
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 获取类别数
num_classes = len(full_dataset.classes)

# 加载预训练的ResNet-34模型
model = models.resnet34(pretrained=True)
# 修改最后的全连接层以匹配类别数
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 将模型移动到GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

best_loss = float('inf')
best_model_state = None

# 训练模型
for epoch in range(2):  # 假设仍然只进行3个epoch
    model.train()  # 设置模型为训练模式
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # 清零梯度缓存
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 每50个batch输出一次
        if (i + 1) % 50 == 0:
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}')

        # 检查是否有最佳损失更新
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            # print(f'New best model saved with loss: {best_loss} at epoch {epoch+1}, batch {i+1}')

# 训练结束后保存表现最佳的模型状态
if best_model_state:
    torch.save(best_model_state, 'best_model.pth')
    print(f"Best model saved with loss: {best_loss}")

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total}%')

# 保存模型的最终状态（如果需要）
torch.save(model.state_dict(), 'model.pth')
