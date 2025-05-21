import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
#CNN模型
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load('digit_cnn.pth', map_location=device, weights_only=True))

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(),  # 确保图像是灰度图
    transforms.Resize((28, 28)),  # 调整到28x28
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def mnist_style_preprocess(image_path):
    # 灰度
    image = Image.open(image_path).convert('L')
    # 增强对比度
    image = ImageEnhance.Contrast(image).enhance(2.0)  # 数值可调，2.0~3.0更明显

    # 二值化
    image_np = np.array(image)
    threshold = 100  # 阈值调低一点，黑的更黑
    image_np = (image_np > threshold) * 255  # 黑底白字
    image_bin = Image.fromarray(np.uint8(image_np))

    # 找到非白色区域（即数字）的边界
    coords = np.column_stack(np.where(image_np < 255))
    if coords.size == 0:
        # 没有数字，直接返回空白28x28
        return Image.new('L', (28, 28), 0)  # 黑底

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 裁剪出数字区域
    cropped = image_bin.crop((x_min, y_min, x_max + 1, y_max + 1))

    # 缩放到26x26
    digit = cropped.resize((20, 20), Image.Resampling.LANCZOS)
    digit = ImageOps.invert(digit)
    #新建28x28黑底图，把数字粘贴到中心
    new_img = Image.new('L', (28, 28), 0)  # 黑底
    upper_left = ((28 - 20) // 2, (28 - 20) // 2)
    new_img.paste(digit, upper_left)
    
    return new_img

# 7. 测试单张图片
def predict_image(image_path, model_path='digit_cnn.pth'):
    # 加载模型
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = mnist_style_preprocess(image_path)
    image = transform(image).unsqueeze(0).to(device)  # 添加batch维度

    # 预测
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()
        
    return prediction