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



# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(),  # 确保图像是灰度图
    transforms.Resize((28, 28)),  # 调整到28x28
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def mnist_style_preprocess(image):
    # 灰度
    # image = Image.open(image_path).convert('L')
    if isinstance(image, str):
        # 如果是路径字符串，打开图片
        image = Image.open(image).convert('L')
    elif isinstance(image, Image.Image):
        # 如果是PIL Image对象，转换为灰度
        image = image.convert('L')
    else:
        raise TypeError("输入必须是图片路径或PIL Image对象")
    
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

# 测试单张图片
def predict_image(image_path, model_path='digit_cnn.pth',model=None,device=None):
    # 加载模型
    if model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def detect_digits_line(image_path, model_path='digit_cnn.pth'):
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 读取图像
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = image_path
        
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 按x坐标排序轮廓
    digit_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 过滤掉太小的区域
        if w > 20 and h > 20:
            digit_regions.append((x, y, w, h))
    
    # 按x坐标排序
    digit_regions.sort(key=lambda x: x[0])
    
    # 识别每个数字
    results = []
    for x, y, w, h in digit_regions:
        # 提取数字区域
        digit_roi = img[y:y+h, x:x+w]
        # 转换为PIL Image
        digit_pil = Image.fromarray(cv2.cvtColor(digit_roi, cv2.COLOR_BGR2RGB))
        # 预测
        prediction = predict_image(digit_pil, model_path=model_path, model=model, device=device)
        results.append(prediction)
        # 在原图上画框和显示结果
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, str(prediction), (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return results


# 使用示例
# #摄像头识别（受环境影响过大）
# def capture_and_recognize():
#     # 初始化摄像头
#     cap = cv2.VideoCapture(1)
#     # 加载模型
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = DigitCNN().to(device)
#     model.load_state_dict(torch.load('digit_cnn.pth', map_location=device))
#     model.eval()

#     # 添加帧率控制和结果显示变量
#     frame_count = 0
#     skip_frames = 15  # 每15帧进行一次识别
#     last_prediction = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("无法获取摄像头画面")
#             break

#         # 在画面中央绘制一个方框
#         height, width = frame.shape[:2]
#         box_size = min(width, height) // 2
#         x1 = (width - box_size) // 2
#         y1 = (height - box_size) // 2
#         x2 = x1 + box_size
#         y2 = y1 + box_size
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         # 每隔一定帧数进行识别
#         if frame_count % skip_frames == 0:
#             # 截取方框内的图像
#             roi = frame[y1:y2, x1:x2]
#             # 转换为PIL Image
#             roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
#             # 预处理
#             processed_image = mnist_style_preprocess(roi_pil)
#             # 转换为tensor并预测
#             image_tensor = transform(processed_image).unsqueeze(0).to(device)
            
#             with torch.no_grad():
#                 output = model(image_tensor)
#                 _, predicted = torch.max(output, 1)
#                 last_prediction = predicted.item()

#         # 在画面上显示识别结果
#         if last_prediction is not None:
#             cv2.putText(frame, f"Prediction: {last_prediction}", 
#                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
#                        1, (0, 255, 0), 2)

#         # 显示画面
#         cv2.imshow('Digit Recognition', frame)

#         # 检查是否按下q键退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         frame_count += 1

#     cap.release()
#     cv2.destroyAllWindows()

# capture_and_recognize()
