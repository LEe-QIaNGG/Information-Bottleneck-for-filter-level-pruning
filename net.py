import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设你有一个CNN模型和数据
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 13 * 13, 10)  # 直接连接到输出层

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.adaptive_avg_pool2d(x, (13, 13))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class DoubleConvCNN(nn.Module):
    def __init__(self):
        super(DoubleConvCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3)
        self.fc1 = nn.Linear(48 * 13 * 13, 10)  # 更新为正确的特征数量

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.adaptive_avg_pool2d(x, (15, 15))  # 更新为正确的池化输出
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x