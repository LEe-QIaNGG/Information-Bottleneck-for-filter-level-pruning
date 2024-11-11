import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net import SimpleCNN
from utils import calculate_filter_mutual_information, plot_layer_mutual_information,evaluate

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载CIFAR-10数据集
cifar_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10('./data', train=False, transform=transform)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size=batch_size)

# 实例化模型
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 钩子函数来获取中间层的激活值
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 注册钩子
model.conv1.register_forward_hook(get_activation('conv1'))

# 训练循环
num_epochs = 11

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # 每个epoch开始时的互信息计算
    model.eval()
    if epoch%2==0:
        with torch.no_grad():
            # 获取一批数据用于计算互信息
            X_mi, _ = next(iter(test_loader))
            X_mi = X_mi.to(device)
            output_mi = model(X_mi)
            
            # 计算每层的互信息
            I_XT_conv1, I_TY_conv1 = calculate_filter_mutual_information(X_mi, activations['conv1'], output_mi)
                
            # 绘制每层的互信息直方图
            plot_layer_mutual_information(I_XT_conv1, I_TY_conv1, 'Conv1',epoch)
    # 训练过程
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {running_loss/100:.4f}')
            running_loss = 0.0
    
    # 在测试集上评估模型
    accuracy = evaluate(model, test_loader, device)
    print(f'Epoch {epoch+1} 测试集准确率: {accuracy:.2f}%')
