import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net import SimpleCNN
from utils import calculate_filter_mutual_information, plot_layer_mutual_information,evaluate,plot_training_curves
import matplotlib.pyplot as plt
import numpy as np
import time

def train_model(config):
    beta = config['beta']
    num_prune = config['num_prune']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设置数据加载和预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载CIFAR-10数据集
    cifar_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

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
    num_epochs = 12

    # 用于记录损失和准确率
    train_losses = []
    test_accuracies = []
    epoch_losses = []
    test_times = []
    metrics = []
    logdir = './results'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_losses = []     
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_losses.append(loss.item())
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                    f'Loss: {running_loss/200:.4f}')
                running_loss = 0.0
        
        # 记录每个epoch的平均损失
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
        
        model.eval()
        if epoch % 3 == 0:
            with torch.no_grad():
                # 获取一批数据用于计算互信息
                X_mi, _ = next(iter(test_loader))
                X_mi = X_mi.to(device)
                output_mi = model(X_mi)
                # 计算每层的互信息
                I_XT, I_TY = calculate_filter_mutual_information(X_mi, activations['conv1'], output_mi)
                information_plane = np.array(I_XT) - beta * np.array(I_TY)
                plot_layer_mutual_information(information_plane, 'Conv1', epoch, logdir)
                # 找出information plane最大的n个值及其对应的filter索引
                top_n_indices = np.argsort(information_plane)[-num_prune:][::-1]  # 获取最大的n个索引
                top_n_values = information_plane[top_n_indices]  # 获取对应的值
                
                # 打印最大的n个值及其索引
                for i, (idx, val) in enumerate(zip(top_n_indices, top_n_values)):
                    print(f'Epoch {epoch}: 第{i+1}大信息值 {val:.4f}, 对应filter索引 {idx}')
                
                # 对这n个filter进行剪枝 - 将权重和偏置设为0
                with torch.no_grad():
                    for idx in top_n_indices:
                        model.conv1.weight.data[idx] = 0
                        if model.conv1.bias is not None:
                            model.conv1.bias.data[idx] = 0
                
                # 用红色虚线标注剪枝时刻
                plt.axvline(x=epoch, color='red', linestyle='--', label='剪枝时刻')
        
        # 在测试集上评估模型
        accuracy, avg_time = evaluate(model, test_loader, device)
        test_accuracies.append(accuracy)
        test_times.append(avg_time)
        if epoch % 3 == 0:
            metrics.append(accuracy/ avg_time)
        print(f'Epoch {epoch+1} 测试集准确率: {accuracy:.2f}%')
    # 在训练结束后调用函数
    plot_training_curves(epoch_losses, test_accuracies, test_times, num_epochs, logdir)
    
