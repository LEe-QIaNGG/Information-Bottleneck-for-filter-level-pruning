import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net import DoubleConvCNN, SimpleCNN
from utils import calculate_filter_mutual_information, plot_layer_mutual_information,evaluate,plot_training_curves
import matplotlib.pyplot as plt
import numpy as np
import time
import os

def train_model(config):
    beta = config['beta']
    num_prune = config['num_prune']
    prune_interval = 5
    num_epochs = 50
    is_prune = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设置数据加载和预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载CIFAR-10数据集
    cifar_train = datasets.CIFAR10('./data', train=True, download=False, transform=transform)
    cifar_test = datasets.CIFAR10('./data', train=False, download=False, transform=transform)

    # 创建数据加载器
    batch_size = 512
    train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(cifar_test, batch_size=batch_size)

    # 实例化模型
    model = DoubleConvCNN().to(device)
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
    model.conv2.register_forward_hook(get_activation('conv2'))

    # 用于记录损失和准确率
    train_losses = []
    test_accuracies = []
    epoch_losses = []
    test_times = []
    metrics = []
    pruning_epochs = []
    logdir = './results/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if not os.path.exists(logdir):
        os.makedirs(logdir)

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
            
            if batch_idx % 300 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                    f'Loss: {running_loss/300:.4f}')
                running_loss = 0.0
        
        # 记录每个epoch的平均损失
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
        
        model.eval()
        if epoch % prune_interval == 0 and is_prune:
            with torch.no_grad():
                X_mi, _ = next(iter(test_loader))
                X_mi = X_mi.to(device)
                output_mi = model(X_mi)
                # 计算每层的互信息
                I_XT1, I_TY1 = calculate_filter_mutual_information(X_mi, activations['conv1'], output_mi)
                I_XT2, I_TY2 = calculate_filter_mutual_information(X_mi, activations['conv2'], output_mi)
                information_plane_1 = np.array(I_XT1) - beta * np.array(I_TY1)
                information_plane_2 = np.array(I_XT2) - beta * np.array(I_TY2)
                # plot_layer_mutual_information(information_plane, 'Conv1', epoch, logdir)
                top_n_indices_1 = np.argsort(information_plane_1)[-num_prune:][::-1]  # 获取最大的n个索引
                top_n_values_1 = information_plane_1[top_n_indices_1]  # 获取对应的值
                top_n_indices_2 = np.argsort(information_plane_2)[-num_prune*2:][::-1]  # 获取最大的n个索引
                top_n_values_2 = information_plane_2[top_n_indices_2]  # 获取对应的值
                
                # 打印最大的n个值及其索引
                # for i, (idx, val) in enumerate(zip(top_n_indices_1, top_n_values_1)):
                #     print(f'Epoch {epoch}: 第{i+1}大信息值 {val:.4f}, 对应filter索引 {idx}')
                # for i, (idx, val) in enumerate(zip(top_n_indices_2, top_n_values_2)):
                #     print(f'Epoch {epoch}: 第{i+1}大信息值 {val:.4f}, 对应filter索引 {idx}')
                
                # 对这n个filter进行剪枝 - 将权重和偏置设为0
                for idx in top_n_indices_1:
                    model.conv1.weight.data[idx] = 0
                    if model.conv1.bias is not None:
                        model.conv1.bias.data[idx] = 0
                for idx in top_n_indices_2:
                    model.conv2.weight.data[idx] = 0
                    if model.conv2.bias is not None:
                        model.conv2.bias.data[idx] = 0
                pruning_epochs.append(epoch)
        
        # 在测试集上评估模型
        accuracy, avg_time = evaluate(model, test_loader, device)
        test_accuracies.append(accuracy)
        test_times.append(avg_time)
        if epoch % prune_interval == 0:
            metrics.append(accuracy/ avg_time)
        print(f'Epoch {epoch+1} 测试集准确率: {accuracy:.2f}%')
        print(f'Epoch {epoch+1} 测试集推理速度: {avg_time:.2f} ms')
    # 在训练结束后调用函数
    plot_training_curves(epoch_losses, test_accuracies, test_times, logdir, pruning_epochs,is_prune)
    
if __name__ == '__main__':
    config = {
        'beta': 2,
        'num_prune': 1
    }
    train_model(config)