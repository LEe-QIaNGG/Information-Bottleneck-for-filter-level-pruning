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
from ray import tune
from ray.tune import CLIReporter
from functools import partial

def train_model(config):
    beta = config['beta']
    num_prune = config['num_prune']
    prune_interval = config['prune_interval']
    num_epochs = config['num_epochs']
    is_prune = config['is_prune']
    prune_layer1 = config['prune_layer1'] # 是否剪枝第一层
    prune_layer2 = config['prune_layer2'] # 是否剪枝第二层

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
                if prune_layer1:
                    I_XT1, I_TY1 = calculate_filter_mutual_information(X_mi, activations['conv1'], output_mi)
                    information_plane_1 = np.array(I_XT1) - beta * np.array(I_TY1)
                    top_n_indices_1 = np.argsort(information_plane_1)[-num_prune:][::-1]
                    top_n_values_1 = information_plane_1[top_n_indices_1]
                    # 对第一层进行剪枝
                    for idx in top_n_indices_1:
                        model.conv1.weight.data[idx] = 0
                        if model.conv1.bias is not None:
                            model.conv1.bias.data[idx] = 0

                if prune_layer2:
                    I_XT2, I_TY2 = calculate_filter_mutual_information(X_mi, activations['conv2'], output_mi)
                    information_plane_2 = np.array(I_XT2) - beta * np.array(I_TY2)
                    top_n_indices_2 = np.argsort(information_plane_2)[-num_prune*2:][::-1]
                    top_n_values_2 = information_plane_2[top_n_indices_2]
                    # 对第二层进行剪枝
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
        # print(f'Epoch {epoch+1} 测试集准确率: {accuracy:.2f}%')
        # print(f'Epoch {epoch+1} 测试集推理速度: {avg_time:.2f} ms')
    # 在训练结束后调用函数
    # plot_training_curves(epoch_losses, test_accuracies, test_times, logdir, pruning_epochs,is_prune)

    return np.mean(metrics)

def tune_hyperparameters(num_samples=10):
    config = {
        "beta": tune.uniform(0.5, 3.0),
        "num_prune": tune.choice([2, 4, 6, 8]),
        "prune_interval": tune.choice([3, 5, 7]),
        "num_epochs": 50,
        "is_prune": True,
        "prune_layer1": tune.choice([True, False]),
        "prune_layer2": lambda spec: not spec.config.prune_layer1  # 确保与prune_layer1相反
    }
    
    reporter = CLIReporter(
        parameter_columns=["beta", "num_prune", "prune_interval", "prune_layer1", "prune_layer2"],
        metric_columns=["mean_metric"]
    )

    analysis = tune.run(
        train_model,
        config=config,
        num_samples=num_samples,
        progress_reporter=reporter,
        resources_per_trial={"cpu": 2, "gpu": 0.5},  # 根据您的硬件调整
        metric="mean_metric",
        mode="max"
    )
    
    best_config = analysis.get_best_config(metric="mean_metric", mode="max")
    print("最佳配置:", best_config)
    return best_config

if __name__ == '__main__':
    best_config = tune_hyperparameters(num_samples=10)
    # 使用最佳配置运行最终模型
    final_metric = train_model(best_config)
    print(f"使用最佳配置的最终指标: {final_metric}")