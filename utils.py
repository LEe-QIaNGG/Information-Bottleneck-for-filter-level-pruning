import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
import matplotlib.pyplot as plt
import torch
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_time = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 记录推理开始时间
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            
            outputs = model(data)
            
            # 记录推理结束时间
            end_time.record()
            torch.cuda.synchronize()
            total_time += start_time.elapsed_time(end_time)
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = 100 * correct / total
    avg_time = total_time / len(test_loader)  # 计算平均推理时间
    return accuracy, avg_time

def plot_layer_mutual_information(information_plane, layer_name,epoch,logdir):
    plt.figure(figsize=(6, 4))
    # 计算第50%分位数
    median = np.median(information_plane)
    plt.hist(information_plane, bins=20, alpha=0.7, label='I(X;T) - beta*I(T;Y)')
    # 添加红色虚线标记中位数
    plt.axvline(x=median, color='red', linestyle='--', label='50% threshold')
    plt.title(f'{layer_name} layer information bottleneck')
    plt.xlabel('I(X;T) - beta*I(T;Y)')
    plt.ylabel('frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{logdir}/{layer_name}_information_bottleneck_{epoch}.png')
    plt.close()

def plot_training_curves(epoch_losses, test_accuracies, test_times, logdir, pruning_epochs=[],is_prune=False):
    num_epochs = len(epoch_losses)
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 1, 1)
    plt.plot(range(1, num_epochs + 1), epoch_losses, 'b-')
    plt.title('training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 在loss曲线上标注剪枝时刻
    if is_prune:
        for epoch in pruning_epochs:
            plt.axvline(x=epoch, color='red', linestyle='--', label='prune' if epoch == pruning_epochs[0] else "")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{logdir}/training_loss.png')
    plt.close()

    # 绘制准确率和归一化推理速度曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 1, 1)
    plt.plot(range(1, num_epochs + 1), test_accuracies, 'r-', label='accuracy')
    normalized_times = 1 / np.array(test_times)  # 计算推理时间的倒数
    normalized_times = normalized_times / np.max(normalized_times)  # 归一化
    normalized_times=np.sqrt(normalized_times)
    plt.plot(range(1, num_epochs + 1), normalized_times*100, 'g-', label='normalized inference speed')
    plt.title('test accuracy and normalized inference speed')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%) / Normalized Inference Speed')
    
    # 标注剪枝时刻
    if is_prune:
        for epoch in pruning_epochs:
            plt.axvline(x=epoch, color='red', linestyle='--', label='prune' if epoch == pruning_epochs[0] else "")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{logdir}/accuracy_and_inference_speed.png')
    plt.close()

# 计算每个滤波器的互信息
def calculate_filter_mutual_information(input_data, layer_activations, output_data, n_bins=20):
    num_filters = layer_activations.shape[1]
    I_XT = []
    I_TY = []
    
    for i in range(num_filters):
        filter_activations = layer_activations[:, i, :, :]
        
        # 计算 I(X; T_i) 和 I(T_i; Y) for 第 i 个滤波器
        I_XT_filter = mutual_information(input_data, filter_activations, n_bins=n_bins)
        I_TY_filter = mutual_information(filter_activations, output_data, n_bins=n_bins)
        
        I_XT.append(I_XT_filter)
        I_TY.append(I_TY_filter)
    
    return I_XT, I_TY

# 定义一个函数来计算互信息
def mutual_information(x, y, n_bins=20):
    # 将输入数据展平，并确保没有梯度信息
    x = x.detach().reshape(-1).cpu().numpy()  # 使用 reshape 替代 view
    y = y.detach().reshape(-1).cpu().numpy()  # 使用 reshape 替代 view
    
    # 确保 x 和 y 具有相同的长度
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    # 使用 KMeans 进行量化
    x = KMeans(n_clusters=n_bins).fit_predict(x.reshape(-1, 1))
    y = KMeans(n_clusters=n_bins).fit_predict(y.reshape(-1, 1))
    
    # 计算联合直方图
    p_xy, _, _ = np.histogram2d(x, y, bins=n_bins, density=True)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    # 计算互信息 I(X;Y) = H(X) + H(Y) - H(X,Y)
    H_x = entropy(p_x)
    H_y = entropy(p_y)
    H_xy = entropy(p_xy.flatten())
    return H_x + H_y - H_xy


