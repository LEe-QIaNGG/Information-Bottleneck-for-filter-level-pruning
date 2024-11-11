import torch
from sklearn.cluster import KMeans
from scipy.stats import entropy
import numpy as np
from net import SimpleCNN

# 实例化模型
model = SimpleCNN()
model.eval()

# 假设输入数据 X
X = torch.randn(10, 1, 28, 28)  # 10张单通道28x28图像

# 钩子函数来获取中间层的激活值
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 注册钩子
model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))

# 计算激活值
output = model(X)

# 提取 conv1 和 conv2 的特征图
conv1_activations = activations['conv1']
conv2_activations = activations['conv2']

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

# 计算 conv1 和 conv2 中每个滤波器的 I(X; T) 和 I(T; Y)
I_XT_conv1, I_TY_conv1 = calculate_filter_mutual_information(X, conv1_activations, output)

print("I(X; T) for each filter in conv1:", I_XT_conv1)
print("I(T; Y) for each filter in conv1:", I_TY_conv1)