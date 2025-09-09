import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000) # 生成数据集

def load_array(data_arrays, batch_size, is_train=True):  
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays) # 这里的 * 号表示把 data_arrays 这个 tuple 解包
    # 这个函数的作用是把 features 和 labels 组合在一起，形成一个数据集
    # TensorDataset 是 PyTorch 提供的一个数据处理类
    # 它的输入是一些张量，这些张量的第一个维度必须相同
    # 它会把这些张量按行组合在一起，形成一个数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # shuffle表示是否打乱数据
# DataLoader 是 PyTorch 提供的一个非常强大的数据加载工具
# 它可以帮你自动处理数据的索引、打乱、批次切分等操作

### 2. **功能封装**
# 高级API帮你自动处理了：
# - ✅ **索引管理**：不需要手动生成和管理索引
# - ✅ **数据打乱**：通过 `shuffle=is_train` 参数控制
# - ✅ **批次切分**：自动处理最后一个不完整批次
# - ✅ **内存优化**：内置高效的数据加载机制

# ### 3. **更强的灵活性**
# ```python
# # 支持多个数组同时处理
# features, labels, weights = load_data()
# loader = load_array((features, labels, weights), batch_size=32)

# # 训练时打乱，验证时不打乱
# train_loader = load_array((X_train, y_train), 32, is_train=True)
# val_loader = load_array((X_val, y_val), 32, is_train=False)
# ```

# ### 4. **性能优化**
# - **多线程加载**：DataLoader支持 `num_workers` 参数
# - **预取机制**：可以在GPU计算时预加载下一批数据
# - **内存固定**：支持 `pin_memory=True` 加速CPU到GPU的数据传输

# ### 5. **生态系统集成**
# ```python
# # 与PyTorch训练循环完美集成
# for batch_features, batch_labels in train_loader:
#     # 直接使用，无需额外处理
#     predictions = model(batch_features)
#     # ...
# ```

# ## 关键技术细节

# **解包操作 `*data_arrays`**：
# ```python
# # 假设 data_arrays = (features, labels)
# # *data_arrays 等价于 features, labels
# dataset = data.TensorDataset(features, labels)
# ```

# **一个实际的对比**：
# ````python
# # 使用你的原生实现
# for X_batch, y_batch in data_iter(32, features, labels):
#     # 需要确保数据类型正确
#     pass

# # 使用高级API
# train_loader = load_array((features, labels), 32)
# for X_batch, y_batch in train_loader:
#     # 自动处理类型转换和设备管理
#     pass
# ````

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter)) # 取出第一个小批量

from torch import nn

net = nn.Sequential(nn.Linear(2, 1)) # 2个输入特征，1个输出
# Sequential 是一个容器，里面可以包含多个神经网络层
# 它会按照添加的顺序依次执行这些层

# # 方法1：直接传入层作为参数
# net = nn.Sequential(
#     nn.Linear(2, 4),    # 输入2个特征，输出4个特征
#     nn.ReLU(),          # 激活函数
#     nn.Linear(4, 1)     # 输入4个特征，输出1个特征
# )

# # 方法2：使用有序字典，可以给每层命名
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#     ('linear1', nn.Linear(2, 4)),
#     ('relu1', nn.ReLU()),
#     ('linear2', nn.Linear(4, 1))
# ]))
# ````

# ## 使用示例

# ````python
# # 创建模型
# model = nn.Sequential(
#     nn.Linear(784, 128),   # 全连接层：784 -> 128
#     nn.ReLU(),             # ReLU激活函数
#     nn.Dropout(0.2),       # Dropout层，防止过拟合
#     nn.Linear(128, 64),    # 全连接层：128 -> 64
#     nn.ReLU(),             # ReLU激活函数
#     nn.Linear(64, 10)      # 输出层：64 -> 10（10个类别）
# )

# # 前向传播
# x = torch.randn(32, 784)  # 批次大小32，输入特征784
# output = model(x)         # 输出形状：(32, 10)
# print(output.shape)
# ````

# ## 主要特点

# 1. **顺序执行**：数据会按照添加层的顺序依次通过每一层
# 2. **简洁明了**：适合构建简单的前馈神经网络
# 3. **自动连接**：自动处理层与层之间的连接

# ## 适用场景

# - ✅ 简单的前馈网络
# - ✅ 卷积神经网络的主干部分
# - ❌ 复杂的网络结构（如跳跃连接、分支结构）

# 对于更复杂的网络架构，建议继承 `nn.Module` 类来自定义模型。