import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 使用 这本书的工具包 d2l 来加载数据集
# 它完成了数据的预处理（包括归一化和格式转换），分割训练集和测试集
# 并创建了 PyTorch 的数据迭代器

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    # 通过 keepdim=True 保持 sum 后的张量维度不变：比如不加本来是[2,3], 通过keepdim后变为[[2],[3]]
    return X_exp / partition  # 广播：使得分母的形状与 X_exp 相同
# 这样算出的是X_exp的每一行除以该行的和

y = torch.tensor([0, 2]) # 2个样本的标签
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]) # 2个样本的预测概率

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1 :  # len(y_hat.shape)> 1 说明有多列
        # 如果y_hat的列数大于1，我们就取每一行的最大值作为预测类别
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y # 为什么要这样写：因为y_hat的类型可能是float，而y的类型是int
    return float(cmp.type(y.dtype).sum()) # cmp是一个0-1张量，表示每个预测是否正确 比如 [1,0,1,1,0]，把它转换成和y一样的类型，然后求和

# 训练softmax回归模型

class Accumulator:
    """
    一个用于在多个变量上进行累加操作的工具类。
    该类可以维护n个数值变量，支持批量累加、重置和索引访问操作。
    常用于机器学习中累计损失值、准确率等多个指标的统计。
    Attributes:
        data (list): 存储累加值的列表
    Example:
        >>> acc = Accumulator(3)  # 创建3个累加器
        >>> acc.add(1, 2, 3)      # 累加 [1, 2, 3]
        >>> acc.add(4, 5, 6)      # 再累加 [4, 5, 6]
        >>> print(acc[0])         # 输出: 5.0 (1+4)
        >>> acc.reset()           # 重置所有累加器为0
        初始化累加器。
        Args:
            n (int): 需要累加的变量个数
        对所有变量进行累加操作。
        Args:
            *args: 要累加的数值，数量应与初始化时的n相同
        重置所有累加器的值为0.0。
        通过索引获取指定位置的累加值。
        Args:
            idx (int): 要获取的累加器索引
        Returns:
            float: 指定位置的累加值
            在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用自定义优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])  # 调用自定义的sgd函数
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
# ### 情况1: PyTorch内置优化器对象
# ```python
# # 创建优化器对象
# optimizer = torch.optim.SGD([W, b], lr=0.1)
# train_ch3(..., optimizer)  # updater = optimizer

# # 此时：
# isinstance(optimizer, torch.optim.Optimizer)  # 返回 True
# # 因为 torch.optim.SGD 继承自 torch.optim.Optimizer
# ```

# ### 情况2: 自定义函数（你的情况）
# ```python
# # 定义函数
# def sgd(batch_size):
#     global W, b
#     with torch.no_grad():
#         W -= lr * W.grad / batch_size
#         b -= lr * b.grad / batch_size
#         W.grad.zero_()
#         b.grad.zero_()

# train_ch3(..., sgd)  # updater = sgd

# # 此时：
# isinstance(sgd, torch.optim.Optimizer)  # 返回 False
# # 因为 sgd 是一个函数，不是 Optimizer 类的实例
# ```

# ## 💡 为什么这样设计？
# ```python
# # 不管 updater 是什么类型，只要它能"更新参数"就行

# if isinstance(updater, torch.optim.Optimizer):
#     # 如果是优化器对象，用对象的方法
#     updater.zero_grad()  # 清零梯度
#     updater.step()       # 更新参数
# else:
#     # 如果是函数，直接调用函数
#     updater(batch_size)  # 调用 sgd(batch_size)
# ```

# ## 🎯 类型检查示例

# ```python
# import torch

# # 示例1: 优化器对象
# W = torch.randn(2, 3, requires_grad=True)
# optimizer = torch.optim.SGD([W], lr=0.1)
# print(isinstance(optimizer, torch.optim.Optimizer))  # True
# print(type(optimizer))  # <class 'torch.optim.sgd.SGD'>

# # 示例2: 函数
# def my_sgd(batch_size):
#     pass

# print(isinstance(my_sgd, torch.optim.Optimizer))  # False
# print(type(my_sgd))  # <class 'function'>

# # 示例3: 继承关系
# print(isinstance(optimizer, torch.optim.SGD))      # True
# print(isinstance(optimizer, torch.optim.Optimizer)) # True (继承关系)
# ```

# ## ✅ 总结

# - **`isinstance()`**: 检查对象是否属于指定类型
# - **`updater`**: 参数更新器，可以是优化器对象或函数，**不是迭代器**
# - **作用**: 让同一个训练函数支持两种不同的参数更新方式
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型"""
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:.3f}, '
              f'train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}')

# 定义模型参数
num_inputs = 784  # 28*28的图像展平后的特征数
num_outputs = 10  # 10个类别（0-9）

# 初始化模型参数
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 定义模型
def net(X):
    """softmax回归模型"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 定义损失函数
def cross_entropy_loss(y_hat, y):
    """交叉熵损失函数"""
    return cross_entropy(y_hat, y).mean()

# 定义优化器（手动实现SGD）
lr = 0.1  # 学习率

def sgd(batch_size):
    """小批量随机梯度下降"""
    global W, b  # 声明使用全局变量
    with torch.no_grad():
        W -= lr * W.grad / batch_size
        b -= lr * b.grad / batch_size
        W.grad.zero_()
        b.grad.zero_()

# 实际工作中常用的PyTorch优化器示例

# # 1. SGD优化器（随机梯度下降）
# optimizer_sgd = torch.optim.SGD([W, b], lr=0.1, momentum=0.9, weight_decay=1e-4)

# # 2. Adam优化器（最常用，自适应学习率）
# optimizer_adam = torch.optim.Adam([W, b], lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

# # 3. AdamW优化器（改进的Adam，更好的权重衰减）
# optimizer_adamw = torch.optim.AdamW([W, b], lr=0.001, weight_decay=0.01)

# # 4. RMSprop优化器
# optimizer_rmsprop = torch.optim.RMSprop([W, b], lr=0.01, alpha=0.99)

# # 实际训练时的标准流程：
# def standard_training_loop():
#     """实际工作中的标准训练循环"""
#     # 选择优化器（最常用Adam）
#     optimizer = torch.optim.Adam([W, b], lr=0.001)
    
#     for epoch in range(num_epochs):
#         for X, y in train_iter:
#             # 1. 前向传播
#             y_hat = net(X)
#             loss = cross_entropy_loss(y_hat, y)
            
#             # 2. 清零梯度（重要！）
#             optimizer.zero_grad()
            
#             # 3. 反向传播
#             loss.backward()
            
#             # 4. 更新参数
#             optimizer.step()

# 测试一些预测
def predict(net, test_iter, n=6):
    """预测并显示结果"""
    print(f"\n预测示例（前{n}个样本）:")
    print("-" * 30)
    for X, y in test_iter:
        trues = d2l.get_fashion_mnist_labels(y)
        preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
        for i in range(min(n, len(trues))):
            print(f"真实: {trues[i]:>12}, 预测: {preds[i]:>12}")
        break

# 主程序 - 解决Windows多进程问题
if __name__ == '__main__':
    # 开始训练
    print("开始训练softmax回归模型...")
    print(f"训练集大小: {len(train_iter)} 批次")
    print(f"测试集大小: {len(test_iter)} 批次")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {lr}")
    print("-" * 50)

    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy_loss, num_epochs, sgd)

    print("-" * 50)
    print("训练完成！")

    # 显示最终参数
    print(f"\n最终参数形状:")
    print(f"权重 W: {W.shape}")
    print(f"偏置 b: {b.shape}")

    # 保存训练好的参数
    torch.save({'W': W, 'b': b}, 'd:/Code/Temp/python/softmax_params.pth')
    print(f"\n参数已保存到: d:/Code/Temp/python/softmax_params.pth")

    # 执行预测
    predict(net, test_iter)

# 加载训练好的参数
def load_model_params(filepath='d:/Code/Temp/python/softmax_params.pth'):
    """加载保存的模型参数"""
    try:
        params = torch.load(filepath)
        global W, b
        W = params['W']
        b = params['b']
        print(f"成功加载参数: {filepath}")
        print(f"权重 W 形状: {W.shape}")
        print(f"偏置 b 形状: {b.shape}")
        return True
    except FileNotFoundError:
        print(f"参数文件不存在: {filepath}")
        return False

# 使用示例：
# if load_model_params():
#     # 参数加载成功，可以直接使用模型进行预测
#     predict(net, test_iter)
# else:
#     # 参数文件不存在，需要重新训练
#     print("需要重新训练模型...")

# 添加一个函数来查看训练进度
def train_with_progress(net, train_iter, test_iter, loss, num_epochs, updater):
    """带进度显示的训练函数"""
    print("🚀 开始训练...")
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        
        # 显示进度条
        progress = (epoch + 1) / num_epochs
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f'Epoch {epoch + 1:2d}/{num_epochs} |{bar}| '
              f'Loss: {train_metrics[0]:.3f}, '
              f'Train Acc: {train_metrics[1]:.3f}, '
              f'Test Acc: {test_acc:.3f}')
    
    print("✅ 训练完成！")

# 修改训练调用（可选使用）
# 如果想要更好的进度显示，可以用这个函数替换 train_ch3
# train_with_progress(net, train_iter, test_iter, cross_entropy_loss, num_epochs, sgd)
