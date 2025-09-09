import random
import torch

def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    """Ŷ = Xw 是将所有样本的计算一次性完成的高效矩阵形式"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 生成一个特征矩阵，列数为样本数量，行数为w的长度
    y = X @ w + b
    # y是通过线性模型生成的真实值
    y += torch.normal(0, 0.01, y.shape)
    # 加上一个和y大小相同的噪声
    return X, y.reshape((-1, 1)) # reshape将y变成列向量
    # -1 不是一个具体的数字，而是一个占位符。它告诉 reshape 函数，以确保重塑前后总的元素数量不变
    # 1 表示将y变成只有一列的二维数组

def data_iter(batch_size, features, labels):
    # 获取数据集的总样本数
    num_examples = len(features)
    
    # 生成一个从 0 到 num_examples-1 的索引列表
    # 比如 num_examples = 1000, indices 就是 [0, 1, 2, ..., 999]
    indices = list(range(num_examples))
    
    # 将索引列表随机打乱
    # 我们是打乱索引，而不是 features 和 labels 数据
    random.shuffle(indices)
    
    # 从头到尾以 batch_size 为步长进行切分
    # i 会依次等于 0, batch_size, 2*batch_size, ...
    for i in range(0, num_examples, batch_size):
        
        # 确定当前批次的索引范围，并创建成张量
        # 这里的 min() 是为了处理最后一个批次可能不足 batch_size 的情况
        # 比如总共1000个样本，batch_size=128，最后一个批次只有 1000 % 128 = 104个样本
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        
        # 使用 yield 返回当前批次的数据。
        # yield 是一个生成器关键字。它会返回一对 (features, labels) 的小批量数据，
        # 然后暂停在这里。下次调用时，它会从这里继续执行，进入下一次 for 循环。
        # 这样做的好处是节省内存，不需要一次性把所有批次都创建好。
        yield features[batch_indices], labels[batch_indices]

def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size): 
    """小批量随机梯度下降"""
    
    # 关键的上下文管理器
    with torch.no_grad():
    # 不需要计算梯度，所以我们用这个上下文管理器包裹代码块
    # 在这里是手动更新参数 param 的值。这个更新操作本身是优化算法的一部分，而不是模型的一部分
    # 如果不加，那么 w 和 b 的梯度会被计算图记录下来
    # 它影响的是在下面使用 param.grad 进行减法运算的时候不会记录计算图

        # 遍历所有需要更新的参数
        for param in params:
            
            # 核心更新逻辑
            param -= lr * param.grad / batch_size
            
            # 梯度清零
            param.grad.zero_()

def linreg(X, w, b):
    """线性回归模型"""
    return X @ w + b

# 生成合成数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 初始化参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练参数
batch_size = 10
lr = 0.03 # 学习率
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')