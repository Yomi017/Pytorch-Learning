import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K): # X: 输入矩阵 K: 卷积核
    """计算二维互相关运算"""
    h, w = K.shape # 卷积核的高和宽
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)) # 输出矩阵的高和宽
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum() # 矩阵元素对应相乘再求和
    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
    
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

X = torch.ones((6, 8))
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

for epoch in range(10):
    Y_hat = conv2d(X)  # 前向计算
    l = (Y_hat - Y) ** 2  # 均方误差损失函数
    conv2d.zero_grad()  # 梯度清零
    l.sum().backward()  # 反向传播
    # 使用梯度下降法更新权重
    conv2d.weight.data -= lr * conv2d.weight.grad
    print(f'epoch {epoch + 1}, loss {l.sum():.3f}')
print(conv2d.weight.data.reshape((1, 2))) # 打印学习到的卷积核权重
