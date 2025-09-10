import torch
from torch import nn
from d2l import torch as d2l

# 池化层
# mode='max'表示最大池化，mode='avg'表示平均池化
# 和卷积层类似，池化层也可以有多个通道，只不过是把乘法换成了求最大值或平均值
# 池化层没有权重参数
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y