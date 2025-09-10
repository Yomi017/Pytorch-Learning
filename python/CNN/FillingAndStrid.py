import torch
from torch import nn

# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape) # 变成四维张量，为了适应Conv2d的输入格式
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:]) # 变回二维张量
    # Y.shape[2:] 是输出的高和宽

# conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1) # padding=1表示在每一边填充1行或1列
# conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1)) # 高方向填充2，宽方向填充1
# 一般填充大小为(kernel_size-1)/2的整数部分，这样可以保持输入输出的高和宽相同，即 p = k//2
# 输出的形状为：(n_h - k_h + p_h + 1) x (n_w - k_w + p_w + 1)

# 如果加上步幅stride参数，输出的形状为：
# ((n_h - k_h + p_h) / s_h + 1) x