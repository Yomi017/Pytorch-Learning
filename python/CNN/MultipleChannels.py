import torch
from d2l import torch as d2l
import torch.nn.functional as F

def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
# 这里的操作是先corr2d计算每个通道的卷积结果，然后把所有通道的结果加在一起
# 这里的zip函数是Python内置函数，用于将多个可迭代对象打包成一个元组的迭代器
# 实际上在实际代码我们用的是: F.conv2d(X, K) 

# 测试是否相等
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(X.shape, K.shape) # X是3维张量，K是3维张量
print(corr2d_multi_in(X, K)==F.conv2d(X, K.unsqueeze(0)).squeeze(0)) # unsqueeze(0)在第0维增加一个维度，变成四维张量
# 这里的squeeze(0)是把第0维去掉，变回三维张量
# torch.stack()函数是将多个张量沿着指定维度拼接起来

def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

# 测试是否相等
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]],
               [[[1.0, 2.0], [3.0, 4.0]], [[0.0, 1.0], [2.0, 3.0]]]])
print(corr2d_multi_in_out(X, K)==F.conv2d(X, K))

# 如果核是1x1的卷积核，那么卷积层就相当于对每个像素点乘以一个权重，然后加上一个偏置
# 1x1卷积核的作用是改变通道数，而不改变高和宽
# 这样其实就变成了全连接层
# 1x1卷积核的计算量很小，可以用来减少参数数量