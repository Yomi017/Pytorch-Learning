import numpy as np
import torch
from mpl_toolkits import mplot3d
from d2l import torch as d2l

def f(x):
    return x * torch.cos(np.pi * x) # 风险函数

def g(x):
    return f(x) + 0.2 * torch.cos(5 * np.pi * x) # 实际上的，有噪声的风险函数，即目标函数（我们只能优化这个） 

# 经验风险是训练数据集的平均损失，而风险则是整个数据群的预期损失
