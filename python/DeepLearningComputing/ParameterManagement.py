import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

print(net[2].state_dict()) # 打印第二个线性层的参数 state_dict() 返回一个有序字典，包含了模块的所有参数和持久缓冲区
# 持久缓冲区是指那些不需要梯度更新的张量，比如批量归一化层的均值和方差

print(type(net[2].bias)) # 打印bias的类型，bias是一个参数张量 Parameter 是 Tensor 的子类，表示模型的可学习参数，这里是打印偏置bias的类型
print(net[2].bias) # 打印偏置bias的值
print(net[2].bias.data) # 打印偏置bias的值

print(net[2].weight.grad) # 打印权重weight的梯度，刚初始化时梯度为None

print(*[(name, param.shape) for name, param in net[0].named_parameters()]) # 打印第一个线性层的所有参数的名字和形状
print(*[(name, param.shape) for name, param in net.named_parameters()]) # 打印整个网络的所有参数的名字和形状
# .named_parameters() 返回一个生成器，生成模块的所有参数的名字和参数张量
# 这一行中，(name, param.shape) 是一个元组，name是参数的名字，param.shape是参数的形状
# 加上(name, param.shape)就是用for循环遍历所有参数的名字和形状，生成一个有结构的数据结构
# * 是解包操作符，将生成器中的每个元素作为单独的参数传递给print函数
# for name, param in net.named_parameters() 这一部分是一个迭代器，遍历网络的所有参数，其中name是参数的名字，param是参数张量

net.state_dict()['2.bias'].data # 通过state_dict访问参数
# 这里的 '2.bias' 是指网络中第三个模块（索引从0开始）的偏置参数
# 因为net是Sequential容器，包含3个模块：索引0是nn.Linear(4,8)，索引1是nn.ReLU()，索引2是nn.Linear(8,1)
# ReLU激活函数没有参数，所以'2.bias'对应的是第二个Linear层的偏置参数
# net.state_dict() 返回一个有序字典，包含了网络的所有参数和持久缓冲区