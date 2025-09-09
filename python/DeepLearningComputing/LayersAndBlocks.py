import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# 这里定义了一个简单的多层感知机（MLP）
# 输入层：20维 隐藏层：256个神经元，激活函数ReLU 输出层：10维
# nn.Sequential 是一个容器，按顺序将各个层组合在一起
# 如果我们要一个三层的，可以看例子：nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

X = torch.rand(2, 20) # 随机输入
net(X) # 前向计算

class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params
        super().__init__() # super() 函数是用来调用父类（超类）的方法的
        # 父类是nn.Module，它的__init__方法会初始化一些必要的成员变量，包括self._modules，self._parameters等
        # self._modules是一个有序字典，保存了所有子模块；self._parameters是一个有序字典，保存了所有参数，包括权重和偏置
        self.hidden = nn.Linear(20, 256)  # 隐藏层 输入20维，输出256维
        self.out = nn.Linear(256, 10)  # 输出层
        # 已经自动初始化了权重和偏置
        # 权重weight是一个矩阵，偏置bias是一个向量

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))

net = MLP()
net(X)

# 打印参数tensor
# print("Hidden layer weights:", net.hidden.weight)
# print("Hidden layer bias:", net.hidden.bias)
# print("Output layer weights:", net.out.weight)
# print("Output layer bias:", net.out.bias)

# 自定义一个类似nn.Sequential的类
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            # OrderedDict是一个有序字典，记录了成员添加的顺序
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
    
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)