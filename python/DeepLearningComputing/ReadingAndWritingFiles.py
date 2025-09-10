import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4) 
y = torch.zeros(4)
torch.save(x, 'x-file') # 保存张量到文件
x2 = torch.load('x-file') # 从文件加载张量
torch.save([x, y],'x-files') # 保存多个张量到一个文件
x2, y2 = torch.load('x-files') # 从文件加载多个张量

# 多层感知机作为例子
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

clone = MLP() # 新建一个相同结构的模型，但是参数是随机初始化的
clone.load_state_dict(torch.load('mlp.params')) # load_state_dict() 函数将文件中的参数加载到模型中
clone.eval() # .eval() 是 nn.Module 的一个方法，用于将模型设置为评估模式。
# 评估模式的意思是
# 评估模式下，某些层（如 dropout 和 batch normalization）会表现出不同的行为