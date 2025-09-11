import torch
from torch import nn
from d2l import torch as d2l

# VGG块
# 使用多个卷积层堆叠来提取特征，然后通过池化层降低空间维度
# 卷积层使用3x3的卷积核，填充为1以保持高宽不变
# 池化层使用2x2的窗口，步幅为2以减半
# 实际上每个块是一样的，只是通道数不同
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1)) # 保持高宽不变，padding=(kernel_size-1)/2
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2)) # 高宽减半
    return nn.Sequential(*layers)

# NiN块
# 使用1x1卷积层来增加非线性，同时减少参数数量
# 每个NiN块包含一个常规卷积层，后接两个1x1卷积层，每个卷积层后面都跟一个ReLU激活函数
# 实际上1x1卷积相当于对每个像素点进行全连接操作
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels,
                  kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels,
                  kernel_size=1),
        nn.ReLU()
    )

# ResNet块
# 每个ResNet块包含两个3x3卷积层和一个跳跃连接
# 如果输入输出通道数不同，或者步幅不为1，则使用1x1卷积调整输入
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.relu = nn.ReLU()

    def forward(self, X):
        Y = self.conv1(X)
        Y = self.relu(Y)
        Y = self.conv2(Y)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)

# DenseNet块
class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, growth_rate):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(num_convs):
            self.net.add_module(f'conv{i}',
                self.conv_block(in_channels + i * growth_rate,
                                growth_rate))
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1))
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1) # 在通道维上连结
        return X
    
