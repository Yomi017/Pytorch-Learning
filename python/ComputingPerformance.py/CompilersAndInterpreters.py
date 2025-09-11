import torch
from torch import nn
from d2l import torch as d2l

def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)

net = torch.jit.script(net)
net(x)

# 这里的 `torch.jit.script` 是 PyTorch 的一个功能，用于将动态计算图转换为静态计算图，从而提升模型的执行效率。通过 `torch.jit.script`，PyTorch 会对模型进行静态分析和优化，使得模型在推理阶段能够更快地运行。
# 这种转换特别适用于部署模型，因为静态图通常比动态图更高效，尤其是在需要高吞吐量和低延迟的生产环境中。
# 需要注意的是，`torch.jit.script` 并不适用于所有模型，尤其是那些包含复杂控制流（如条件语句和循环）的模型。在使用 `torch.jit.script` 时，可能需要对模型代码进行一些调整，以确保其能够被成功转换为静态图。
# 其实就是在训练完之后，我们只需要前向传播就行了，不需要反向传播，所以可以把模型转换成静态图，这样可以提升前向传播的效率。
# 另外，`torch.jit.script` 还可以帮助我们捕捉和修复一些潜在的错误，因为它会对代码进行静态分析，确保代码在转换为静态图时不会出现问题。