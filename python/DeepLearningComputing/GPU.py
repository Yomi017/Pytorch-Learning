import torch
from torch import nn

torch.device('cpu') # 使用 CPU
torch.device('cuda') # 等同于 torch.device('cuda:0')
torch.device('cuda:1') # 使用第二块 GPU

print(torch.cuda.device_count()) # GPU 的数量
print(torch.cuda.is_available()) # 检查 CUDA 是否可用
print(torch.cuda.get_device_name(0)) # 打印第一块 GPU 的名称
print(torch.cuda.current_device()) # 当前使用的 GPU 设备索引

x = torch.tensor([1, 2, 3])
print(x.device) # 张量一般在 CPU 上创建

# 指定张量在 GPU 上创建
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

X = torch.ones(2, 3, device=try_gpu()) # 在第一个GPU上创建
print(X)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu()) # 将模型参数复制到第一个GPU
print(net(X))
print(net[0].weight.data.device) # 查看模型参数所在设备