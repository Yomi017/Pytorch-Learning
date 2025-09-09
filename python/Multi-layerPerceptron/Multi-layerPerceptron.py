import torch

# ReLU
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)

# sigmoid
y = torch.sigmoid(x)

# tanh
y = torch.tanh(x)

y.backward(torch.ones_like(x), retain_graph=True)
# torch.ones_like(x) 这个和 y.sum().backward() 是等价的

