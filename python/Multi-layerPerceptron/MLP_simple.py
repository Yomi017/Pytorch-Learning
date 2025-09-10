import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
# 这个函数是在初始化模型参数：将所有线性层的权重初始化为均值为0，标准差为0.01的正态分布随机数
# 这个函数会被apply方法使用，作用是初始化每一层的权重
# m是传入的每一层

net.apply(init_weights)
# 使用apply方法将init_weights函数应用到net的每一层

batch_size, lr, num_epochs = 256, 0.1, 10 # 批量大小、学习率、迭代周期
loss = nn.CrossEntropyLoss(reduction='none') # 交叉熵损失函数 reduction='none'表示不对损失值做任何缩减
# SGD (Stochastic Gradient Descent) - 随机梯度下降
# 特点：使用固定学习率，直接按梯度方向更新参数
# 公式：θ = θ - lr * ∇θ
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# Adam (Adaptive Moment Estimation) - 自适应矩估计
# 特点：自适应学习率，结合了动量和RMSprop的优点
# 维护每个参数的一阶矩(动量)和二阶矩(梯度平方的移动平均)
# 通常收敛更快，对超参数不敏感
# trainer = torch.optim.Adam(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

from MLP import train_ch3

if __name__ == '__main__':
    print("🚀 开始训练简化版MLP...")
    print(f"📊 模型结构:\n{net}")
    print(f"📈 参数数量: {sum(p.numel() for p in net.parameters()):,}")
    print("-" * 50)
    
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    print("\n✅ 训练完成！")
