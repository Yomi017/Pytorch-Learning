
import torch
from torch import nn
from d2l import torch as d2l

T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32) # 时间步
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,)) # 正弦信号 + 噪声
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3)) # 画出前1000个点

tau = 4
features = torch.zeros((T - tau, tau)) # T - tau行，tau列的零矩阵
for i in range(tau):
    features[:, i] = x[i: T - tau + i] # 每一列是一个时间步的输入
labels = x[tau:].reshape((-1, 1)) # 目标是从第tau个点开始的所有点

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True) # 训练数据迭代器 load_array()函数会自动打乱数据
test_iter = d2l.load_array((features[n_train:], labels[n_train:]),
                           batch_size, is_train=False) # 测试数据迭代器
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# 平方损失
loss = nn.MSELoss(reduction='none')

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step() # step()函数会更新所有参数
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net() # 获取网络
train(net, train_iter, loss, 20, 0.01)

# 单步预测
onestep_preds = net(features)
# 使用 d2l.plot 函数绘制图形
d2l.plot(
    # --- X轴数据 ---
    # 提供了两个X轴序列：
    # 1. time: 原始数据点对应的时间序列 (例如 [1, 2, 3, ..., 1000])
    # 2. time[tau:]: 预测数据点对应的时间序列。
    #    因为预测是从第 tau 个时间点开始的，所以X轴也要从相应位置开始。
    [time, time[tau:]],

    # --- Y轴数据 ---
    # 提供了两个Y轴序列，与上面的X轴一一对应：
    # 1. x.detach().numpy(): 原始的、真实的序列数据。
    #    .detach() 是为了切断计算图，防止梯度计算。
    #    .numpy() 是将PyTorch张量转换为NumPy数组，以便绘图库使用。
    # 2. onestep_preds.detach().numpy(): 模型进行单步预测得到的结果序列。
    [x.detach().numpy(), onestep_preds.detach().numpy()],

    # --- 图形标签设置 ---
    # 'time' : 设置X轴的标签为 'time'
    # 'x'    : 设置Y轴的标签为 'x'
    'time', 'x',

    # --- 图例设置 ---
    # legend=['data', '1-step preds']: 为两条曲线设置图例。
    # 'data' 对应第一条曲线 (原始数据)
    # '1-step preds' 对应第二条曲线 (单步预测结果)
    legend=['data', '1-step preds'],

    # --- 坐标轴范围设置 ---
    # xlim=[1, 1000]: 将X轴的显示范围限制在 1 到 1000 之间。
    xlim=[1, 1000],

    # --- 画布大小设置 ---
    # figsize=(6, 3): 设置图形的尺寸为 6英寸宽，3英寸高。
    figsize=(6, 3)
)

# 多步预测
# 创建一个空的“预测记录本”
multistep_preds = torch.zeros(T)

# 2. 把“历史”抄写到记录本上
multistep_preds[: n_train + tau] = x[: n_train + tau]

# 3. 开始循环，一天一天地进行“预言”
for i in range(n_train + tau, T):
    # 4. 关键步骤：用“过去的记录”（包含自己的预测）来预测未来
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))
