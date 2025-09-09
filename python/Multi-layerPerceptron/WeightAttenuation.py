import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 设置matplotlib后端，确保图形能正确显示
plt.switch_backend('TkAgg')  # 或者尝试 'Qt5Agg', 'Agg'

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
# 定义真实的权重和偏置（ground truth）
true_w = torch.ones((num_inputs, 1)) * 0.01  # 200x1 的权重向量，每个值都是 0.01
true_b = 0.05  # 偏置项
# 生成合成数据：X 和对应的标签 y
train_data = d2l.synthetic_data(true_w, true_b, n_train)  # 生成 20 个训练样本
test_data = d2l.synthetic_data(true_w, true_b, n_test)   # 生成 100 个测试样本
train_iter = d2l.load_array(train_data, batch_size)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# 初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2 # L2范数惩罚项

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss # 定义线性回归模型和均方误差损失函数
    # linreg函数的输入是X，w，b，输出是预测值y_hat
    # squared_loss函数的输入是y_hat和y，输出是每个样本的平方误差
    # lambda X: d2l.linreg(X, w, b), d2l.squared_loss 其实是创建了两个函数对象
    num_epochs, lr = 100, 0.003 # 训练轮数和学习率
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test']) # 训练损失和测试损失
    # xlabel: x轴标签 ylabel: y轴标签 yscale: y轴刻度 xlim: x轴范围 legend: 图例
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())
    return w, b

if __name__ == '__main__':
    print("🚀 开始权重衰减实验...")
    
    print("\n📊 实验1: 不使用权重衰减 (λ=0)")
    w1, b1 = train(lambd=0)
    
    print("\n📊 实验2: 使用权重衰减 (λ=3)")  
    w2, b2 = train(lambd=3)
    
    print(f"\n📈 对比结果:")
    print(f"无权重衰减时 - w的L2范数: {torch.norm(w1).item():.4f}")
    print(f"有权重衰减时 - w的L2范数: {torch.norm(w2).item():.4f}")
    print(f"权重衰减效果: L2范数减少了 {(torch.norm(w1) - torch.norm(w2)).item():.4f}")
    
    # 确保显示图形
    try:
        plt.show()
    except Exception as e:
        print(f"图形显示时发生错误: {e}")
    print("\n✅ 训练完成！图形已显示，请查看弹出的窗口。")

# 简洁版
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())
