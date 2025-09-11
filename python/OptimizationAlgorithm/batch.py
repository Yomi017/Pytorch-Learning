import torch
from torch import nn
from d2l import torch as d2l

def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # 初始化模型
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights) # 初始化权重

    optimizer = trainer_fn(net.parameters(), **hyperparams) # 传入超参数
    # trainer_fn是优化算法的构造函数，用于实例化优化算法，传入模型参数和超参数
    # 比如Adam优化算法的构造函数是torch.optim.Adam，能自动处理权重衰减，Adam的公式如下
    # m(t) = beta1*m(t-1) + (1-beta1)*g(t)
    # v(t) = beta2*v(t-1) + (1-beta2)*g(t)^2
    # m_hat(t) = m(t)/(1-beta1^t)
    # hyperparams是一个字典，包含优化算法所需的超参数
    loss = nn.MSELoss(reduction='none') # 均方误差损失函数
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad() # 梯度清零
            out = net(X) # 前向计算
            y = y.reshape(out.shape) # 改变y的形状以匹配out
            l = loss(out, y) # 计算损失
            l.mean().backward() # 反向传播，计算梯度
            optimizer.step() # 更新参数
            # step()会自动处理权重衰减
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # MSELoss计算平方误差时不带系数1/2
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')


# step()工作原理：

# *   `optimizer = trainer_fn(net.parameters(), **hyperparams)` 这一行是**一次性的设置和初始化**。
# *   `optimizer.step()` 这一行是利用这个设置好的`optimizer`对象来**执行一次更新**。

# 想象一下你要雇一个修理工（**Optimizer**）来修理你的机器（**Model**）。

# #### **第 1 步：雇佣和交待任务 (初始化)**

# ```python
# optimizer = trainer_fn(net.parameters(), **hyperparams)
# ```

# 这行代码只在训练循环开始**之前执行一次**。它做了两件关键的事情：

# 1.  **创建修理工实例**: `trainer_fn` (比如 `torch.optim.Adam` 或 `torch.optim.SGD`) 创建了一个优化器对象。我们就叫他“亚当修理工”吧。
# 2.  **交待任务**:
#     *   `net.parameters()`: 你告诉亚当：“你要负责修理的，就是这台机器（`net`）里的**所有这些零件**（`parameters`）。” 优化器此时会**获取并持有了模型所有参数的引用**。这意味着它知道内存中每个参数（权重`w`和偏置`b`）的位置。
#     *   `**hyperparams`: 你递给亚当一个工具箱（超参数字典），里面有扳手（学习率 `lr`）、润滑油（动量 `momentum`）等等。你说：“以后修理的时候，就用这些工具和设置。”

# **完成这一步后，亚当修理工已经准备就绪。他知道要修哪个机器的哪些零件，也知道该用什么工具。**

# #### **第 2 步：诊断问题 (计算梯度)**

# ```python
# l.mean().backward()
# ```

# 这一步发生在训练循环的**每一次迭代**中。

# *   你的机器（模型）处理了一批任务（数据 `X`），但结果不太理想，产生了误差（`loss`）。
# *   `l.mean().backward()` 就像一个诊断程序。它会计算出**每个零件（参数）应该朝哪个方向调整，以及调整的幅度有多大**。
# *   这个“诊断报告”就是**梯度（gradient）**，它被自动存储在每个参数的 `.grad` 属性里（例如 `w.grad`, `b.grad`）。

# **此时，亚当修理工还没动手，但他已经看到了每个零件旁边都贴上了一张“修理说明”（梯度）。**

# #### **第 3 步：动手修理 (更新参数)**

# ```python
# optimizer.step()
# ```

# 这一步也发生在训练循环的**每一次迭代**中，紧跟在 `backward()` 之后。

# *   你对亚当修理工喊了一声：“动手！” (`.step()`)。
# *   亚当修理工会做以下事情：
#     1.  遍历他在第1步中被告知要负责的**所有零件** (`net.parameters()`)。
#     2.  查看每个零件上贴着的“修理说明” (`p.grad`)。
#     3.  打开他在第1步收到的工具箱 (`hyperparams`)，拿出扳手（学习率 `lr`）。
#     4.  根据优化算法自身的复杂规则（比如 Adam 的公式），结合“修理说明”和“工具”，对零件进行**实际的调整**（更新参数值）。

# **这个过程会直接修改模型参数的值，因为优化器从一开始就持有这些参数的引用。**

# ---

# ### 总结

# *   `optimizer = ...` 是 **“配置阶段”**：它只执行一次，告诉优化器要管哪些参数，以及用什么超参数。
# *   `optimizer.step()` 是 **“执行阶段”**：它在每次迭代时被调用，利用之前配置好的信息和最新计算出的梯度，去真正地更新模型参数。
