import torch
from d2l import torch as d2l

def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 初始化模型
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # 训练模型
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean() # 这里用mean来计算批量样本的平均损失，原因在下面
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]



# “既然要浓缩成一个标量，用 `.sum()` 求和行不行？”

# 答案是：技术上可行，但实践中非常不好！

# *   **如果用 `.sum()`**: 假设 `batch_size` 是 10，总损失可能是 5.0。如果 `batch_size` 变成 100，总损失可能就变成了 50.0。这意味着你计算出的梯度大小会**直接依赖于你的 `batch_size`**。如果你改变了 `batch_size`，梯度会相应地放大或缩小 10 倍，这会导致你的学习率 (`lr`) 变得极不稳定。你为 `batch_size=10` 调好的 `lr`，在 `batch_size=100` 时可能就太大了，导致模型无法收敛。

# *   **如果用 `.mean()`**: 无论你的 `batch_size` 是 10、100 还是 1000，计算出的平均损失都会在同一个量级上。这使得损失的期望值（以及梯度的期望值）**与 `batch_size` 无关**。这样做的好处是：
#     *   **超参数更稳定**：你调整好的学习率 `lr` 在更换 `batch_size` 时，依然有参考价值，不需要进行数量级上的调整。
#     *   **损失更具可比性**：你可以直观地比较不同实验（即使 `batch_size` 不同）的损失值，因为它们都在同一个尺度上。

# ### 总结

# 简单来说：

# 1.  **从技术上**，`loss()` 对一个 mini-batch 计算后得到的是一个损失向量，而 `l.backward()` 需要一个标量（单个数值）作为起点来进行求导。
# 2.  **从实践上**，使用 `.mean()` 而不是 `.sum()`，可以消除批量大小（batch size）对损失值和梯度大小的影响，使得训练过程和超参数（尤其是学习率）更加稳定和通用。这符合我们优化**整个数据集上的平均损失（经验风险）**这一最终目标。