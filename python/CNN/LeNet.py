import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

# 关闭TensorFlow警告信息
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), # padding=2表示在每一边填充2行或2列
    nn.AvgPool2d(kernel_size=2, stride=2), # 池化层，kernel_size=2表示池化窗口为2x2，stride=2表示步幅为2
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(), # 这里没有填充，所以输出的高和宽会减小
    nn.AvgPool2d(kernel_size=2, stride=2), # 池化层，kernel_size=2表示池化窗口为2x2，stride=2表示步幅为2
    nn.Flatten(), # 展平多维的输入数据为二维
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(), # 将16个5x5特征图展平为400维向量，映射到120个隐藏单元
    nn.Linear(120, 84), nn.Sigmoid(), # 120指的是前一层的输出，映射到84个隐藏单元
    nn.Linear(84, 10))
# nn.Dense() # 全连接层，等价于nn.Linear()
# nn.Dropout() # Dropout层，防止过拟合
# nn.ReLU() # ReLU激活函数，LeNet-5原版使用sigmoid，但是AlexNet等后续网络发现ReLU更好
# Conv2d 卷积层 -> Sigmoid 激活函数 -> AvgPool2d 池化层 -> Conv2d 卷积层 -> Sigmoid 激活函数 
# -> AvgPool2d 池化层 -> Flatten 展平 -> Linear 全连接层 -> Sigmoid 激活函数 -> Linear 全连接层 
# -> Sigmoid 激活函数 -> Linear 全连接层
# 总共10层： 2个卷积层，2个池化层，1个展平层，3个全连接层，3个激活函数层
# 这是LeNet-5网络结构

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 评估模型在GPU上的准确率
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 训练模型
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    def init_weights(m): # 初始化模型参数
        if type(m) == nn.Linear or type(m) == nn.Conv2d: # 如果是线性层或卷积层
            nn.init.xavier_uniform_(m.weight) # Xavier均匀分布初始化，适合于Sigmoid激活函数
            # 不用正态分布初始化，因为正态分布初始化会使得权重过大或过小，导致梯度消失或爆炸
    net.apply(init_weights) # 初始化模型参数
    print('training on', device)
    net.to(device) # 将模型参数复制到device上，device可以是'cpu'或'cuda'
    
    # 设置TensorBoard
    log_dir = f"runs/LeNet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f'TensorBoard logs saving to: {log_dir}')
    print('Run: tensorboard --logdir=runs')
    
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr) # 随机梯度下降优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4) # Adam优化器 + 权重衰减
    loss = nn.CrossEntropyLoss() # 交叉熵
    timer, num_batches = d2l.Timer(), len(train_iter) # 计时器，训练批次数
    
    # 记录模型结构到TensorBoard
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    writer.add_graph(net, dummy_input)
    # 训练周期
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3) # 训练损失之和，训练准确率之和，样本数
         # 训练模式
        net.train()
        for i, (X, y) in enumerate(train_iter): # 遍历训练数据集
            timer.start()
            optimizer.zero_grad() # 梯度清零
            X, y = X.to(device), y.to(device) # 将数据复制到device上
            y_hat = net(X) # 前向计算
            l = loss(y_hat, y) # 计算损失
            l.backward() # 反向传播计算梯度
            optimizer.step() # 更新参数
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0]) # 累加损失、准确率、样本数
            timer.stop() 
            train_l = metric[0] / metric[2] # 平均损失
            train_acc = metric[1] / metric[2] # 平均准确率
            
            # 记录训练过程到TensorBoard（每100个batch记录一次）
            global_step = epoch * num_batches + i
            if (i + 1) % 100 == 0 or i == num_batches - 1:
                writer.add_scalar('Loss/Train', train_l, global_step)
                writer.add_scalar('Accuracy/Train', train_acc, global_step)
                
        # 每个epoch结束后记录测试准确率
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        
        # 记录学习率
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 记录权重和梯度的直方图（每5个epoch记录一次）
        if epoch % 5 == 0:
            for name, param in net.named_parameters():
                writer.add_histogram(f'Weights/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        print(f'epoch {epoch + 1}, loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    
    # 关闭TensorBoard writer
    writer.close()
    
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    print(f'TensorBoard logs saved to: {log_dir}')
    print('To view results, run: tensorboard --logdir=runs')
    
lr, num_epochs = 0.001, 10  # 学习率：Adam通常用0.001
# TensorBoard 使用说明
"""
🎯 TensorBoard 可视化功能：

1. 📊 训练损失和准确率曲线
2. 🧠 模型结构图
3. 📈 权重和梯度分布直方图
4. ⚡ 学习率变化
5. 🎨 图像样本（可选）

🚀 使用步骤：
1. 运行训练脚本
2. 打开终端，运行: tensorboard --logdir=runs
3. 在浏览器打开: http://localhost:6006
4. 查看各种可视化图表

📁 日志文件保存在: runs/LeNet_YYYYMMDD_HHMMSS/
"""

def log_sample_images(writer, data_iter, device, epoch=0):
    """记录样本图像到TensorBoard"""
    # 获取一批样本
    images, labels = next(iter(data_iter))
    images = images[:8].to(device)  # 只取前8个样本
    labels = labels[:8].to(device)
    
    # 记录原始图像
    writer.add_images('Sample_Images/Original', images, epoch)
    
    # 记录预测结果
    net.eval()
    with torch.no_grad():
        predictions = net(images)
        predicted_labels = predictions.argmax(dim=1)
    
    # 创建标题（真实标签 vs 预测标签）
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    for i in range(len(images)):
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted_labels[i]]
        writer.add_text(f'Predictions/Sample_{i}', 
                       f'True: {true_label}, Pred: {pred_label}', epoch)

# 主训练执行
if __name__ == '__main__':
    print("🚀 开始训练LeNet-5模型...")
    print("📊 使用TensorBoard进行可视化监控")
    print("-" * 50)
    
    # 检查设备
    device = d2l.try_gpu()
    print(f"🔧 使用设备: {device}")
    
    # 开始训练
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
    
    print("\n" + "="*50)
    print("🎉 训练完成！")
    print("📊 查看TensorBoard:")
    print("   1. 打开终端")
    print("   2. 运行: tensorboard --logdir=runs")
    print("   3. 打开浏览器: http://localhost:6006")
    print("="*50)