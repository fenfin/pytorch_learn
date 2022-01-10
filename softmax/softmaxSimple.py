import torch
from torch import nn
from torch.nn import init
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# 获取和读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义和初始化模型
# softmax回归的输出层是一个全连接层，所以我们用一个线性模块就可以了。
# 因为前面我们数据返回的每个batch样本x的形状为(batch_size, 1, 28, 28),
# 所以我们要先用view()将x的形状转换成(batch_size, 784)才送入全连接层。
num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):  # x (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y


net = LinearNet(num_inputs, num_outputs)
# 对x的形状转换功能自定义一个FlattenLayer
# class FlattenLayer(nn.Module):
#     def __init__(self):
#         super(FlattenLayer, self).__init__()
#     def forward(self, x): # x shape: (batch, *, *, ...)
#         return x.view(x.shape[0], -1)
# # 可以更方便定义模型
# from collections import OrderedDict
#
# net = nn.Sequential(
#     # FlattenLayer(),
#     # nn.Linear(num_inputs, num_outputs)
#     OrderedDict([
#         ('flatten', FlattenLayer()),
#         ('linear', nn.Linear(num_inputs, num_outputs))
#     ])
# )

# 使用均值0，标准差0.01的正态分布随机初始化模型的权重参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# softmax和交叉熵损失函数
loss = nn.CrossEntropyLoss()

# 定义优化算法
# 使用学习率0.1的小批量随机梯度下降作为优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练模型
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

# 模型预测
X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里得_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


show_fashion_mnist(X[0:9], titles[0:9])
