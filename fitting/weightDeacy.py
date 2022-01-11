# 高维线性回归实验
# y=0.05+∑i=1 0.01xi+c
import torch
import torch.nn as nn
import matplotlib.pylab as plt
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1)*0.01, 0.05

features = torch.randn(n_train + n_test, num_inputs)
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    plt.show()


# 从零开始实现权重衰减
# 初始化模型参数
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# 定义L2范数惩罚项
# 这里只惩罚模型的权重参数
def l2_penalty(w):
    return (w**2).sum() / 2


# 定义训练和测试
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加了L2范数惩罚项
            l = loss(net(X, w, b), y) + lambd*l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w ,b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs+1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().item())


# 观察过拟合
fit_and_plot(lambd=0)  # lambd=0,没有使用权重衰减

# 使用权重衰减
fit_and_plot(lambd=3)


# 简洁实现权重衰减
# 这里我们直接在构造优化器实例时通过weight_decay参数来指定权重衰减超参数。
# 默认下，PyTorch会对权重和偏差同时衰减。我们可以分别对权重和偏差构造优化器实例，
# 从而只对权重衰减。
def fit_and_plot_pytorch(wd):
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd)
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)  # 不对偏参数衰减

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            l.backward()  # 求解梯度

            # 分别调用step函数，更新权重和偏差
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'loss',
             range(1, num_epochs+1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())


fit_and_plot_pytorch(0)
fit_and_plot_pytorch(3)
