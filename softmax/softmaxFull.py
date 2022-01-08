import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

# 获取和读取数据
# 我们将使用Fashion-MNIST数据集，并设置批量大小为256。
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
# 跟线性回归中的例子一样，我们将使用向量表示每个样本。已知每个样本输入是高和宽均为28像素的图像。
# 模型的输入向量的长度是 28×28=784：该向量的每个元素对应图像中每个像素。
# 由于图像有10个类别，单层神经网络输出层的输出个数为10，
# 因此softmax回归的权重和偏差参数分别为784×10和1×10的矩阵。
num_inputs = 784
num_outputs = 10
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 实现softmax运算
'''
下面我们就可以定义前面小节里介绍的softmax运算了。在下面的函数中，矩阵X的行数是样本数，列数是输出个数。
为了表达样本预测各个输出的概率，softmax运算会先通过exp函数对每个元素做指数运算，再对exp矩阵同行元素求和，
最后令矩阵每行各元素与该行元素之和相除。这样一来，最终得到的矩阵每行元素和为1且非负。
因此，该矩阵每行都是合法的概率分布。
softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别上的预测概率。
'''
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)  # dim=1 对同一行；dim=0 对同一列；keepdim=true 保留行和列两个维度
    return X_exp / partition  # 这里应用了广播机制


# 定义模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


# 定义损失函数
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))


def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 计算分类准确率 y_hat.argmax(dim=1)返回矩阵y_hat每行中最大元素的索引
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1)==y).float().mean().item()


# 评价模型net在数据集data-iter上的准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 训练模型
num_epochs, lr = 5, 0.1


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch+1, train_l_sum/n, train_acc_sum/n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# 预测结果
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

