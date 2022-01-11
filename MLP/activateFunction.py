import torch
import numpy as np
import matplotlib.pylab as plt
import d2lzh_pytorch as d2l
import sys
sys.path.append("..")


# ReLU激活函数
def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    plt.show() # 加上这句，否则图形显示不出来


# 通过Tensor提供的relu函数来绘制ReLU函数
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
# xyplot(x, y, 'relu')
# y.sum().backward()
# xyplot(x, x.grad, 'grad of relu')

# sigmod函数
y = x.sigmoid()
xyplot(x, y, 'sigmoid')

# sigmoid'(x) = sigmoid(x)(1-sigmoid(x))
# sigmoid函数导数
# x.grad.zero_() # 运行时这个要先注释掉，否则会报错
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')

# tanh函数
y = x.tanh()
xyplot(x, y, 'tanh')

# 导数tanh'(x) = 1-tanh^2(x)
# 绘制导数
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')

# 多层感知机就是含有至少一个隐藏层的由全连接层组成的神经网络，
# 且每个隐藏层的输出通过激活函数进行变换。
# 多层感知机的层数和各隐藏层中隐藏单元个数都是超参数。
# 以单隐藏层为例并沿用本节之前定义的符号，

