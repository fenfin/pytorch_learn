import matplotlib_inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.nn as nn
import torch.utils.data as Data
import sys
sys.path.append("..")
from d2lzh_pytorch import *

# 生成数据集
num_inputs = 2  # 输入参数的数量
num_examples = 1000  # 要生成的数据个数
true_w = [2, -3.4]
true_b = 4.2
# features是训练数据集特征
# lables是标签
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

# 读取数据
batch_size = 10
# 将训练的特征和标签混合
dataset = Data.TensorDataset(features,labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# 读取并打印第一个小批量数据样本
# for X,y in data_iter:
#     print(X,y)
#     break


# 定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
# print(net)  # 打印网络结构
# for parm in net.parameters():
#     print(parm)
'''
定义模型 使用Sequential，是个有序容器
写法一
   net = nn.Sequential(
   nn.Linear(num_inputs, 1)
   # 此处还可以传入其他层
   )
   
写法二
   net = nn.Sequential()
   net.add_module('linear', nn.Linear(num_inputs, 1))
   # net.add_module ......

写法三
   from _collections import OrderedDict
   net = nn.Sequential(OrderedDict([
    ('linear', nn.Linear(num_inputs, 1))
    # ......
]))
'''
# 初始化模型参数
# 我们通过init.normal_将权重参数每个元素初始化为随机采样于均值为0、标准差为0.01的正态分布。偏差会初始化为零
from torch.nn import init
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
'''
可以为不同子网络设置不同的学习率
optimizer = optim.SGD([
{'params':net.subnet1.parameters()} #如果不指定学习率，使用最外层默认学习率0.03
{'params':net.subnet1.parameters(), 'lr':0.01}
], lr=0.03)

实时调整学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1 # 学习率为之前的0.1倍

'''
# 训练模型
# 在使用Gluon训练模型时，我们通过调用optim实例的step函数来迭代模型参数。
# 按照小批量随机梯度下降的定义，我们在step函数中指明批量大小，从而对批量中样本梯度求平均
num_epochs = 3  # 迭代次数
for epoch in range(1, num_epochs+1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  #梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

# 比较模型参数和真实，从net获得需要的层
dense = net.linear
print(true_w, dense.weight)
print(true_b, dense.bias)