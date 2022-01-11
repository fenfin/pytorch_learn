# 多层感知机简洁实现
import torch
from torch import nn
from torch.nn import init
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# 定义模型
num_inputs = 784
num_ouputs = 10
num_hiddens = 256

net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_ouputs),
)
for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)


# 读取数据并训练模型
batch_zise = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_zise)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5  # 迭代次数
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_zise, None, None, optimizer)
# 报错的话，需要修改d2lzh_pytorch中的utils文件，使用第三节的evaluate_accuracy函数
