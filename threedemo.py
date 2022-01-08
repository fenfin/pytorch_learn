import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data
import time
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l
'''
我们通过torchvision的torchvision.datasets来下载这个数据集。第一次调用时会自动从网上获取数据。
我们通过参数train来指定获取训练数据集或测试数据集（testing data set）。
测试数据集也叫测试集（testing set），只用来评价模型的表现，并不用来训练模型。

另外我们还指定了参数transform = transforms.ToTensor()使所有数据转换为Tensor，如果不进行转换则返回的是PIL图片。
transforms.ToTensor()将尺寸为 (H x W x C) 且数据位于[0, 255]的PIL图片或者数据类型为np.uint8的NumPy数组
转换为尺寸为(C x H x W)且数据类型为torch.float32且位于[0.0, 1.0]的Tensor。

注意： 由于像素值为0到255的整数，所以刚好是uint8所能表示的范围，
包括transforms.ToTensor()在内的一些关于图片的函数就默认输入的是uint8型，
若不是，可能不会报错但可能得不到想要的结果。所以，如果用像素值(0-255整数)表示图片数据，
那么一律将其类型设置成uint8，避免不必要的bug

下面的mnist_train和mnist_test都是torch.utils.data.Dataset的子类，
所以我们可以用len()来获取该数据集的大小，还可以用下标来获取具体的一个样本。
训练集中和测试集中的每个类别的图像数分别为6,000和1,000。
因为有10个类别，所以训练集和测试集的样本数分别为60,000和10,000。
'''
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

# print(type(mnist_train))
# print(len(mnist_train), len(mnist_test))

# 通过下标访问任意一个样本
# feature, label = mnist_train[0]
# print(feature.shape, label) # channel*height*width


# 将数值标签转换成相应的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
                   'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


#在一行里画出多张图像和对应标签的函数
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


# 看一下训练数据集中前10个样本的图像内容和文本标签
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
d2l.show_fashion_mnist(X, get_fashion_mnist_labels(y))
# show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 读取小批量
# 我们将在训练数据集上训练模型，并将训练好的模型在测试数据集上评价模型的表现。
# 前面说过，mnist_train是torch.utils.data.Dataset的子类，
# 所以我们可以将其传入torch.utils.data.DataLoader来创建一个读取小批量数据样本的DataLoader实例。
# 在实践中，数据读取经常是训练的性能瓶颈，特别当模型较简单或者计算硬件性能较高时。
# PyTorch的DataLoader中一个很方便的功能是允许使用多进程来加速数据读取。
# 这里我们通过参数num_workers来设置4个进程读取数据。
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 查看读取一遍训练数据需要的时间
start = time.time()
for X, y in test_iter:
    continue
print('%.2f sec' % (time.time()-start))