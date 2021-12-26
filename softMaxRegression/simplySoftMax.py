#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ML 
@File    ：simplySoftMax.py
@Author  ：hujinrun
@Date    ：2021/12/26 4:14 下午 
'''
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
# 图像是个二维的数据需要定义Flatten层来进行展开
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

# 初始化线性层的参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)