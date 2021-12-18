import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
# nn是神经网络的缩写
from torch import nn

# 连接模型的容器，模型会光联起来
# Linear是线性层（这里输入维度是2，输出维度是1）
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# mean square loss - 均方损失
loss = nn.MSELoss()
# Stochastic Gradient Descent - 随机梯度下降
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 构造数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        trainer.zero_grad() # 重置梯度
        l = loss(net(X) ,y) # 计算损失
        l.backward() # 反向传播
        trainer.step() # 更新参数
    with torch.no_grad():
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print(f'w的估计误差： {true_w - w.reshape(true_w.shape)}')
b = net[0].bias.data
print(f'b的估计误差： {true_b - b}')

