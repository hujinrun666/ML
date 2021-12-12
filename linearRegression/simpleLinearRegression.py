import random
import torch
from d2l import torch as d2l

# 随机初始化数据
def generateData(w, b, num_example):
    X = torch.normal(0, 0.01, (num_example, len(w))) ## 生成训练数据
    Y = torch.matmul(X, w)+b ## 生成结果数据
    Y += torch.normal(0, 0.01, Y.shape) ## 添加噪声
    return X, Y.reshape(-1,1)

# 数据加载函数
def data_iter(batch_size, features, labels):
    dataLen = len(features)
    indices = list(range(dataLen))
    random.shuffle(indices) # 对索引随机化
    for i in range(0, dataLen, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, dataLen)])
        yield features[batch_indices], labels[batch_indices]

# 模型函数
def linearReg(X, w, b):
    return torch.matmul(X,w)+b

# 损失函数
def squaredLoss(y_hat, y):
    return (y_hat-y.reshape(y_hat.shape))**2/2

# 优化函数（小批量梯度下降）
def sgd(params, lr, batchSize):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batchSize
            param.grad.zero_()

lr = 0.3
num_epochs = 30
net = linearReg
loss = squaredLoss

true_w = torch.tensor([4.1,3.2])
true_b = 4.0
batch_size = 10
features, labels = generateData(true_w, true_b, 10000)
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
for epoch in range(num_epochs):
    for X, Y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b),Y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch+1},loss {float(train_l.mean()):f}')
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