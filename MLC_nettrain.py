import torch
import matplotlib.pyplot as plt
import xlrd
import numpy as np
import torch.nn.functional as F     # 激励函数都在这

# excels数据读取
data = xlrd.open_workbook('MLC.xlsx')
k0 = data.sheet_by_index(0)
k1 = data.sheet_by_index(1)
k2 = data.sheet_by_index(2)
k3 = data.sheet_by_index(3)

# 训练数据整理
k0_row0 = torch.Tensor(k0.row_values(0,0,None))
k0_row1 = torch.Tensor(k0.row_values(1,0,None))
k1_row0 = torch.Tensor(k1.row_values(0,0,None))
k1_row1 = torch.Tensor(k1.row_values(1,0,None))
k2_row0 = torch.Tensor(k2.row_values(0,0,None))
k2_row1 = torch.Tensor(k2.row_values(1,0,None))
k3_row0 = torch.Tensor(k3.row_values(0,0,None))
k3_row1 = torch.Tensor(k3.row_values(1,0,None))

k0_input = torch.cat((torch.unsqueeze(k0_row0,dim = 1),torch.unsqueeze(k0_row1,dim = 1)),dim = 1)
k1_input = torch.cat((torch.unsqueeze(k1_row0,dim = 1),torch.unsqueeze(k1_row1,dim = 1)),dim = 1)
k2_input = torch.cat((torch.unsqueeze(k2_row0,dim = 1),torch.unsqueeze(k2_row1,dim = 1)),dim = 1)
k3_input = torch.cat((torch.unsqueeze(k3_row0,dim = 1),torch.unsqueeze(k3_row1,dim = 1)),dim = 1)
data_input = torch.cat((k0_input,k1_input,k2_input,k3_input),dim = 0)
k0_output = []
k1_output = []
k2_output = []
k3_output = []
for i_0 in range(0,k0.ncols,1):
    k0_output.append(k0.col_values(i_0,2,None))
for i_1 in range(0,k1.ncols,1):
    k1_output.append(k1.col_values(i_1,2,None))
for i_2 in range(0,k2.ncols,1):
    k2_output.append(k2.col_values(i_2,2,None))
for i_3 in range(0,k3.ncols,1):
    k3_output.append(k3.col_values(i_3,2,None))

k0_output = torch.Tensor(k0_output)
k1_output = torch.Tensor(k1_output)
k2_output = torch.Tensor(k2_output)
k3_output = torch.Tensor(k3_output)

# print(np.array(k0_output).shape)
# print(np.array(k1_output).shape)
# print(np.array(k2_output).shape)
# print(np.array(k3_output).shape)

data_output = torch.cat((k0_output,k1_output,k2_output,k3_output),dim = 0)
print(np.array(data_output).shape)
print(np.array(data_input).shape)
x_label = torch.linspace(1,115,115)

plt.ion()
plt.figure(1)
plt.scatter(x_label.data.numpy(), np.array(data_output)[1,:])

plt.show()
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x

net = Net(n_feature=2, n_hidden=10, n_output=115)

print(net)  # net 的结构
# optimizer 是训练的工具
optimizer = torch.optim.Adamax(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

for t in range(7000):
    prediction = net(data_input)     # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(prediction, data_output)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    # 接着上面来
    if t % 20 == 0:
        # plot and show learning process
        plt.scatter(x_label.data.numpy(), np.array(data_output)[2,:])
        plt.draw()
        plt.plot(x_label.data.numpy(),prediction.data.numpy()[2,:], 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss, fontdict={'size': 20, 'color': 'blue'})
        plt.pause(0.1)


 # 保存神经网络
torch.save(net,'net.pkl')           # 保存整个神经网络的结构和模型参数