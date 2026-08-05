import numpy as np
from CnnFc import CNN_LSTM,MyDataset,min_max
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pyswarms as ps
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.functions import single_obj as fx
#===============GPU==================
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")
print("Use cuda: %s, gpu id: %d.", use_cuda, gpu)

#=====================训练集==============
data = np.loadtxt('./train_ParaHzAbsorb.txt')
print('data.shape', data.shape)
x_train = data[:, 0:6]
y_train = data[:, 6:7]
ptp_value = y_train.ptp(axis=0)
min_value = y_train.min(axis=0)
HZ= np.array(range(480,1622,2))
#==============加载模型======================
filter_size, conv_input_channel, conv_output_channel = 4, 1, 4  # cnn
hidden_size, num_layers = 100, 2  # LSTM
output_size = 1  # Fnn
dropout = 0.15
training = True
model = CNN_LSTM(filter_size, conv_input_channel, conv_output_channel,
                 hidden_size, num_layers, output_size, dropout, training)
model.to(device)
model.load_state_dict(torch.load('./CNN9988.pth'))
#=======================# 定义适应度函数=================
def func2(x,a = 3):
    """
    最小化适应度函数
    :param x:
    :return:
    """
    #x = int("x",2)
    ##print(x.shape)
    return 3 * np.cos(x[:,0] * x[:,1]) + x[:,0] + x[:,1] ** 2#在函数前加一个负号，既可以求原函数的极大值
def model_fun(x):
    print('x.shape', x.shape)
    x_test = []
    for i in range(x.shape[0]):
        new_para = x[i,:].reshape(1, -1).repeat(571, axis=0)
        parameter = np.concatenate((new_para,HZ.reshape(-1, 1)), axis=1)
        x_test.append(parameter)

    x_test = np.array(x_test).reshape(-1,6)
    y_test = np.array([i for i in range(x_test.shape[0])])
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    x_test_scaler = (x_test - min_value) / ptp_value


    test_dataset = MyDataset(x_test_scaler, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    y_pred = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_dataloader, 1):
            data_x = x.to(torch.float32).to(device)
            y_pred.append(model(data_x).reshape(-1, ).cpu())
    pred = np.array(y_pred) * ptp_value + min_value
    print(pred.shape)
    y_pre= pred.reshape(-1,571,1)
    cost = []
    for i in range(y_pre.shape[0]):
        cost.append(np.max(y_pre[i]))
    return np.array(cost).reshape(-1,1)

if __name__ == "__main__":
    #==========参数变化的上下限===========================
    constraints = (np.array([18,4,4,4,4]),
                   np.array([28,7,7,7,7]))
    #=======================例子群算法=====================
    options = {'c1': 2, 'c2': 2, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=5, options=options,bounds=constraints)
    cost, pos = optimizer.optimize(func2, iters=2)
    #======================绘制损失曲线=============
    print('最优位置：', pos, '最优适应度：', cost)
    print(type(optimizer.cost_history),'\n',len(optimizer.cost_history))
    coss = optimizer.cost_history
    plt.figure(figsize=(8, 5))
    plt.plot(range(1,len(coss)+1),coss, linewidth=2)
    plt.title("Coss", fontdict={'fontsize': 12})
    plt.xlabel('epochs', fontdict={'fontsize': 12})
    plt.ylabel('value', fontdict={'fontsize': 12})
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc='upper right')
    #plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()

    # xopt, fopt = pso(func2, lb, ub,debug=True)#f_ieqcons=con
    # print(xopt,fopt)
