import numpy as np
from FCmodel import FCmodel ,MyDataset,min_max
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pyswarms as ps
import torch.nn as nn
import random
class CNN_FC(nn.Module):
    def __init__(self, inputsize, n_filter, conv_input_channel, conv_output_channel,
                 hidden_size, num_layers, output_size, dropout, training):
        super(CNN_FC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv_output_channel = conv_output_channel
        self.dropout = nn.Dropout(dropout)
        self.training = training
        # 卷积核
        self.filter_sizes = [(x) for x in range(1, n_filter)]  # n-gram window
        # 3 个卷积层，卷积核大小分别为 [2,1], [3,1], [4,1]
        self.conv1 = nn.Sequential(
            nn.Conv1d(conv_input_channel, conv_output_channel,
                      kernel_size=self.filter_sizes[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=inputsize - self.filter_sizes[0] + 1))

        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_input_channel, conv_output_channel,
                      kernel_size=self.filter_sizes[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=inputsize - self.filter_sizes[1] + 1))

        self.fc = nn.Sequential(nn.Linear(inputsize + 2 * conv_output_channel, 64), nn.ReLU(),
                                nn.Linear(in_features=64, out_features=512), nn.ReLU(),
                                nn.Linear(in_features=512, out_features=128), nn.ReLU(),
                                nn.Linear(in_features=128, out_features=32), nn.ReLU(),
                                nn.Linear(32, output_size))  # 16时，达到0.95，batch = 9096
        self.training = training

    def forward(self, x):
        x0 = x.unsqueeze(1)
        x1 = self.conv1(x0).permute(0, 2, 1)
        x2 = self.conv2(x0).permute(0, 2, 1)
        x = torch.cat([x0, x1, x2], dim=-1).squeeze()
        out = self.fc(x)
        return out
#===============GPU==================
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")
print("Use cuda: %s, gpu id: %d.", use_cuda, gpu)

#=====================训练集model1==============
data = np.loadtxt('./latin_train_absor_para.txt')
print('data.shape', data.shape)
x_train = data[:, 0:8]
y_train = data[:, 8:9]
x_ptp_value = x_train.ptp(axis=0)
x_min_value = x_train.min(axis=0)

y_ptp_value = y_train.ptp(axis=0)
y_min_value = y_train.min(axis=0)
HZ= np.array(range(480,1622,2))
#==============加载模型model1======================
output_size = 1
training = False
model = FCmodel(output_size, training)
model.to(device)
model.load_state_dict(torch.load('./4.2model1_0.9997_r0_6e-6.pth'))

#=================加载模型model2=========================
data2 = np.loadtxt('E:/Graduation_project/2_code/graduation_project/'
                   'chapter_4/inverse_model2/Latin_train_absorpara_model2.txt')
print('data2.shape', data2.shape)
x_train2 = data2[:, 0:-5]
y_train2 = data2[:, -5:]
x_ptp_value2 = x_train2.ptp(axis=0)
x_min_value2 = x_train2.min(axis=0)
y_ptp_value2 = y_train2.ptp(axis=0)
y_min_value2 = y_train2.min(axis=0)

filter_size, conv_input_channel, conv_output_channel = 3, 1, 2  # cnn
hidden_size, num_layers = 100, 2  # LSTM
inputsize = x_train2.shape[-1]
output_size = len(y_train2[0])  # Fnn
dropout = 0.1
training = False
model2 = CNN_FC(inputsize, filter_size, conv_input_channel, conv_output_channel,
                 hidden_size, num_layers, output_size, dropout, training)
model2.load_state_dict(torch.load('E:/Graduation_project/2_code/graduation_project/'
                   'chapter_4/inverse_model2/Latin_model2_0.9917_卷积2.pth'))
model2.to(device)


#=======================# 定义适应度函数=================
def segment(x):
    num = 13
    leng = x.shape[1]//5
    #print('leng',leng)
    arr = []
    for i in range(leng):
        peak_p = (x[:, i*5]).reshape(-1, 1)
        l = (x[:, i*5] - num).reshape(-1, 1)
        r = (x[:, i*5] + num).reshape(-1, 1)
        x1 = np.concatenate((l, peak_p, r, x[:, i*5+1:(i+1)*5]), axis=1)
        #print('x1.shape',x1.shape)
        arr.append(x1)
    return np.array(arr)
def model_pred(x_test):
    y_test = np.array([i for i in range(x_test.shape[0])])
    x_test_scaler = (x_test - x_min_value) / x_ptp_value
    test_dataset = MyDataset(x_test_scaler, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    y_pred = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_dataloader, 1):
            data_x = x.to(torch.float32).to(device)
            y_pred.append(model(data_x).reshape(-1, 1).cpu())
    pred = np.array(y_pred) * y_ptp_value + y_min_value
    y_pre = pred.reshape(-1, 1)
    #print('model_pred(x)',y_pre.shape)
    return y_pre
def model2_pred(x_test):
    y_test = np.array([0 for i in range(x_test.shape[0])])
    x_test_scaler = (x_test - x_min_value2) / x_ptp_value2
    #print('x_test_scaler.shape',x_test_scaler.shape)
    test_dataset = MyDataset(x_test_scaler, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    y_pred = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_dataloader, 1):
            data_x = x.to(torch.float32).to(device)
            y_pred.append(model2(data_x).reshape(-1, 1).cpu())
    #print('pred', np.array(y_pred).shape)
    pred = np.array(y_pred).reshape(-1,5) * y_ptp_value2 + y_min_value2
    #print('pred', pred.shape)
    y_pre = pred.reshape(-1, 5)
    #print('model_pred(x)',y_pre.shape)
    return y_pre

def model_fun(x):
    arr= np.round(x, 1)#(20*5)
    #print('model_fun(x): arr.shape', arr.shape)
    #=======补足左右频率======random.choice
    l =np.round(arr[:, 0],0)
    r = np.round(arr[:, 2], 0)
    for i in range(len(l)):
        r[i] = random.choice(l_r_hz[str(int(l[i]))])
    l =np.round(arr[:, 1],0)-l*2
    r = np.round(arr[:, 1],0) +r
    #print('l',l)
    peak= np.round(arr[:, 1],0).reshape(-1, 1)
    absorb_obj = np.round(x[:,3], 3)
    #print('r',r)
    arr = np.concatenate((l.reshape(-1,1), peak, r.reshape(-1,1), absorb_obj.reshape(-1,1),arr[:, 4:]), axis=1)
    #print('arr.shape',arr.shape,arr)
    # np.savetxt('./para_arr.txt')
    #=====================model1数据处理==================
    x_test = []
    for i in range(arr.shape[0]):
        new_para = arr[i, :].reshape(1, -1).repeat(571, axis=0)
        parameter = np.concatenate((new_para, HZ.reshape(-1, 1)), axis=1)
        x_test.append(parameter)
    x_test = np.array(x_test).reshape(-1, 8)#x_test (1142, 8)
    #print('x_test',x_test.shape)
    absorb = model_pred(x_test)##所求曲线
    #print('absorb', absorb.shape,np.max(absorb))#(1142, 1)
    #np.savetxt('./pso_absorb.txt',absorb)
    #===================model2=====================
    arr2 = np.concatenate((x_test, absorb), axis=1)
    #print('arr ', arr.shape)
    para = (model2_pred(arr2)).reshape(-1,571,5)
    para = np.round(np.mean(para,axis=1),1)

    #print('para.shape',para.shape,para[0:2])
    for i in range(para.shape[0]):
        if para[i,1]<4 or para[i,2]<4 or para[i,3]<4 or para[i,4]<4:
            absorb[i*571:(i+1)*571]=0
        elif para[i,1]>7 or para[i,2]>7 or para[i,3]>7 or para[i,4]>7:
            absorb[i*571:(i+1)*571]=0

    #result里面是几个列表，每个列表是一个二维矩阵（20*n*571）n=dim//5
    #求对应的每一个组合的和值
    absorb = absorb.reshape(x.shape[0], 571)
    #print('absorb.shape', absorb.shape)
    y_pre = []
    for i in range(absorb.shape[0]):
        a = absorb[i].reshape(-1,1) + y_obj
        y_pre.append(a)

    y_pre = np.array(y_pre).reshape(x.shape[0], 571)
    for i in range(y_pre.shape[0]):
        for j in range(y_pre.shape[1]):
            if y_pre[i][j]>1:
                y_pre[i][j] = 1
    cost = np.sum(y_pre>=1 ,axis=1)
    print('cost.shape:', np.array(cost).shape,cost)
    print('best_arr',arr[np.argmax(cost)])
    print('best_para',para[np.argmax(cost)])
    print('=========================================================')
    return (np.array(cost).ravel())*(-1)

def predict():
    x_test = []
    arr = np.array([9.780e+02, 1.000e+03, 1.020e+03, 9.460e-01, 2.230e+01, 6.900e+00, 1.950e+01]).reshape(1, -1)
    for i in range(arr.shape[0]):
        new_para = arr[i, :].reshape(1, -1).repeat(571, axis=0)
        parameter = np.concatenate((new_para, HZ.reshape(-1, 1)), axis=1)
        x_test.append(parameter)
    x_test = np.array(x_test).reshape(-1, 8)  # x_test (1142, 8)
    print(x_test)
    # print('x_test',x_test.shape)
    absorb = model_pred(x_test)  ##所求曲线
    np.savetxt('E:/Graduation_project/1000-1.txt', absorb)
#====================目标曲线=====================
#57 964
#y_obj= (((np.loadtxt('./latin_test_absor_para.txt'))[57*571:58*571])[:,-1]).reshape(-1,1)
#y_obj= ((np.loadtxt('E:/Graduation_project/964-2.txt',skiprows=5,encoding='utf-8'))[:,1]).reshape(-1,1)
y_obj= ((np.loadtxt('E:/Graduation_project/4.4 700-900六结构/790-5.txt',
                    skiprows=5,encoding='utf-8'))[:,1]).reshape(-1,1)
#y_obj= np.array([0*i for i in range(0,571)]).reshape(-1,1)
l_r_hz = {'6':[14,16],'7':[12,14,16,18],'8':[14,16,18,20],
          '9':[16,18,20,22],'10':[18,20,22],'11':[20,22]}
if __name__ == "__main__":
    #目标吸声曲线
    #==========参数变化的上下限===========================
    n = 0
    #l_p,p_p,r_p,p_a,W,w1,S
    #左右频率根据峰值频率，设置为左右各16HZ；峰值系数希望在0.9-0.95，W设置为18-28；w14-7;S为16-28
    #变量：p_p，p_a，W
    constraints = (np.array([5.5,   815,  0.5,  0.87, 18, 4, 15]),
                   np.array([11.4,  820, 4.49,  0.93, 28, 7, 20]))
    #=======================例子群算法=====================
    options = {'c1': 2, 'c2': 2, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=7, options=options,bounds=constraints)
    cost, pos = optimizer.optimize(model_fun, iters=40)

    #=================保存最最优位置和吸 声系数==================
    print('最优位置：', pos, '最优适应度：', cost)
    pos_absorb = np.concatenate((pos.reshape(1,-1),np.array([cost]).reshape(1,-1)),axis=1)
    np.savetxt('./3.5.3pos_absorb{}.txt'.format(n), pos_absorb)
    #==============保存损失==================
    #print(type(optimizer.cost_history),'\n',len(optimizer.cost_history))
    coss = optimizer.cost_history
    np.savetxt('./3.5.3pso_cost{}.txt'.format(n),coss)
    #f.close()
      #===============绘制损失曲线===============
    plt.figure(figsize=(8, 5))
    plt.plot(range(1,len(coss)+1),coss, linewidth=2)
    plt.title("Cost", fontdict={'fontsize': 12})
    plt.xlabel('epochs', fontdict={'fontsize': 12})
    plt.ylabel('value', fontdict={'fontsize': 12})
    plt.yticks(size=12)
    plt.xticks(size=12)
    # plt.rcParams.update({'font.size': 12})
    # plt.legend(loc='upper right')
    # plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()

    # xopt, fopt = pso(func2, lb, ub,debug=True)#f_ieqcons=con
    # print(xopt,fopt)
# [1.11821164e+01 9.27814244e+02 2.89939109e+00 9.09052418e-01 2.13784858e+01 4.56883144e+00 1.71692900e+01]




