import numpy as np
import pandas as pd

def re_comsol(bandwidth):
    print('需要重新仿真:')
    re_comsol = []
    for i in range(len(bandwidth)):
        if bandwidth[i] == 1000:

            re_comsol.append(i)
    print(len(re_comsol))
    np.savetxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/comsol模型/2.4.1仿真结果/2.4.1re_comsol.txt',np.array(re_comsol),fmt='%d')
    re_c = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/comsol模型/2.4.1仿真结果/2.4.1re_comsol.txt')
    print(re_c.shape)
#判断带宽是否存在
#不存在则记录下标，绘图查看，并重新仿真
S = []
a_max= []#峰值系数
f_max= []#峰值频率
bandwidth = []#带宽
for j in range(1024):
    name2 = j
    name1 = r'E:\毕设：空间盘绕/0.chapter_data_save/2chaper\comsol模型/2.4.1sobol/2.4.1_'
    name3 = '.txt'
    name = name1+str(name2)+name3
    a = np.loadtxt(name, encoding='UTF-8', skiprows=5)
    x = a[:, 0]
    y = a[:, 1]

    max_idx = np.argmax(y)
    a_y = y[max_idx]#系数
    a_x = x[max_idx]#频率

    a_max.append(a_y)
    f_max.append(a_x)

    daikuan = []
    for i in range(len(y)):
        if a_y<=0.21:
            if abs(y[i] - 0.5 * a_y) <= 0.005:
                daikuan.append(x[i])
        if a_y<=0.37:
            if abs(y[i] - 0.5 * a_y) <= 0.01:
                daikuan.append(x[i])
        else:
            if abs(y[i] - 0.5 * a_y) <= 0.025:
                daikuan.append(x[i])
    bandwidth.append(daikuan[-1] - daikuan[0])


s=pd.cut(a_max, [0,0.1,0.2,0.30,0.4,0.50,0.6,0.7,0.8,0.9,1])
print('系数区间频率',s.value_counts())
print(bandwidth)
np.savetxt(r'E:\毕设：空间盘绕/0.chapter_data_save/2chaper/comsol模型/2.4.1sobol/2.4.1峰值系数.txt',np.array(a_max),fmt='%.2f')
np.savetxt(r'E:\毕设：空间盘绕/0.chapter_data_save/2chaper/comsol模型/2.4.1sobol/2.4.1峰值频率.txt',np.array(f_max),fmt='%d')
np.savetxt(r'E:\毕设：空间盘绕/0.chapter_data_save/2chaper/comsol模型/2.4.1sobol/2.4.1相对带宽.txt',np.array(bandwidth),fmt='%d')
for i in range(len(bandwidth)):
    if bandwidth[i] >= 80:
        print(i,bandwidth[i])

print(bandwidth[848])
