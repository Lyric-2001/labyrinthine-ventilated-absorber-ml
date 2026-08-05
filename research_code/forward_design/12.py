import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd
from pylab import *
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
#支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']



def val():
    data = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/3chapter/test_ParaHzAbsorb.txt')
    print(data.shape)
    index = []
    for i in range(5):
        n1 = np.random.choice(149)
        index.append(n1)
    index.sort()
    print(index)
    da = []
    for i in index:
        l = 0+571*(i)
        r = 571*(i+1)
        da.append(data[l:r])
    da = (np.array(da)).reshape(-1,7)
    print(da.shape)
    print(da[0:20])
    np.savetxt('E:/毕设：空间盘绕/0.chapter_data_save/3chapter/3.3.3val.txt',da)

def pic():
    val = np.loadtxt('./CNN998_val_predict.txt')
    y_pred = val.reshape(5,-1,1)
    print(y_pred.shape)

    test = np.loadtxt('./3.3.3val.txt')[:,-1]
    y_test = test.reshape(5, -1, 1)
    print(y_test.shape)
    plt.figure(figsize=(16, 5))
    # 标签字体设置
    plt.subplot(1, 2, 1)
    plt.plot(range(650, 950, 2), y_test[3, 85:235], label="true", linewidth=2)
    plt.plot(range(650, 950, 2), y_pred[3, 85:235], '--', label="predict", linewidth=2)
    plt.title("(c)", fontdict={'fontsize': 12})
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.yticks(size=11)
    plt.xticks(size=11)
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(range(750, 1050, 2), y_test[4, 135:285], label="true", linewidth=2)
    plt.plot(range(750, 1050, 2), y_pred[4, 135:285], '--',label="predict", linewidth=2)
    plt.title("(d)", fontdict={'fontsize': 12})
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.yticks(size=11)
    plt.xticks(size=11)
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc='upper right')

    # plt.subplot(1, 3, 3)
    # plt.plot(range(890, 1250, 2), y_test[2, 205:385], label="true", linewidth=2)
    # plt.plot(range(890, 1250, 2), y_pred[2, 205:385], '--', label="predict", linewidth=2)
    # plt.title("(c)", fontdict={'fontsize': 12})
    # plt.xlabel('frequency(Hz)', fontdict={'fontsize': 12})
    # plt.ylabel('absorb', fontdict={'fontsize': 12})
    # plt.rcParams.update({'font.size': 12})
    # plt.legend(loc='upper right')
    plt.show()
def rand_val():
    print(np.random.randint(180, 280))
    data = []
    for i in range(4):
        a = np.random.randint(180, 280) / 10
        a2 = np.random.randint(40, 70) / 10
        a3 = np.random.randint(40, 70) / 10
        a4 = np.random.randint(40, 70) / 10
        a5 = np.random.randint(40, 70) / 10
        data.append([a, a2, a3, a4, a5])
    print(data)
    da = np.array(data)
    print(da.shape)
    np.savetxt('./rand_val.txt', da,fmt='%.1f')
def rand_mse():
    da = []
    for i in range(4):

        y = np.loadtxt('E:/毕设：空间盘绕/2 代码/graduation_project/chapter_3/rand_val/3.1_{}.txt'.format(i),
                       skiprows=5,encoding='utf-8')[:,1]
        da.append(y)
    da = np.array(da).reshape(-1,1)
    print(da.shape)
    pre = np.loadtxt('E:/毕设：空间盘绕/2 代码/graduation_project/chapter_3/rand_val/CNN998_rand_predict.txt')
    print(pre.shape)
    print(r2_score(da,pre),'\t',mean_squared_error(da,pre))
def pic2():
    val = np.loadtxt('./CNN998_val_predict.txt')
    y_pred = val.reshape(5,-1,1)
    print(y_pred.shape)

    test = np.loadtxt('./3.3.3val.txt')[:,-1]
    y_test = test.reshape(5, -1, 1)
    print(y_test.shape)
    plt.figure(figsize=(16, 5))
    # 标签字体设置
    plt.subplot(1, 2, 1)
    plt.plot(range(680, 980, 2), y_test[0, 100:250], label="true", linewidth=2)
    plt.plot(range(680, 980, 2), y_pred[0, 100:250], '--', label="predict", linewidth=2)
    plt.title("(a)", fontdict={'fontsize': 12})
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.yticks(size=11)
    plt.xticks(size=11)
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(range(700, 1000, 2), y_test[1, 110:260], label="true", linewidth=2)
    plt.plot(range(700, 1000, 2), y_pred[1, 110:260], '--',label="predict", linewidth=2)
    plt.title("(b)", fontdict={'fontsize': 12})
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.yticks(size=11)
    plt.xticks(size=11)
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc='upper right')

    # plt.subplot(1, 3, 3)
    # plt.plot(range(920, 1220, 2), y_test[2, 220:370], label="true", linewidth=2)
    # plt.plot(range(920, 1220, 2), y_pred[2, 220:370], '--', label="predict", linewidth=2)
    # plt.title("(c)", fontdict={'fontsize': 14})
    # plt.xlabel('频率(Hz)', fontdict={'fontsize': 16})
    # plt.ylabel('吸声系数', fontdict={'fontsize': 16})
    # plt.yticks(size=14)
    # plt.xticks(size=14)
    # plt.rcParams.update({'font.size': 14})
    # plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    pic2()
    #rand_val()
    # rand_mse()
    # 导入所需的库


