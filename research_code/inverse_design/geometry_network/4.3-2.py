import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pylab import *
#支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

def picture(y, y_pre):
    plt.figure(figsize=(8, 5))#plt.figure(figsize=(8, 5))
    plt.plot(range(920, 1320, 2), y[220:420], label="true", linewidth=2)
    plt.plot(range(920, 1320, 2), y_pre[220:420], '--', label="predict", linewidth=2)
    plt.xlabel('frequency(Hz)', fontdict={'fontsize': 12})
    plt.ylabel('absorb', fontdict={'fontsize': 12})
    # plt.title("absorb")
    #  plt.yticks(size=12)
    #    plt.xticks(size=12)
    plt.legend()
    plt.show()
def search(test_data):

    test = test_data.reshape(150,571)
    print(test.shape)
    index = []
    for i in range(150):
        a = np.argmax(test[i])
        index.append(a)
    return index

def picture2(y):
    #=====================四张图片========================
    plt.figure(figsize=(16, 5))
    y1 = y[66*571:67*571]
    y_pre1 = np.loadtxt('E:/Graduation_project/1.chapter_data_save/4chapter/'
                        '4.3.4验证数据仿真结果.txt',
                        skiprows=5,encoding='utf-8')[:,1]
    plt.subplot(1, 2, 1)
    plt.plot(range(560, 960, 2), y1[40:240], label="true", linewidth=2)
    plt.plot(range(560, 960, 2), y_pre1[40:240], '--', label="predict", linewidth=2)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.legend()

    y2 = y[106 * 571:107 * 571]
    y_pre2 = np.loadtxt('E:/Graduation_project/1.chapter_data_save/4chapter/4.3.4验证数据仿真结果2.txt',
                        skiprows=5, encoding='utf-8')[:,1]
    plt.subplot(1, 2, 2)
    plt.plot(range(610, 1010, 2), y2[65:265], label="true", linewidth=2)
    plt.plot(range(610, 1010, 2), y_pre2[65:265], '--', label="predict", linewidth=2)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.legend()
    plt.show()
def picture3(y):
    #=====================四张图片========================
    plt.figure(figsize=(8, 5))
    y1 = y[47*571:48*571]
    y_pre1 = np.loadtxt('./4.3.4最大误差数据仿真结果-47.txt',
                        skiprows=5,encoding='utf-8')[:,1]
    plt.plot(range(640, 940, 2), y1[80:230], label="true", linewidth=2)
    plt.plot(range(640, 940, 2), y_pre1[80:230], '--', label="predict", linewidth=2)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.legend()
    plt.show()
def y_index(pre,test_data):
    print(pre.shape)

    y_pre = []
    index = search(test_data)
    for i in range(150):
        y_pre.append(pre[i*571+index[i]])
    return y_pre
def y_mean(pre):
    y_pre = pre.reshape(150, -1, 5)
    y_pre = np.mean(y_pre, axis=1)
    y_pre = np.round(y_pre, 1)
    return y_pre

if __name__ == "__main__":

    #=========================================
    test_data = np.loadtxt('./Latin_test_absorpara_model2.txt')
    print('test_data.shape', test_data.shape)
    x_test = test_data[:, 0:-5]
    y_test = test_data[:, -5:]
    pred = np.loadtxt('./y_pre_0.9917.txt')
    y = y_test[::571]
    pre_mean = y_mean(pred)
    print('r2_score=%.4f' % (r2_score(y, np.array(pre_mean))))
    #picture3(x_test[:,-1])
    print(y[47,:])
    #picture2(x_test[:,-1])
    print(y[66, :])
    #=========================================
    # m = 47
    # n = m + 1
    # picture(y_test[m * 571:n * 571], pred[m * 571:n * 571])
    # print(x_test.shape)
    # print(x_test[16*571],'\n',x_test[50*571],'\n',x_test[99*571],
    #       '\n',x_test[125*571],'\n',x_test[41*571])
    # picture22(y_test, pred, 16, 50, 99, 125)
    #===================================
    # a = np.array([[1,2,3],[1.2,2.5,3.4]])
    # b = np.array([[1.1,2.2,4],[1.22,2.51,3.45]])
    # print(a.shape)
    # error,MeanMax = MaxError(a,b)
    # print(error,MeanMax)
    # print(np.max(error))