import numpy as np
import matplotlib.pyplot as plt
from pylab import *
#支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
def picture_3():
    a = np.loadtxt('E:/Graduation_project/1.chapter_data_save/3chapter/Untitled.txt',
                   skiprows=5, encoding='utf-8')
    b = np.loadtxt('E:/Graduation_project/1.chapter_data_save/3chapter/4-2重新采样/virtual_y.txt')
    plt.figure(figsize=(10, 8))
    plt.plot(range(910, 1450, 2), a[215:485, 1], label='数值解')
    plt.plot(range(910, 1450, 2), b[0, 215:485], '--', label='数值解')
    # plt.plot(range(f_l, f_r, f_interval), c[:, 1],label='通风数值解',color='green')

    # plt.savefig('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.1结构{}验证图.png'.format(name))
    plt.show()

def MaxError(y_true,y_pre):
    #y_true,y_pre:n*571
    error = []
    for i in range(y_true.shape[0]):
        a = (y_true[i,:]-y_pre[i,:])
        a = np.max(abs(a))
        error.append(a)
    MeanMax = (np.sum(error)) / y_true.shape[0]
    print('np.argmax(error)', np.argmax(error))
    return error,MeanMax

def picture(y, y_pre):
    plt.figure(figsize=(8, 5))    #plt.figure(figsize=(8, 5))
    plt.plot(range(690, 990, 2), y[105:255], label="true", linewidth=2)
    plt.plot(range(690, 990, 2), y_pre[105:255], '--', label="predict", linewidth=2)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    # plt.title("absorb")
    #  plt.yticks(size=12)
    #    plt.xticks(size=12)
    plt.legend()
    plt.show()
def picture2(y, y_pre,m1,m2,m3,m4):
    #=====================四张图片========================
    plt.figure(figsize=(16, 10))
    n = m1 + 1
    y1 = y[m1 * 571:n * 571]
    y_pre1 = y_pre[m1 * 571:n * 571]

    plt.subplot(2, 2, 1)
    plt.plot(range(780, 1080, 2), y1[150:300], label="true", linewidth=2)
    plt.plot(range(780, 1080, 2), y1[150:300], '--', label="predict", linewidth=2)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.legend()

    n = m2 + 1
    y2 = y[m2 * 571:n * 571]
    y_pre2 = y_pre[m2 * 571:n * 571]
    plt.subplot(2, 2, 2)
    plt.plot(range(670, 970, 2), y2[95:245], label="true", linewidth=2)
    plt.plot(range(670, 970, 2), y_pre2[95:245], '--', label="predict", linewidth=2)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.legend()

    n = m3 + 1
    y3 = y[m3 * 571:n * 571]
    y_pre3 = y_pre[m3 * 571:n * 571]
    plt.subplot(2, 2, 3)
    plt.plot(range(610, 910, 2), y3[65:215], label="true", linewidth=2)
    plt.plot(range(610, 910, 2), y_pre3[65:215], '--', label="predict", linewidth=2)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.legend()

    n = m4 + 1
    y4 = y[m4 * 571:n * 571]
    y_pre4 = y_pre[m4 * 571:n * 571]
    plt.subplot(2, 2, 4)
    plt.plot(range(910, 1210, 2), y4[215:365], label="true", linewidth=2)
    plt.plot(range(910, 1210, 2), y_pre4[215:365], '--', label="predict", linewidth=2)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.legend()

    plt.show()



if __name__ == "__main__":

    test_data = np.loadtxt('latin_test_absor_para.txt')
    x_test = test_data[:, 0:-1]
    y_test = test_data[:, -1:]
    print(x_test.shape)
    pred = (np.loadtxt('./y_pre_0.9997_2.txt')).reshape(-1, 1)
    print('test_data.shape', y_test.shape, pred.shape)
    m =0
    n = m + 1
    #picture(y_test[m * 571:n * 571], pred[m * 571:n * 571])
    print('x_test[175*571]',x_test[0*571])
    print(x_test[13*571],'\n',x_test[48*571],'\n',x_test[109*571],
          '\n',x_test[140*571],)
    #picture2(y_test, pred, 13, 48, 109, 140)
    #===================================
    # a = np.array([[1,2,3],[1.2,2.5,3.4]])
    # b = np.array([[1.1,2.2,4],[1.22,2.51,3.45]])
    # print(a.shape)
    error,MeanMax = MaxError(y_test,pred)
    #print(error,MeanMax)
    print(np.max(error),len(error))
    print(y_test[175],pred[175])
