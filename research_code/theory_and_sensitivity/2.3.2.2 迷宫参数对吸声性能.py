from MeanImped import Impedeance
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
from pylab import mpl
# 设置显示中⽂字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
def test1(n):
    w_i = 2 * 10 ** (-3)  # 每一个通道宽度
    H = 28 * 10 ** (-3)  ##具体含义看论文 通道高度
    W = 28 * 10 ** (-3)  # 通道长度

    L = n * w_i + (n - 1) * 10 ** (-3)  # 各个通道宽度并排的总长度

    f = int(343 / (4 * (W * n + (n) * 0.001)))
    # 频率计算的范围
    f_l = 200
    f_r = 1100 + 1
    f_interval = 5  # 频率扫描间隔
    print(f, f_l, f_r)
    # 调用类
    I = Impedeance(f_l, f_r, w_i, W, L, H, n, f_interval)
    a = I.A()
    x = range(f_l, f_r, f_interval)
    plt.figure(figsize=(10, 7))
    plt.plot(x, a)
    plt.show()
    np.savetxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.1 {}通道数.txt'.format(n), a)


def dk(a, a_y):
    daikuan = []
    for i in range(len(a)):
        if abs(a[i] - 0.5 * a_y) <= 0.02:
            daikuan.append(i)
    print(daikuan)
    print(daikuan[-1] - daikuan[0])


def number(name):
    f_l = 300
    f_r = 1080 + 1
    f_interval = 2  # 频率扫描间隔
    a1 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.1 结构B {}通道数.txt'.format(name[0]), encoding='UTF-8',
                    skiprows=5)
    a2 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.1 结构B {}通道数.txt'.format(name[1]), encoding='UTF-8',
                    skiprows=5)
    a3 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.1 结构B {}通道数.txt'.format(name[2]), encoding='UTF-8',
                    skiprows=5)
    a1 = a1[:, 1]
    a2 = a2[:, 1]
    a3 = a3[:, 1]

    plt.figure(figsize=(8, 5))
    x = range(f_l, f_r, f_interval)

    max_idx = np.argmax(a1)
    a1_x, a1_y = x[max_idx], a1[max_idx]
    dk(a1, a1_y)
    max_idx = np.argmax(a1)
    a1_x, a1_y = x[max_idx], a1[max_idx]
    plt.plot(x, a1, label='n=4', linewidth=2)
    plt.scatter(a1_x, a1_y, color='r', s=20)
    #plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x - 90, a1_y))

    max_idx = np.argmax(a2)
    a1_x, a1_y = x[max_idx], a2[max_idx]
    plt.scatter(a1_x, a1_y, color='r', s=20)
    #plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x + 10, a1_y))
    plt.plot(x, a2, label='n=5', linewidth=2)
    dk(a2, a1_y)

    max_idx = np.argmax(a3)
    a1_x, a1_y = x[max_idx], a3[max_idx]
    plt.scatter(a1_x, a1_y, color='r', s=20)
    #plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x + 10, a1_y))
    plt.plot(x, a3, label='n=6', linewidth=2)
    dk(a3, a1_y)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 13})
    plt.ylabel('吸声系数', fontdict={'fontsize': 13})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc='upper right')
    plt.savefig('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.1三通道总图.png')
    plt.show()


def wideth1(name):
    f_l = 300
    f_r = 530 + 1
    f_interval = 2  # 频率扫描间隔
    a1 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.2结构B宽度{}毫米.txt'.format(name[0]), encoding='UTF-8',
                    skiprows=5)
    a2 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.2结构B宽度{}毫米.txt'.format(name[1]), encoding='UTF-8',
                    skiprows=5)
    a3 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.2结构B宽度{}毫米.txt'.format(name[2]), encoding='UTF-8',
                    skiprows=5)
    a4 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.2结构B宽度{}毫米.txt'.format(name[3]), encoding='UTF-8',
                    skiprows=5)
    a1 = a1[:, 1]
    a2 = a2[:, 1]
    a3 = a3[:, 1]
    a4 = a4[:, 1]
    plt.figure(figsize=(8, 5))
    x = range(f_l, f_r, f_interval)

    max_idx = np.argmax(a1)
    a1_x, a1_y = x[max_idx], a1[max_idx]
    dk(a1, a1_y)
    max_idx = np.argmax(a1)
    a1_x, a1_y = x[max_idx], a1[max_idx]
    plt.plot(x, a1, label='w1=3mm', linewidth=2)
    plt.scatter(a1_x, a1_y, color='r', s=20)
    # plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x - 20, a1_y))

    max_idx = np.argmax(a2)
    a1_x, a1_y = x[max_idx], a2[max_idx]
    plt.scatter(a1_x, a1_y, color='r', s=20)
    # plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x + 10, a1_y))
    plt.plot(x, a2, label='w1=6mm', linewidth=2)
    dk(a2, a1_y)

    max_idx = np.argmax(a3)
    a1_x, a1_y = x[max_idx], a3[max_idx]
    plt.scatter(a1_x, a1_y, color='r', s=20)
    # plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x + 10, a1_y))
    plt.plot(x, a3, label='w1=9mm', linewidth=2)
    dk(a3, a1_y)

    max_idx = np.argmax(a4)
    a1_x, a1_y = x[max_idx], a4[max_idx]
    plt.scatter(a1_x, a1_y, color='r', s=20)
    # plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x + 10, a1_y))
    plt.plot(x, a4, label='w1=12mm')
    dk(a4, a1_y)

    plt.xlabel('频率(Hz)', fontdict={'fontsize': 13})
    plt.ylabel('吸声系数', fontdict={'fontsize': 13})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc='upper right')
    plt.savefig('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.2宽度变化总图.png')
    plt.show()


def wideth2(name):
    f_l = 400
    f_r = 850 + 1
    f_interval = 2  # 频率扫描间隔
    a1 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2W-{}.txt'.format(name[0]), encoding='UTF-8',
                    skiprows=5)
    a2 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2W-{}.txt'.format(name[1]), encoding='UTF-8',
                    skiprows=5)
    a3 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2W-{}.txt'.format(name[2]), encoding='UTF-8',
                    skiprows=5)
    a1 = a1[:, 1]
    a2 = a2[:, 1]
    a3 = a3[:, 1]

    plt.figure(figsize=(10, 6))
    x = range(f_l, f_r, f_interval)

    max_idx = np.argmax(a1)
    a1_x, a1_y = x[max_idx], a1[max_idx]
    dk(a1, a1_y)
    plt.plot(x, a1, label='W-1', linewidth=2)
    plt.scatter(a1_x, a1_y, color='r', s=20)
    # plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x - 20, a1_y))

    max_idx = np.argmax(a2)
    a1_x, a1_y = x[max_idx], a2[max_idx]
    plt.scatter(a1_x, a1_y, color='r', s=20)
    # plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x + 10, a1_y))
    plt.plot(x, a2, label='W-2', linewidth=2)
    dk(a2, a1_y)

    max_idx = np.argmax(a3)
    a1_x, a1_y = x[max_idx], a3[max_idx]
    plt.scatter(a1_x, a1_y, color='r', s=20)
    # plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x + 10, a1_y))
    plt.plot(x, a3, label='W-3', linewidth=2)
    dk(a3, a1_y)

    plt.xlabel('频率(Hz)', fontdict={'fontsize': 13})
    plt.ylabel('吸声系数', fontdict={'fontsize': 13})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc='upper right')
    plt.savefig('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.2宽度变化总图2.png')
    plt.show()

def high(name):
    f_l = 500
    f_r = 750 + 1
    f_interval = 1  # 频率扫描间隔
    a1 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.3高{}毫米.txt'.format(name[0]), encoding='UTF-8',
                    skiprows=5)
    a2 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.3高{}毫米.txt'.format(name[1]), encoding='UTF-8',
                    skiprows=5)
    a3 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.3高{}毫米.txt'.format(name[2]), encoding='UTF-8',
                    skiprows=5)
    a4 = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.3高{}毫米.txt'.format(name[3]), encoding='UTF-8',
                    skiprows=5)
    a1 = a1[:, 1]
    a2 = a2[:, 1]
    a3 = a3[:, 1]
    a4 = a4[:, 1]
    plt.figure(figsize=(8, 5))
    x = range(f_l, f_r, f_interval)

    max_idx = np.argmax(a1)
    a1_x, a1_y = x[max_idx], a1[max_idx]
    dk(a1, a1_y)
    max_idx = np.argmax(a1)
    a1_x, a1_y = x[max_idx], a1[max_idx]
    plt.plot(x, a1, label='W=10mm', linewidth=1)
    plt.scatter(a1_x, a1_y, color='r', s=20)
    # plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x - 20, a1_y))

    max_idx = np.argmax(a2)
    a1_x, a1_y = x[max_idx], a2[max_idx]
    plt.scatter(a1_x, a1_y, color='r', s=20)
    # plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x + 10, a1_y))
    plt.plot(x, a2, label='W=20mm', linewidth=1)
    dk(a2, a1_y)

    max_idx = np.argmax(a3)
    a1_x, a1_y = x[max_idx], a3[max_idx]
    plt.scatter(a1_x, a1_y, color='r', s=20)
    # plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x + 10, a1_y))
    plt.plot(x, a3, label='W=30mm', linewidth=1)
    dk(a3, a1_y)

    max_idx = np.argmax(a4)
    a1_x, a1_y = x[max_idx], a4[max_idx]
    plt.scatter(a1_x, a1_y, color='r', s=20)
    # plt.annotate(f'({a1_x}, {a1_y:.2f})', xy=(a1_x, a1_y), xytext=(a1_x + 10, a1_y))
    plt.plot(x, a4, label='W=40mm', linewidth=1)
    dk(a4, a1_y)

    plt.xlabel('频率(Hz)', fontdict={'fontsize': 13})
    plt.ylabel('吸声系数', fontdict={'fontsize': 13})
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.rcParams.update({'font.size': 12})
    plt.legend(loc='upper right')
    plt.savefig('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.2.3高度变化总图2.png')
    plt.show()

if __name__ == "__main__":
    # 通道数量
    # name = ['3','5','7']
    # number(name)

    # 通道宽度-单个改变
    # name = ['3', '6', '9','12']
    # wideth1(name)

    # 通道宽度-多个变化
    # name = ['1', '2', '3']
    # wideth2(name)

    # 通道高度
    name = ['10', '20', '30','40']
    high(name)