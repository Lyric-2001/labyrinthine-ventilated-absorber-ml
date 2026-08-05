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
