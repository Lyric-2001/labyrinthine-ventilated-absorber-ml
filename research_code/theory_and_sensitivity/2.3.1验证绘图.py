import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pylab import mpl
# from proplot import rc
# # 统一设置轴刻度标签的字体大小
# rc['tick.labelsize'] = 12
# # # 统一设置xy轴名称的字体大小
# rc["axes.labelsize"] = 15
# # # 统一设置轴刻度标签的字体粗细
# rc["axes.labelweight"] = "light"
# # # 统一设置xy轴名称的字体粗细
# rc["tick.labelweight"] = "bold"


# 设置显示中⽂字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

# 哪里需要显示中文就在哪里设置
name = 'A'
if name == 'B':
    f_l = 135
    f_r = 696
else:
    f_l = 260
    f_r = 750 + 1
f_interval = 2  # 频率扫描间隔

a = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.1结构{}理论解.txt'.format(name))
b = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.1结构{}数值仿真解.txt'.format(name), encoding='UTF-8', skiprows=5)
c = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.1结构{}通风圆盘.txt'.format(name), encoding='UTF-8', skiprows=5)
plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
# 绘制吸声曲线
n2 = 5
plt.scatter(range(f_l, f_r, f_interval*n2), [a[i] for i in range(0,len(a),n2)], marker='o', c='none', edgecolors='r',label='理论解')
plt.plot(range(f_l, f_r, f_interval), b[:, 1],label='数值解', linewidth=2)
plt.plot(range(f_l, f_r, f_interval), c[:, 1],label='通风数值解', linewidth=2)
plt.xlabel('频率(Hz)', fontdict={'fontsize': 17})
plt.ylabel('吸声系数', fontdict={'fontsize': 17})
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.rcParams.update({'font.size': 16})
plt.legend(loc='upper right')
#plt.savefig('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.1结构{}验证图.png'.format(name))


name = 'B'
if name == 'B':
    f_l = 135
    f_r = 696
else:
    f_l = 260
    f_r = 750 + 1
f_interval = 2  # 频率扫描间隔
n2 = 10
f_interval =1
a = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.1结构{}理论解.txt'.format(name))
b = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.1结构{}数值仿真解.txt'.format(name), encoding='UTF-8', skiprows=5)
c = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.1结构{}通风圆盘.txt'.format(name), encoding='UTF-8', skiprows=5)

plt.subplot(1, 2, 2)
plt.scatter(range(f_l, f_r, f_interval*n2), [a[i] for i in range(0,len(a),n2)], marker='o', c='none', edgecolors='r',label='理论解')
plt.plot(range(f_l, f_r, f_interval), b[:, 1],label='数值解', linewidth=2)
plt.plot(range(f_l, f_r, f_interval), c[:, 1],label='通风数值解', linewidth=2)
plt.xlabel('频率(Hz)', fontdict={'fontsize': 17})
plt.ylabel('吸声系数', fontdict={'fontsize': 17})
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.rcParams.update({'font.size': 16})
plt.legend(loc='upper right')
plt.show()
