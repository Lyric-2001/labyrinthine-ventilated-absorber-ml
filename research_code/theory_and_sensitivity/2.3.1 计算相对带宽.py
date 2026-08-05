import numpy as np
name = 100
a = np.loadtxt(r'E:/毕设：空间盘绕/0.chapter_data_save/2chaper\comsol模型/2.4.1仿真结果/2.4.1_{}.txt'.format(name)
               , encoding='UTF-8', skiprows=5)[:,1]
max_idx = np.argmax(a)
a_y = a[max_idx]
daikuan = []
for i in range(len(a)):
    if abs(a[i] - 0.5 * a_y)<=0.01:
        daikuan.append(i)
print(daikuan)
print(daikuan[-1]-daikuan[0])
