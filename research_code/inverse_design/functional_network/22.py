import numpy as np
import random
import matplotlib.pyplot as plt
from pylab import *
#支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

def pic_absorb_single():
    plt.figure(figsize=(8, 5))
    data = np.loadtxt('E:/Graduation_project/2_code/graduation_project/chapter_4/inverse_model1/860-988耦合曲线.txt', skiprows=5, encoding='utf-8')
    data2 = np.loadtxt('E:/Graduation_project/2_code/graduation_project/chapter_4/inverse_model1/988.txt', skiprows=5, encoding='utf-8')
    data3 = np.loadtxt('E:/Graduation_project/2_code/graduation_project/chapter_4/inverse_model1/', skiprows=5, encoding='utf-8')
    hz = data[:, 0]
    y_pred = data[:, 1]

    hz2 = data2[:, 0]
    y_pred2 = data2[:, 1]
    hz3 = data3[:, 0]
    y_pred3 = data3[:, 1]
    # 标签字体设置range(1000, 1551, 2)
    plt.plot(hz[160:400], y_pred[160:400], linewidth=2)
    plt.plot(hz2[160:400], y_pred2[160:400],'--', linewidth=1)
    plt.plot(hz3[160:400], y_pred3[160:400], '--', linewidth=1)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.show()
def pic_absorb_single2():
    plt.figure(figsize=(8, 5),dpi=160)
    data = np.loadtxt('E:/Graduation_project/4.4六结构1/六结构总吸声曲线2.txt', skiprows=5, encoding='utf-8')
    data2 = np.loadtxt('E:/Graduation_project/4.4六结构1/Untitled1.txt', skiprows=5, encoding='utf-8')
    data3 = np.loadtxt('E:/Graduation_project/4.4六结构1/Untitled2.txt', skiprows=5, encoding='utf-8')
    data4 = np.loadtxt('E:/Graduation_project/4.4六结构1/Untitled3.txt', skiprows=5, encoding='utf-8')
    data5 = np.loadtxt('E:/Graduation_project/4.4六结构1/Untitled4.txt', skiprows=5, encoding='utf-8')
    data6 = np.loadtxt('E:/Graduation_project/4.4六结构1/Untitled5.txt', skiprows=5, encoding='utf-8')
    data7 = np.loadtxt('E:/Graduation_project/4.4六结构1/Untitled6.txt', skiprows=5, encoding='utf-8')

    hz = data[:, 0]
    y_pred = data[:, 1]

    hz2 = data2[:, 0]
    y_pred2 = data2[:, 1]
    hz3 = data3[:, 0]
    y_pred3 = data3[:, 1]
    hz4 = data4[:, 0]
    y_pred4 = data4[:, 1]
    hz5 = data5[:, 0]
    y_pred5 = data5[:, 1]
    hz6 = data6[:, 0]
    y_pred6 = data6[:, 1]
    hz7 = data7[:, 0]
    y_pred7 = data7[:, 1]
    # 标签字体设置range(1000, 1551, 2)
    plt.plot(hz[160:400], y_pred[160:400], linewidth=2)
    plt.plot(hz2[160:400], y_pred2[160:400],'--', linewidth=1)
    plt.plot(hz3[160:400], y_pred3[160:400], '--', linewidth=1)
    plt.plot(hz4[160:400], y_pred4[160:400], '--', linewidth=1)
    plt.plot(hz5[160:400], y_pred5[160:400], '--', linewidth=1)
    plt.plot(hz6[160:400], y_pred6[160:400], '--', linewidth=1)
    plt.plot(hz7[160:400], y_pred7[160:400], '--', linewidth=1)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 22})
    plt.ylabel('吸声系数', fontdict={'fontsize': 22})
    plt.yticks(size=22)
    plt.xticks(size=22)
    plt.show()
def pic_absorb():
    y_true1 = np.loadtxt('E:/Graduation_project/验证逆向设计网络1000-1.txt',                  skiprows=5, encoding='utf-8')
    x1 = y_true1[:, 0]
    y1 = y_true1[:, 1]
    y_pred1 = np.loadtxt('E:/Graduation_project/1000-1.txt')
    x_pred1 = range(480, 1622, 2)

    y_true2 = np.loadtxt('E:/Graduation_project/验证逆向设计网络1000-2.txt', skiprows=5, encoding='utf-8')
    x2 = y_true2[:, 0]
    y2 = y_true2[:, 1]
    y_pred2 = np.loadtxt('E:/Graduation_project/1000-2.txt')
    x_pred2 = range(480, 1622, 2)

    y_pred2 = np.loadtxt('./4.5_960-2hz.txt', skiprows=5, encoding='utf-8')

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    # 标签字体设置range(1000, 1551, 2)
    plt.plot(x1, y1, linewidth=2)
    plt.plot(x_pred1[185:335], y_pred1[185:335], '--', linewidth=2)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.yticks(size=12)
    plt.xticks(size=12)

    plt.subplot(1,2,2)
    # 标签字体设置range(1000, 1551, 2)
    plt.plot(x2, y2, linewidth=2)
    plt.plot(x_pred2[185:335], y_pred2[185:335], '--', linewidth=2)

    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.show()
def pic_cost():
    cost = np.loadtxt('E:/Graduation_project/2_code/graduation_project/'
                   'chapter_4/inverse_model1/4.3pso_cost0.txt')*(-1)
    plt.figure(figsize=(8, 5))
    #plt.subplot(1,2,1)
    plt.plot(range(1, len(cost) + 1), cost, linewidth=2)

    #plt.title("(a)", fontdict={'fontsize': 12})
    plt.xlabel('Epochs', fontdict={'fontsize': 14})
    plt.ylabel('Cost', fontdict={'fontsize': 14})
    plt.yticks(size=12)
    plt.xticks(size=12)
    #plt.rcParams.update({'font.size': 12})
    #plt.legend(loc='upper right')
    # plt.subplot(1, 2, 2)
    # cost2 = np.loadtxt('./4.4pso_cost0-2.txt') * (-1)
    # plt.plot(range(1, len(cost2) + 1), cost2, linewidth=2)
    # plt.title("(a)", fontdict={'fontsize': 12})
    # plt.xlabel('Epochs', fontdict={'fontsize': 12})
    # plt.ylabel('Value', fontdict={'fontsize': 12})
    # plt.yticks(size=12)
    # plt.xticks(size=12)
    #plt.legend(loc='upper right')
    plt.show()
def pic_comsol():
    y_true = np.loadtxt('E:/Graduation_project/验证逆向设计网络1000-2.txt',
                        skiprows=5, encoding='utf-8')
    x = y_true[:,0]
    y = y_true[:,1]
    y_pred = np.loadtxt('E:/Graduation_project/1000-2.txt')

    x_pred = range(480,1622,2)
    plt.figure(figsize=(8, 5))
    # 标签字体设置
    plt.plot( x,y, linewidth=2)
    plt.plot(x_pred[185:335], y_pred[185:335], '--',linewidth=1)
    plt.xlabel('Frequency(Hz)', fontdict={'fontsize': 12})
    #plt.ylabel('Value', fontdict={'fontsize': 12})
    plt.yticks(size=10)
    plt.xticks(size=10)
    # plt.rcParams.update({'font.size': 12})
    # plt.legend(loc='upper right')
    plt.show()
def fun():
    y_obj = ((np.loadtxt('E:/Graduation_project/1.chapter_data_save/3chapter/train_comsol_inputdata.txt',
                         skiprows=5, encoding='utf-8')))
    print(y_obj.shape)
    f_L = y_obj[:, 0]
    peak_hz = y_obj[:, 1]
    f_r = y_obj[:, 2]
    peak_absorb = y_obj[:, 3]
    H = y_obj[:, 4]
    w1 = y_obj[:, 5]
    s = y_obj[:, 6]

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    #plt.scatter(peak_hz, peak_absorb, label='predict')
    plt.scatter(H, w1)
    plt.xlabel('H', fontdict={'fontsize': 14})
    plt.ylabel('w1', fontdict={'fontsize': 14})
    plt.yticks(size=12)
    plt.xticks(size=12)

    plt.subplot(1, 2, 2)
    # plt.scatter(peak_hz, peak_absorb, label='predict')
    plt.scatter(s, peak_absorb)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.show()
def fun3():
    y_obj = ((np.loadtxt('E:/Graduation_project/1.chapter_data_save/3chapter/train_comsol_inputdata.txt',
                         skiprows=5, encoding='utf-8')))
    print(y_obj.shape)
    f_L = y_obj[:, 0]
    peak_hz = y_obj[:, 1]
    f_r = y_obj[:, 2]
    peak_absorb = y_obj[:, 3]
    H = y_obj[:, 4]
    w1 = y_obj[:, 5]
    s = y_obj[:, 6]

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    #plt.scatter(peak_hz, peak_absorb, label='predict')
    plt.scatter(H, w1)
    plt.xlabel('H', fontdict={'fontsize': 14})
    plt.ylabel('w1', fontdict={'fontsize': 14})
    plt.yticks(size=12)
    plt.xticks(size=12)

    plt.subplot(1, 2, 2)
    # plt.scatter(peak_hz, peak_absorb, label='predict')
    plt.scatter(s, peak_absorb)
    plt.xlabel('w3', fontdict={'fontsize': 14})
    plt.ylabel('w4', fontdict={'fontsize': 14})
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.show()
def fun2():
    y_obj = ((np.loadtxt('E:/Graduation_project/1.chapter_data_save/3chapter/LHS_train_comsol_sample.txt',
                         skiprows=5, encoding='utf-8')))
    print(y_obj.shape)
    H = y_obj[:, 0]
    w1 = y_obj[:, 1]
    w2 = y_obj[:, 2]
    w3 = y_obj[:, 3]
    w4 = y_obj[:, 4]


    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    #plt.scatter(peak_hz, peak_absorb, label='predict')
    plt.scatter(H, w1)
    plt.xlabel('H', fontdict={'fontsize': 16})
    plt.ylabel('w1', fontdict={'fontsize': 16})
    plt.yticks(size=14)
    plt.xticks(size=14)

    plt.subplot(1, 2, 2)
    # plt.scatter(peak_hz, peak_absorb, label='predict')
    plt.scatter(w3, w4)
    plt.xlabel('w3', fontdict={'fontsize': 16})
    plt.ylabel('w4', fontdict={'fontsize': 16})
    plt.yticks(size=14)
    plt.xticks(size=14)
    plt.show()

def pic_absorb_single3():
    plt.figure(figsize=(8, 5),dpi=200)
    data = np.loadtxt('E:/Graduation_project/4.4 700-900六结构/815-6.txt', skiprows=5, encoding='utf-8')
    data2 = np.loadtxt('E:/Graduation_project/4.4 700-900六结构/700-1.txt', skiprows=5, encoding='utf-8')
    data3 = np.loadtxt('E:/Graduation_project/4.4 700-900六结构/单720.txt', skiprows=5, encoding='utf-8')
    data4 = np.loadtxt('E:/Graduation_project/4.4 700-900六结构/单740.txt', skiprows=5, encoding='utf-8')
    data5 = np.loadtxt('E:/Graduation_project/4.4 700-900六结构/单766.txt', skiprows=5, encoding='utf-8')
    data6 = np.loadtxt('E:/Graduation_project/4.4 700-900六结构/单792.txt', skiprows=5, encoding='utf-8')
    data7 = np.loadtxt('E:/Graduation_project/4.4 700-900六结构/单815.txt', skiprows=5, encoding='utf-8')

    hz = data[:, 0]
    y_pred = data[:, 1]

    hz2 = data2[:, 0]
    y_pred2 = data2[:, 1]
    hz3 = data3[:, 0]
    y_pred3 = data3[:, 1]
    hz4 = data4[:, 0]
    y_pred4 = data4[:, 1]
    hz5 = data5[:, 0]
    y_pred5 = data5[:, 1]
    hz6 = data6[:, 0]
    y_pred6 = data6[:, 1]
    hz7 = data7[:, 0]
    y_pred7 = data7[:, 1]
    # 标签字体设置range(1000, 1551, 2)
    plt.plot(hz[25:250], y_pred[25:250], linewidth=2)
    plt.plot(hz2[25:250], y_pred2[25:250],'--', linewidth=1)
    plt.plot(hz3[25:250], y_pred3[25:250], '--', linewidth=1)
    plt.plot(hz4[25:250], y_pred4[25:250], '--', linewidth=1)
    plt.plot(hz5[25:250], y_pred5[25:250], '--', linewidth=1)
    plt.plot(hz6[25:250], y_pred6[25:250], '--', linewidth=1)
    plt.plot(hz7[25:250], y_pred7[25:250], '--', linewidth=1)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 16})
    plt.ylabel('吸声系数', fontdict={'fontsize': 16})
    plt.yticks(size=16)
    plt.xticks(size=16)
    plt.show()

def pic_absorb_single4():
    plt.figure(figsize=(8, 5))
    data = np.loadtxt('./860-988耦合曲线.txt', skiprows=5, encoding='utf-8')
    data2 = np.loadtxt('./988.txt', skiprows=5, encoding='utf-8')
    data3 = np.loadtxt('./图4.14-964hz基准曲线.txt', skiprows=5, encoding='utf-8')
    hz = data[:, 0]
    y_pred = data[:, 1]

    hz2 = data2[:, 0]
    y_pred2 = data2[:, 1]
    hz3 = data3[:, 0]
    y_pred3 = data3[:, 1]
    # 标签字体设置range(1000, 1551, 2)
    plt.plot(hz, y_pred, linewidth=2)
    plt.plot(hz3[20:], y_pred3[20:], '--', linewidth=1)
    plt.plot(hz2, y_pred2,'--', linewidth=1)

    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.show()

def pic_absorb_12():
    y_pred1 = np.loadtxt('./4.5_960hz.txt', skiprows=5, encoding='utf-8')
    y_pred2 = np.loadtxt('./4.5_960-2hz.txt', skiprows=5, encoding='utf-8')

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    # 标签字体设置range(1000, 1551, 2)
    plt.plot(y_pred1[:,0], y_pred1[:,1], linewidth=2)
    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.yticks(size=12)
    plt.xticks(size=12)

    plt.subplot(1,2,2)
    # 标签字体设置range(1000, 1551, 2)
    plt.plot(y_pred2[:,0], y_pred2[:,1], linewidth=2)

    plt.xlabel('频率(Hz)', fontdict={'fontsize': 14})
    plt.ylabel('吸声系数', fontdict={'fontsize': 14})
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.show()
if __name__=='__main__':
    pic_absorb_single2()
    #pic_absorb_single()
    # data = np.loadtxt('E:/Graduation_project/4.4 700-900六结构/815-6.txt', skiprows=5, encoding='utf-8')
    # # data2 = np.loadtxt('./基准曲线.txt', skiprows=5, encoding='utf-8')
    # # data3 = np.loadtxt('./988.txt', skiprows=5, encoding='utf-8')
    # hz = data[:, 0]
    # y_pred = data[:, 1]
    # s = 0
    # hz2 = []
    # y = []
    # for i in range(y_pred.shape[0]):
    #     if y_pred[i]>=0.8:
    #         s+=1
    #         y.append(y_pred[i])
    #         hz2.append(hz[i])
    # print(s*2)
    # print(hz2)
    # print(sum(y)/len(hz2))
    # pic_absorb()



