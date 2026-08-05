import random
import numpy as np
import matplotlib.pyplot as plt
#定义 拉丁超立方随机抽样 函数
def lhs(number,dimension,d):
    #number:抽样数量,dimension:设计变量个数,d:字典表示每个设计变量的上下界
    if dimension != len(d):
        raise Exception('输入数据有误')
    sample = [[] for i in range(number)]  #初始化样本矩阵
    weight_mat = [list(range(number)) for i in range(dimension)] #初始化一个权重矩阵用于采样
    for i in range(number):
        w = []#五个变量的权重随机数
        #循环 分别取出六个变量
        for j in range(dimension):
            r = random.randint(0,len(weight_mat[j])-1)#0~199随机值（因为每次删除一个元素，其实是在长度中取随机值）
            w.append(weight_mat[j].pop(r))#在j下标元素列表中删除r元素，并将r赋给w 保证i+1循环时，取过的不会再取到（分层抽样，每层一个）
        index = 0
        for ele in d:
            low = (d[ele][1]-d[ele][0])/number*w[index] + d[ele][0]#将各变量取值区间分为200份，得到权值对应位置的下限
            up = (d[ele][1]-d[ele][0])/number*(w[index]+1) + d[ele][0]#得到样本点的上限
            ramdom_point = random.uniform(low,up)#在上下限之间随机选取一个样本值
            sample[i].append(ramdom_point)#存入第i组样本的第index个变量
            index += 1
    return sample#获得200组样本点
def test():
    W = (18,28)
    w1 = (4,7 )
    w2 = (4,7 )
    w3 = (4,7 )
    w4 = (4,7 )

    d = {'x1':W,
         'x2':w1,
         'x3':w2,
         'x4':w3,
         'x5':w4,}
    data = np.array(lhs(1200,5,d))
    print(data)
    np.savetxt('./LHS_sample.txt',data,delimiter='\t')
    np.savetxt('./LHS_comsol_sample.txt',data,fmt='%.1f')
#查看所抽样的数据
# file = open('F:/608课题项目/608项目/1.python_program/2.灵敏度、线性回归/LHS_sample.txt')
# l = file.read().split('\n')
# for ele in l:#遍历每一组变量样本
#  print(ele.split('\t'))#将样本组以制表符为分隔符进行打印
if __name__ == '__main__':

    a = np.loadtxt('./LHS_sample.txt')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(a[:, 0], a[:, 1])
    plt.title("(a)", fontdict={'fontsize': 12})
    plt.xlabel('W', fontdict={'fontsize': 12})
    plt.ylabel('w1', fontdict={'fontsize': 12})
    #plt.rcParams.update({'font.size': 12})
    #plt.legend(loc='upper right')

    plt.subplot(1, 2,2)
    plt.scatter(a[:, 3], a[:, 4])
    plt.title("(b)", fontdict={'fontsize': 12})
    plt.xlabel('w3', fontdict={'fontsize': 12})
    plt.ylabel('w4', fontdict={'fontsize': 12})
    #plt.rcParams.update({'font.size': 12})
    #plt.legend(loc='upper right')

    plt.show()

    #a = np.loadtxt('./LHS_comsol_sample.txt')
