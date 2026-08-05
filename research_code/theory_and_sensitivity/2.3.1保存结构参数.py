import numpy as np
#将迷宫结构参数保存 迷宫通道数量最大为7

mean = 1#0:代表均匀截面 1：代表变截面
if mean==1:
    p = 3
    w_i = [3*p,p,2*p,p,p,p]
    L = sum(w_i) + (len(w_i)-1)
else:
    n = 7
    w_i = 2   # 每一个通道宽度
    L = n * w_i + (n+1)  # 各个通道宽度并排的总长度`
H = 38   ##具体含义看论文 通道高度
W = 28   # 通道长度
n = 7  ##通道数量


A = [[]]
A[0].append(H)
A[0].append(W)
A[0].append(L)

#均匀截面
if mean == 0:
    if n >= 7 :
        for i in range(n):
            A[0].append(w_i)
    else:
        for i in range(n):
            A[0].append(w_i)
        for i in range(7-n):
            A[0].append(0)
else:
    for i in range(len(w_i)):
        A[0].append(w_i[i])
    if len(w_i)<7:
        for i in range(7-len(w_i)):
            A[0].append(0)
print(A)
HEADER = 'H\tW\tL\tw1\tw2\tw3\tw4\tw5\tw6\tw7\t单位：mm'
np.savetxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.1结构B参数.txt',A,delimiter="\t",fmt='%.1f',header=HEADER)


