#等效阻抗计算 完成Impedance类
import math,cmath
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class Impedeance:
    D = 1.21
    U = 1.81 * (10 ** (-5))
    Y = 1.41
    K = 0.026
    V = U/D
    Cv = 718.14
    V2 = K/(D*Cv)
    E = 10**(2)
    C = 343
    P0 = D * (C**2) / Y
    Z0 = 1.21 * 343
    def __init__(self,f_left,f_right,w_i, W, L, H, n,f_interval):
        self.f1 = f_left
        self.f2 = f_right
        #self.w = 2*math.pi*f
        self.w_i = w_i    #通道宽度
        self.H = H
        self.L = L
        self.n = n
        #结构宽度
        self.W = W
        self.f_interval = f_interval

    def A(self):
        #print(self.Pw(self.w)/self.Cw(self.w))
        A = []
        for f in range(self.f1,self.f2,self.f_interval):
            w = 2 * math.pi * f
            Pw = self.Pw(w)
            Cw = self.Cw(w)
            Zcur = cmath.sqrt(Pw / Cw)

            leff = math.sqrt((self.W)**2+(self.w_i+0.001)**2)*(self.n-1)+math.sqrt((self.W+0.001)**2)
            Z = self.get_Z(w,leff,Pw,Cw,Zcur)
            Zc = complex(0, -Z * (self.L) / ((self.w_i)))

            a = 1-(abs((Zc-self.Z0)/(Zc+self.Z0)))**2
            A.append(a)
        return A
    def get_Z(self,w,leff,Pw,Cw,Zcur):
        keq = w * cmath.sqrt(Pw * Cw)
        Z = Zcur / cmath.tan(keq * leff)
        return Z

    def Pw(self,w):
        sum1 = 0
        iw = complex(0, w)
        for k in range(0,self.E):
            for n in range(0, self.E):
                temp1 = self.Ak(k) ** 2
                temp2 =self.Bn(n) ** 2
                temp = temp2*temp1*(temp2+temp1+iw/self.V)
                sum1 += 1/temp
        temp3 = (self.D* self.V * (self.w_i**2) * (self.H**2))/(64*iw)
        final = temp3 / sum1
        return final

    def Cw(self,w):
        iw = complex(0, w)
        sum1 = 0
        for k in range(0, self.E):
            for n in range(0, self.E):
                temp1 = self.Ak(k) ** 2
                temp2 = self.Bn(n) ** 2
                temp = temp2 * temp1 * (temp2 + temp1 + iw *self.Y / self.V2)
                sum1 += 1 / temp
        temp3 = (self.Y - 1) * 64 * iw / (self.V2 * (self.w_i ** 2) * (self.H ** 2))
        final = 1/(self.P0) * (1 - temp3 * sum1)
        return final

    def Ak(self,k):
        return (2*k+1)*math.pi/self.w_i
    def Bn(self,n):
        return (2*n+1)*math.pi/self.H

if __name__ == '__main__':
    if plt is None:
        raise RuntimeError('Matplotlib is required when running this file as a script.')
    w_i = 2 * 10 ** (-3)  # 每一个通道宽度
    H = 28 * 10 ** (-3)  ##具体含义看论文 通道高度
    W = 28 * 10 ** (-3)  # 通道长度
    n = 7  ##通道数量
    L = n * w_i + (n-1) * 10 ** (-3)  # 各个通道宽度并排的总长度
    print('L',L)
    f = int(343/(4*(W*n+(n)*0.001)))

    # 频率计算的范围
    f_l = 135
    f_r = 695+1
    f_interval = 1  #频率扫描间隔
    print(f,f_l,f_r)
    # 调用类
    I = Impedeance(f_l, f_r, w_i, W, L, H, n,f_interval)
    a = I.A()

    x = range(f_l, f_r, f_interval)
    max_idx = np.argmax(a)
    a_x, a_y = x[max_idx], a[max_idx]

    plt.figure(figsize=(10, 7))
    plt.scatter(a_x, a_y, color='red', s=10)
    # 绘制吸声曲线
    plt.plot(x, a)
    plt.show()
    #np.savetxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.1结构B理论解.txt',a)
    #
    # a = np.loadtxt('./lmpedance_test')

    # b = np.loadtxt('E:/毕设：空间盘绕/Untitled1.6.1.txt', encoding='UTF-8', skiprows=5)
    # # print(b.shape,b[:,1])
    # plt.figure(figsize=(10, 6))
    # # 绘制吸声曲线
    # plt.plot(range(f_l, f_r, f_interval), b[:, 1])
    # plt.plot(range(f_l, f_r, f_interval), a, 'r')
    # plt.show()

