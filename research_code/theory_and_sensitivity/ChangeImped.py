#等效阻抗计算 完成Impedance类
import math,cmath
import matplotlib.pyplot as plt
import numpy as np

class CImpedeance:
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
    def __init__(self,f_left,f_right,w_i, W, L, H,l_i,f_interval):
        self.f1 = f_left
        self.f2 = f_right
        #self.w = 2*math.pi*f
        self.w_i = w_i    #通道宽度
        self.H = H
        self.L = L
        #self.n = n
        #结构宽度
        self.W = W
        self.f_interval = f_interval
        self.l_i = l_i


    def A(self):
        #print(self.Pw(self.w)/self.Cw(self.w))
        A = []
        for f in range(self.f1,self.f2,self.f_interval):
            w = 2 * math.pi * f
            z2c , z2 , z2l = 0 , 0 , 0
            for i in range(len(self.w_i)-1,-1,-1):#将w_i倒转
                if i == len(self.w_i)-1:
                    z2c , z2 , z2l = self.getzczl(w,i)
                elif i>0 and i < len(self.w_i)-1:
                    li = self.l_i[i]
                    si = self.w_i[i]

                    Pw = self.Pw(w, self.w_i[i])
                    Cw = self.Cw(w, self.w_i[i])

                    ki = w * cmath.sqrt(Pw * Cw)
                    z2l = self.Z1l(z2c, z2, ki, li)

                    li1 = self.l_i[i-1]
                    si1 = self.w_i[i - 1]

                    z2c = cmath.sqrt(Pw / Cw)
                    z2 = z2l * si1/si
                else:
                    l1 = self.l_i[i]

                    Pw1 = self.Pw(w, self.w_i[i])
                    Cw1 = self.Cw(w, self.w_i[i])

                    k1 = w * cmath.sqrt(Pw1 * Cw1)

                    zil = self.Z1l(z2c, z2, k1, l1)

            Zc = zil * (self.L) / (self.w_i[0])
            a = 1-(abs((Zc-self.Z0)/(Zc+self.Z0)))**2
            A.append(a)
        return A
    def getzczl(self,w,i):
        li1 = self.l_i[i-1]
        si1 = self.w_i[i - 1]

        li2 = self.l_i[i]
        si2 = self.w_i[i]

        Pw = self.Pw(w, wi=self.w_i[i])
        Cw = self.Cw(w, wi=self.w_i[i])

        zic = cmath.sqrt(Pw / Cw)
        ki = w * cmath.sqrt(Pw * Cw)
        Zil = complex(0, -zic / cmath.tan(ki * li2))
        zi = Zil * si1 / si2
        return zic,zi,Zil
    def Z1l(self,z2c,z2,k1,l1):
        t1 = complex(0,-z2/cmath.tan(k1*l1))
        t2 = complex(0, -z2c / cmath.tan(k1 * l1))
        return z2c * (t1+z2c)/(z2+t2)


    def Pw(self,w,wi):
        sum1 = 0
        iw = complex(0, w)
        for k in range(0,self.E):
            for n in range(0, self.E):
                temp1 = self.Ak(k,wi) ** 2
                temp2 =self.Bn(n) ** 2
                temp = temp2*temp1*(temp2+temp1+iw/self.V)
                sum1 += 1/temp
        temp3 = (self.D* self.V * (wi**2) * (self.H**2))/(64*iw)
        final = temp3 / sum1
        return final

    def Cw(self,w,wi):
        iw = complex(0, w)
        sum1 = 0
        for k in range(0, self.E):
            for n in range(0, self.E):
                temp1 = self.Ak(k,wi) ** 2
                temp2 = self.Bn(n) ** 2
                temp = temp2 * temp1 * (temp2 + temp1 + iw *self.Y / self.V2)
                sum1 += 1 / temp
        temp3 = (self.Y - 1) * 64 * iw / (self.V2 * (wi ** 2) * (self.H ** 2))
        final = 1/(self.P0) * (1 - temp3 * sum1)
        return final

    def Ak(self,k,wi):
        return (2*k+1)*math.pi/wi
    def Bn(self,n):
        return (2*n+1)*math.pi/self.H


if __name__ == '__main__':
    comsol_data_name = ("E:/Graduation_project/1.chapter_data_save/3chapter/"
                        "train_comsol_inputdata.txt")
    comsol_data = np.loadtxt(comsol_data_name, encoding='UTF-8')
    p = 2 * 10 ** (-3)
    t = 1 * 10 ** (-3)
    HZ = []
    print(comsol_data.shape)
    for num in range(1):
        i = 51
        w_i = [5* t, 5* t, 5* t,5* t]  # 每一个通道宽度
        L = sum(w_i) + (len(w_i)-1) * t  # 各个通道宽度并排的总长度
        H = 36 * t ##具体含义看论文 通道高度
        W = 28* t  # 通道长度

        l_i = []
        for j in range(len(w_i)):
            if i==0:
                li = math.sqrt((W+t) ** 2 + (w_i[j]) ** 2)
            else:
                li = math.sqrt((W) ** 2 + (w_i[j]+t) ** 2)
            l_i.append(li)

        print(i)

        f = int(343 / (4 * (W * (len(w_i) )+ (len(w_i) - 1) * t)))

        # 频率计算的范围
        f_l = f-1500 if f-1600 >0 else 50#650
        f_r = f+1000#871
        f_interval = 50  # 频率扫描间隔
        #print(f, f_l, f_r)
        # 调用类
        I = CImpedeance(f_l, f_r, w_i, W, L, H,l_i,f_interval)
        a = I.A()

        x = range(f_l, f_r, f_interval)
        max_idx = np.argmax(a)
        a_x, a_y = x[max_idx], a[max_idx]
        HZ.append(a_x)
        print(a_x, a_y)
        plt.figure(figsize=(10, 7))
        plt.scatter(a_x, a_y, color='red', s=10)
        # 绘制吸声曲线
        plt.plot(x, a)
        plt.show()
        print(comsol_data[i])
    print(len(HZ))
    #da = np.concatenate((comsol_data[:,0:-1], np.array(HZ).reshape(-1,1)), axis=1)
    #print(da.shape)
    #np.savetxt('E:/Graduation_project/1.chapter_data_save/3chapter/'
               #'test_comsol_inputdata_changelmped.txt',da,fmt='%.1f')





