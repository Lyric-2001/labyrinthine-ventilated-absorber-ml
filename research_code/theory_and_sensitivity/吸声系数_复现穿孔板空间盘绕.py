from sympy import *
import math
import matplotlib.pyplot as plt
def calcult_P(pi,d,S):
    P = 0.25*pi*d*d/S
    return P

def calcult_K(d,w,u):
    K = 0.5*d*(w/u)**0.5
    return K

#计算微孔板阻抗
def calcult_Xh(P,K,c,d,u,t,):
    t1 = 32*u*t
    t2 = P*c*d*d
    a1 = (1+K*K/32)**0.5
    a2 = 2**0.5*K*d/(t*8)

    Xh = t1/t2*(a1+a2)
    return Xh

#计算微孔板电抗
def calcult_Yh(P, K, c, d, w, t, ):
    t1 = w * t
    t2 = P * c
    a1 = 1+1/((9 + K * K / 2) ** 0.5)+0.85*d/t

    Yh = t1 / t2 * a1
    return Yh

#计算微孔板电抗
def calcult_Yc(S,S2,c0, w, l):
    t1 = S/S2
    t2 = math.cos(w*l/c0)/math.sin(w*l/c0)  #simplify(cot(w*l/c0))
    Yc = -t1*t2
    return Yc

#计算吸声系数
def cal_alpha(xh,yh,yc):
    t1 = 4*xh
    t2 = (1+xh)**2 + (yh+yc)**2
    a = t1/t2
    return a


def mm_to_m(x):
    return x*10**-3


def calcult(pi,c, d, u, t,s,s2,W_af,a,b,w,leff):
    P = calcult_P(pi, d, s)
    K = calcult_K(d, W_af, u)
    xh = calcult_Xh(P, K, c, d, u, t)
    yh = calcult_Yh(P, K, c, d, W_af, t)

    yc = calcult_Yc(s, s2, c, W_af, leff)
    res = cal_alpha(xh, yh, yc)
    return res

import cmath
def calcul_leff(c,w,r):#错误代入后计算不出来
    t1 = c/w

    temp =  (1+r) / (1-r)
    temp = -temp.imag
    t2 = 1/simplify(cot(temp))

    return t1*t2

if __name__ == '__main__':
    # 小孔直径 m
    d = mm_to_m(3.3)
    a = mm_to_m(100)
    t = mm_to_m(0.2)
    w = mm_to_m(12)  # 厚度
    b = mm_to_m(2)
    pi = math.pi


    #动黏度系数u pa s
    u = 1.56*(10**-5)
    #角频率w


    #声速c m/s
    c = 343
    #密度kg/m^3
    rho = 1.21

    #穿孔板横截面积
    s = a*a#0.25*pi*d**2##a*a
    #空腔横截面积
    s2 = w*w
    leff = calcul_leff(c, 2 * pi * 124, complex(0.425,-0.0447))
    print(leff)
    leff = 6.35*a #13 * a - 56 * b - 42 * w #此值，论文中的有效长度leff是从模拟中得到的
    print(leff)
    alpha_c = []
    zero= []
    XH = []
    fre = [i for i in range(100, 150, 1)]
    max_value = [0,0]#频率和值
    for f in fre:
        #绘制实部虚部
        # W_af = 2 * pi * f
        # P = calcult_P(pi, d, s)
        # K = calcult_K(d, W_af, u)
        # xh = calcult_Xh(P, K, c, d, u, t)#与频率有关 #与孔径d有关（先确定d）
        # XH.append(xh)
        # yh = calcult_Yh(P, K, c, d, W_af, t)#与频率有关
        # yc = calcult_Yc(s, s2, c, W_af, leff)
        # zero.append(yh+yc)

        #绘制吸声曲线
        W_af = 2 * pi * f
        alpha=calcult(pi,c, d, u, t,s,s2,W_af,a,b,w,leff)

        alpha_c.append(alpha)
        if max_value[1]<alpha :
            max_value[0] = f
            max_value[1] = alpha

    plt.figure(figsize=(20,12))
    # 绘制吸声曲线
    plt.plot(fre,alpha_c)

    # 绘制实部虚部
    # plt.plot(fre, XH)
    # plt.scatter(fre,zero)
    # plt.plot(fre, [0 for _ in fre],'-')
    # plt.plot(fre, [1 for _ in fre], '.')
    plt.show()

    print(max_value)