#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
def get_peak(parameter,PeakHZ,savename):
    """
    #文件1：获得结构参数、峰值频率文件——建模根据参数，预测峰值频率的模型---x:结构参数 y:峰值频率
    """
    #print('old_para:',parameter.shape)
#     print(parameter.shape)
    new_para = parameter[:,0:-1]#[0:3,0:-1]#
    # print('get_peak new_para:',new_para.shape)
    # print('get_peak peak_para:', len(PeakHZ))
    peak_para = np.concatenate((new_para,np.array(PeakHZ).reshape(-1,1)),axis =1)
    # print('new_para:','\n',new_para[0:3])
    # print('peak_para:','\n',peak_para[0:3])
    # print(peak_para.shape)
    np.savetxt(savename,peak_para,fmt='%.1f')

def get_absorb(parameter,HalfKHZ,n,savename):
    """
    #文件2：获得结构参数、频率、峰值500HZ内的吸声系数——建模根据参数和频率，预测峰值频率左右250HZ吸声系数的模型 参数数量为7
    #一个样本x:500*8（结构参数+hz）  y:频率对应的吸声系数500*1
    #或者：x：结构参数7    y：频率对应的吸声系数500
    :param parameter:
    :param HalfKHZ:
    :param n:
    :return:
    """
    da = []
    for i in range(n):
        new_para = parameter[i, 0:-1].reshape(1, -1)
        #print('new_para:',new_para.shape,new_para)
        HZ = HalfKHZ[i][0].reshape(-1, 1)  #251*1
        new_para = new_para.repeat(HZ.shape[0], axis=0)
        print('new_para.shape',new_para.shape)
        absorb = HalfKHZ[i][1].reshape(-1, 1)  # 251*1
        #print(HZ[0:3],absorb[0:3])
        print('new_para:',new_para.shape)
        print('HZ:',HZ.shape)
        print(i,'absorb:',absorb.shape)
        da.append(np.concatenate((new_para, np.array(HZ),np.array(absorb)), axis=1))
    #rint(np.array(da).shape,da[0])
    da = np.array(da).reshape(-1, 7)
    #print(np.array(da).shape, da[0])
    np.savetxt(savename,da,fmt='%.5f')
def procec_data(parameter,dirname,save_ParaHzAbsorb,save_ParaPeakHz):
    PeakHZ = []
    PeakAbsorb = []
    HalfKHZ = []#获得峰值频率【-150，150】对应的吸声系数，并生成一个个文件---确保每个吸声文件的峰值频率都至少有左右各250HZ
    for i in range(0,parameter.shape[0]):
        filename = dirname + str(i)+".txt"
        data = np.loadtxt(filename,skiprows=5,encoding='utf-8')
        location = np.argmax(data[:,1])
        PeakHZ.append(data[location][0])#保存峰值频率
        PeakAbsorb.append(data[location][1])#保存峰值系数
        HalfKHZ.append([data[:,0],data[:,1]])
    # print("PeakHZ",PeakHZ)
    # print("PeakAbsorb",PeakAbsorb)
    # print(HalfKHZ[1][0][0],HalfKHZ[1][1][0])
    print('HalfKHZ:',np.array(HalfKHZ).shape)

    get_peak(parameter,PeakHZ,save_ParaPeakHz)
    get_absorb(parameter,HalfKHZ,parameter.shape[0],save_ParaHzAbsorb)


if __name__ == "__main__":
    num = 0
    name = 'train' if num >0 else "test"
    savedir = 'E:/毕设：空间盘绕/0.chapter_data_save/3chapter/'


    parameter = np.loadtxt((savedir+'{}_comsol_inputdata.txt'.format(name)))
    dirname = savedir+'{}_comsol_result/3.1_'.format(name)

    save_ParaHzAbsorb = savedir+'{}_ParaHzAbsorb.txt'.format(name)
    save_ParaPeakHz = savedir+"{}_ParaPeakHz.txt".format(name)

    print(parameter.shape)
    procec_data(parameter,dirname,save_ParaHzAbsorb,save_ParaPeakHz)
    print("end")



