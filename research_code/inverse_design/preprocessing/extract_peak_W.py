import numpy as np
import os
def get_band(data2):
    absorb = data2[:, 1]
    location = np.argmax(data2[:, 1])
    half_absorb = data2[location][1]*0.5
    band = []
    for i in range(len(absorb)):
        if absorb[i]>=half_absorb:
            band.append(i)
    return band[0],band[-1]
def get_para_absorb(dirname, dataname, savename):
    #峰值频率、峰值系数、通道长度、通道总宽度、首要通道宽度
    data = np.loadtxt(dataname)
    print('data.shape',data.shape,data.shape[0])
    peak = []
    peak_absorb = []
    HalfKHZ = []
    HZ = []
    for i in range(0,data.shape[0]):
        filename = dirname + str(i)+".txt"
        data2 = np.loadtxt(filename,skiprows=5,encoding='utf-8')
        #print('data2:',data2.shape)
        location = np.argmax(data2[:,1])
        band_l, band_r = get_band(data2)
        peak.append([band_l,location,band_r])
        peak_absorb.append(data2[location][1])#保存峰值系数
        HZ.append(data2[:, 0])
        HalfKHZ.append(data2[:, 1])
        #print(len(HalfKHZ[i]))

    L = data[:,0].reshape(-1,1)
    sum_w = [sum(data[i,1:-1]) for i in range(data.shape[0])]
    w1 = data[:,1].reshape(-1,1)
    # print(sum_w[0:10],len(sum_w))
    parameter = np.concatenate((np.array(peak).reshape(-1,3),
                                np.array(peak_absorb).reshape(-1,1),
                               L,w1,np.array(sum_w).reshape(-1,1),), axis=1)
    #print('parameter.shape:',parameter.shape)

    da = []
    HalfKHZ = np.array(HalfKHZ)
    HZ = np.array(HZ)
    #print('HalfKHZ.shape:', HalfKHZ.shape)

    for i in range(parameter.shape[0]):
        new_para = parameter[i,:].reshape(1,-1).repeat(571, axis=0)
        #print('new_para.shape,HalfKHZ[i].shape',new_para.shape,HalfKHZ[i].shape)
        da.append(np.concatenate((new_para,HZ[i].reshape(-1,1),HalfKHZ[i].reshape(-1,1)),
                                 axis=1))
    # da=np.concatenate((parameter,np.array(HalfKHZ)), axis=1)
    #print(np.array(da).shape)
    da = np.array(da).reshape(-1,9)
    print(da.shape)
    np.savetxt(savename, da, fmt='%.6f')


if __name__ == "__main__":
    n = 1
    name = 'train' if n>0 else 'test'
    dirname = 'E:/Graduation_project/1.chapter_data_save/3chapter/{}_comsol_result/3.1_'.format(name)
    # 峰值频率、峰值系数、通道长度、通道总宽度、首要通道宽度
    dataname = 'E:/Graduation_project/2_code/graduation_project/chapter_3/{}_ParaPeakHz.txt'.format(name)
    savename = 'E:/Graduation_project/2_code/graduation_project/chapter_4/inverse_model1/{}_para_absor.txt'.format(name)
    get_para_absorb(dirname, dataname, savename)
