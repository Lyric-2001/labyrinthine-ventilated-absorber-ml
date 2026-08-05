import numpy as np
import os
def get_band2(data2):
    absorb = data2[:, 1]
    location = np.argmax(data2[:, 1])
    band = []
    for i in range(len(absorb)):
        if absorb[i]>=0.1:
            band.append(i)
    return band[0],band[-1]
def get_band(data2):
    absorb = data2[:, 1]
    location = np.argmax(data2[:, 1])
    half_absorb = data2[location][1]*0.5
    band = []
    for i in range(len(absorb)):
        if absorb[i]>=half_absorb:
            band.append(i)
    return band[0],band[-1]
def get_absorb_para(dirname, dataname, savename):
    # 吸声曲线->>>>结构参数
    data = np.loadtxt(dataname)
    print('data.shape', data.shape)
    para = data[:,:-1]
    print('para.shape', para.shape)

    HalfKHZ = []

    for i in range(0, data.shape[0]):
        filename = dirname + str(i) + ".txt"
        data2 = np.loadtxt(filename, skiprows=5, encoding='utf-8')
        # print('data2:',data2.shape)
        HalfKHZ.append(data2[:, 1])
    HalfKHZ = np.array(HalfKHZ)
    print('HalfKHZ.shape:', HalfKHZ.shape)
    da = np.concatenate((HalfKHZ,para), axis=1)
    # da = np.array(da).reshape(-1, 11)
    print('da.shape:', da.shape)
    # print(da[0])
    np.savetxt(savename, da, fmt='%.6f')
def get_para_absorb(dirname, dataname, savename):
    # 吸声曲线：峰值频率和系数、频率、系数、左、右频率、通道长度、首要宽度、总宽度、->>>>结构参数
    data = np.loadtxt(dataname)
    print('data.shape', data.shape, data.shape[0])
    peak = []
    peak_absorb = []
    HalfKHZ = []
    HZ = []
    for i in range(0, data.shape[0]):
        filename = dirname + str(i) + ".txt"
        data2 = np.loadtxt(filename, skiprows=5, encoding='utf-8')
        # print('data2:',data2.shape)
        location = np.argmax(data2[:, 1])
        band_l, band_r = get_band(data2)
        band_l2, band_r2 = get_band2(data2)
        peak.append([data2[band_l, 0],data2[band_r, 0],data2[band_l2, 0],data2[band_r2, 0]])
        peak_absorb.append([data2[location, 0],data2[location][1]])  # 保存峰值系数
        HZ.append(data2[:, 0])
        HalfKHZ.append(data2[:, 1])
        # print(len(HalfKHZ[i]))

    L = data[:, 0].reshape(-1, 1)
    sum_w = [sum(data[i, 1:-1]) for i in range(data.shape[0])]
    w1 = data[:, 1].reshape(-1, 1)
    # print(sum_w[0:10],len(sum_w))
    parameter = np.concatenate((L,w1,np.array(sum_w).reshape(-1,1),
                                np.array(peak_absorb).reshape(-1, 2),
                                np.array(peak).reshape(-1, 4),
                                ), axis=1)

    da = []
    for i in range(parameter.shape[0]):
        new_para = data[i, 0:-1].reshape(1, -1).repeat(571, axis=0)
        new_peak = parameter[i, :].reshape(1, -1).repeat(571, axis=0)
        print('new_para.shape,HalfKHZ[i].shape',new_para.shape,HalfKHZ[i].shape,new_peak.shape)
        da.append(np.concatenate((new_peak,HZ[i].reshape(-1, 1),
                                  HalfKHZ[i].reshape(-1, 1),new_para), axis=1))#HZ[i].reshape(-1, 1),
    # da=np.concatenate((parameter,np.array(HalfKHZ)), axis=1)
    # # # print(np.array(da).shape)
    da = np.array(da).reshape(-1, 16)
    print('da.shape:', da.shape)
    #print(da[0])
    np.savetxt(savename, da, fmt='%.8f')
def get_para_absorb2(dirname, dataname, savename):
    # 吸声曲线：峰值频率和系数、左右频率、通道长度、首要宽度、总宽度、吸声曲线->>>>结构参数
    data = np.loadtxt(dataname)
    print('data.shape', data.shape, data.shape[0])
    peak = []
    peak_absorb = []
    HalfKHZ = []
    HZ = []
    for i in range(0, data.shape[0]):
        filename = dirname + str(i) + ".txt"
        data2 = np.loadtxt(filename, skiprows=5, encoding='utf-8')
        # print('data2:',data2.shape)
        location = np.argmax(data2[:, 1])
        band_l, band_r = get_band(data2)
        peak.append([data2[band_l, 0], data2[location, 0], data2[band_r, 0]])
        peak_absorb.append(data2[location][1])  # 保存峰值系数
        HZ.append(data2[:, 0])
        HalfKHZ.append(data2[:, 1])
        # print(len(HalfKHZ[i]))

    L = data[:, 0].reshape(-1, 1)
    sum_w = [sum(data[i, 1:-1]) for i in range(data.shape[0])]
    w1 = data[:, 1].reshape(-1, 1)
    # print(sum_w[0:10],len(sum_w))
    parameter = np.concatenate((np.array(peak).reshape(-1, 3),
                                np.array(peak_absorb).reshape(-1, 1),
                                L, w1, np.array(sum_w).reshape(-1, 1),), axis=1)

    da=np.concatenate((parameter,np.array(HalfKHZ),data[:,0:-1]), axis=1)

    print('da.shape:', da.shape)
    #print(da[0])
    np.savetxt(savename, da, fmt='%.8f')

if __name__ == "__main__":
    #x:吸收曲线 y:结构参数
    n = 0
    name = 'train' if n>0 else 'test'
    dirname = ('E:/Graduation_project/1.chapter_data_save/'
               '3chapter/{}_comsol_result/3.1_').format(name)

    dataname = ('E:/Graduation_project/2_code/graduation_project/'
                'chapter_3/{}_ParaPeakHz.txt').format(name)
    savename = ('E:/Graduation_project/2_code/graduation_project/'
                'chapter_4/inverse_model2/{}_absor_para.txt').format(name)
    #get_para_absorb2(dirname, dataname, savename)
    #get_absorb_para(dirname, dataname, savename)
    get_para_absorb(dirname, dataname, savename)