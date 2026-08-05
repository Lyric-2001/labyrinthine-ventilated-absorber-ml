import numpy as np
import os
def get_band2(data2):
    location = np.argmax(data2)
    half_absorb = data2[location] * 0.5
    band = []
    for i in range(len(data2)):
        if data2[i] >= half_absorb:
            band.append(i)
    return band[0], band[-1]
def get_band(data2):
    location = np.argmax(data2)
    half_absorb = data2[location]*0.5
    band = []
    for i in range(len(data2)):
        if data2[i]>=half_absorb:
            band.append(i)
    return band[0],band[-1]

def get_para_absorb(x, y, savename):
    # 吸声曲线：峰值频率和系数、频率、系数、左、右频率、通道长度、首要宽度、总宽度、->>>>结构参数
    data = x
    peak = []
    peak_absorb = []
    HalfKHZ = []
    HZ = np.array([i for i in range(480,1622,2)])
    for i in range(0, x.shape[0]):
        data2 = y[i]
        print('data2:',data2.shape)
        location = np.argmax(data2)
        band_l, band_r = get_band(data2)
        #band_l2, band_r2 = get_band2(data2)
        peak.append([data2[band_l],data2[band_r]])
        peak_absorb.append([480+2*location,data2[location]])  # 保存峰值系数
        HalfKHZ.append(data2[:])
        # print(len(HalfKHZ[i]))

    L = data[:, 0].reshape(-1, 1)
    sum_w = [sum(data[i, 1:-1]) for i in range(data.shape[0])]
    w1 = data[:, 1].reshape(-1, 1)
    # print(sum_w[0:10],len(sum_w))
    parameter = np.concatenate((L,w1,np.array(sum_w).reshape(-1,1),
                                np.array(peak_absorb).reshape(-1, 2),
                                np.array(peak).reshape(-1, 2),
                                ), axis=1)

    da = []
    for i in range(parameter.shape[0]):
        new_para = data[i, 0:-1].reshape(1, -1).repeat(571, axis=0)
        new_peak = parameter[i, :].reshape(1, -1).repeat(571, axis=0)
        print('new_para.shape,HalfKHZ[i].shape',new_para.shape,HalfKHZ[i].shape,
              new_peak.shape,HZ.shape)
        da.append(np.concatenate((new_peak,HZ.reshape(-1, 1),
                                  HalfKHZ[i].reshape(-1, 1),new_para), axis=1))#HZ[i].reshape(-1, 1),
    # da=np.concatenate((parameter,np.array(HalfKHZ)), axis=1)
    # # # print(np.array(da).shape)
    da = np.array(da).reshape(-1, 14)
    print('da.shape:', da.shape)
    #print(da[0])
    np.savetxt(savename, da, fmt='%.8f')

if __name__ == "__main__":
    # x:吸收曲线 y:结构参数

    name = 'train'

    x = np.loadtxt('E:/Graduation_project/2_code/graduation_project/'
                'chapter_4/virtual_data/train_comsol_inputdata.txt')
    y= np.loadtxt('E:/Graduation_project/2_code/graduation_project/'
                'chapter_4/virtual_data/virtual_y.txt')
    savename = ('E:/Graduation_project/2_code/graduation_project/'
                'chapter_4/virtual_data/{}_virtual_absor_para.txt').format(name)
    print(x.shape,y.shape)

    get_para_absorb(x,y, savename)



