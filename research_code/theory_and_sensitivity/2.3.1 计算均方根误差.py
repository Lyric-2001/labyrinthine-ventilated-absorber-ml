# -*- coding: utf-8 -*-

import math
import numpy as np

def get_average(records):
    """
    平均值
    """
    return sum(records) / len(records)


def get_variance(records):
    """
    方差 反映一个数据集的离散程度
    """
    average = get_average(records)
    return sum([(x - average) ** 2 for x in records]) / len(records)


def get_standard_deviation(records):
    """
    标准差 == 均方差 反映一个数据集的离散程度
    """
    variance = get_variance(records)
    return math.sqrt(variance)


def get_rms(records):
    """
    均方根值 反映的是有效值而不是平均值
    """
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None


def get_mae(records_real, records_predict):
    """
    平均绝对误差
    """
    if len(records_real) == len(records_predict):
        return sum([abs(x - y) for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


if __name__ == '__main__':
    name = 'A'
    #a = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.1结构{}理论解.txt'.format(name))
    b = np.loadtxt('C:/Users/DIY/Desktop/2.4.1_313.txt', encoding='UTF-8',
                   skiprows=5)
    c = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/comsol模型/2.4.1仿真结果/2.4.1_313.txt',
                   encoding='UTF-8',skiprows=5)
    # 均方根误差
    #rmse1 = get_rmse(a, b[:, 1])  # 0.81

    #print('a-b',rmse1)

    rmse2 = get_rmse(b[:, 1], c[:, 1])  # 0.81

    print('b-c',rmse2)


