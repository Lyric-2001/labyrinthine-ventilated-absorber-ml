import numpy as np
from itertools import product
import time as t

def matrix(m, n, number):  # 产生一个m行n列，插值点数量为number的采样矩阵
    """
    :param m: 参数数量
    :param n:采样个数
    :param number: 插值点数
    :return:
    """
    assert n < number ** m, '采样数量超过最大数量'
    sample = np.array(list(product(range(1, number + 1), repeat=m))).T#所有的插值点组合（参数数,插值点数**参数数）
    mat = np.array([[0] * n for i in range(m)])#采样矩阵（参数数,采样数）
    # 生成矩阵第一列
    for i in range(m):
        mat[i][0] = (i + 1) % number if (i + 1) % number else number
    # mat[1][0] = 5
    # 生成初始矩阵第一行
    for j in range(number):
        mat[0][j] = j + 1
    # 填满初始矩阵
    for i in range(1, m * number):
        # 将第一行置换到最后一行
        if i % number == 0:
            mat = np.vstack((mat[1:, :], mat[0, :]))
            for j in range(number):
                # if i + j > n-1:
                #     break
                mat[0][i + j] = j + 1

        for j in range(1, m):
            mat[j][i] = insert(mat[:j + 1, :i + 1], number)
    mat = np.vstack((mat[1:, :], mat[0, :]))

    # 移除已采样本
    for i in range(m * number):
        column = mat[:, i]#上述计算出的一种组合
        # print(list(column))
        for j in range(len(sample.T)):
            # print(sample[:,j].shape)
            if list(column) == list(sample[:, j]):
                sample = np.delete(sample, j, axis=-1)#将sample中与mat中的组合删除
                break
        # print(len(sample))
        # 填写扩充矩阵
    for i in range(m * number, n):
        l = np.array([])
        for ele in sample.T:#每一种组合
            distance = np.sum((mat[:, :i].T - ele) ** 2, axis=0) ** 0.5#计算mat中前i行，与任一组合的距离
            l = np.append(l, np.sum(distance ** -100) ** 0.01)
        index = np.argmin(l)
        mat[:, i] = sample[:, index]
        sample = np.delete(sample, index, axis=-1)
        print(l)
        # print(len(sample))
    np.savetxt('sample.csv', mat, delimiter=',')
    return mat


def d(vector1, vector2):  # 矩阵填充
    return np.sum((vector1 - vector2) ** 2) ** 0.5


def insert(mat, number):
    l = []
    # print(mat.shape[1])
    for i in range(number):
        mat[-1][-1] = i + 1#让待填为插值的值
        distance = 0
        # 此处必须减一，否则全是inf
        target = False
        for k in range(mat.shape[1] - 1):
            if d(mat[:, k], mat[:, -1]) == 0:
                target = True
                break
            distance += d(mat[:, k], mat[:, -1]) ** -100
        if target:
            l.append(10)
        else:
            l.append(distance ** 0.01)
    l = np.array(l)
    # print(l)
    return np.argmin(l) + 1


def transfer(mat, m, n, number, bound):
    # number 为每个变量的插值点数量
    # bound 为一个字典,记录每一个不确定区间变量的变化范围
    # 形式如 {‘x1':(1,2)}
    interval_range = []
    low = []
    for key, value in bound.items():
        interval_range.append([(value[1] - value[0]) / number])
        low.append([value[0]])
    np.array(interval_range)
    np.array(low)
    print(interval_range)
    print(low)
    mat = (mat - 1) * interval_range + low + np.random.rand(m, n) * interval_range
    return mat

if __name__ == '__main__':
    name = "train"#'test'#"train"
    dir_name = "E:/Graduation_project/1.chapter_data_save/3chapter/4-2重新采样/"
    bound = {
        'W': (18, 28),'x1': (4, 7), 'x2': (4, 7),
        'x3': (4, 7), 'x4': (4, 7), }
    m = 5
    num = 600 if name == "train" else 150
    number = 4
    mat = matrix(m, num, number)
    mat = transfer(mat, m, num, number, bound).T
    print(mat.shape)

    np.savetxt(dir_name+name+'_origin_inputdata.txt', mat)

    f_array = []
    for i in range(len(mat)):
        f = int(343 / (4 * (mat[i][0] * 0.001 * 4 + 3 * 0.001)))
        f_array.append(f)
    # print('范围进行限制：\n', sample_scaled)
    print(mat.shape)

    merged_arr = np.column_stack((mat, f_array))
    np.savetxt(dir_name+name+'_comsol_inputdata.txt', merged_arr, fmt='%.1f')
