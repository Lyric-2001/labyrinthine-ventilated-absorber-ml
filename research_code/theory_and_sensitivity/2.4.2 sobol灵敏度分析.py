from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import math
from SALib.test_functions import Ishigami
import SALib
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib import ProblemSpec

def test():
    x_data = np.loadtxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/comsol模型/2.4.1sobol/origin_sobol_sample.txt')
    x = x_data[:, :]
    # print(X.shape)
    y = np.loadtxt('E:\毕设：空间盘绕/0.chapter_data_save/2chaper/comsol模型/2.4.1sobol/2.4.1相对带宽.txt')
    print(x.shape, y.shape)
    # # 设置参数范围
    sp = ProblemSpec({
        'names': ['W', 'H', 'x1', 'x2', 'x3', 'x4', 'x5'],
        'bounds': [[15, 28], [10, 28], [2, 12], [2, 12], [1, 10], [1, 10], [1, 10]],
        'outputs': ['Y']
    })
    # # 使用已有的采样数据进行敏感性分析
    sp.set_samples(np.array(x))  # 样本数据

    sp.set_results(y)  # 结果
    sp.analyze_sobol()
    # # 打印结果=
    print(sp)

def evalute(x):
    return x[0]+2*x[1]+x[3]*x[2]-1+x[4]+x[5]+x[6]

def sobol2():
    problem = {
        'num_vars': 7,
        'names': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
        'bounds': [[15, 28], [10, 28], [2, 12], [2, 12], [1, 10], [1, 10], [1, 10]]
    }
    L_max = 43
    # sobol采样
    sample_scaled = saltelli.sample(problem, 64, calc_second_order=True)  # n*(x+2)=8*(7+2)=72
    print('sample_scaled',sample_scaled.shape)
    sample_scaled = np.round(sample_scaled)
    np.savetxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/comsol模型/2.4.1sobol/origin_sobol_sample.txt', sample_scaled, fmt='%.2f')

    f_array = []
    # 遍历这个数据集，将通道宽度之和>45的设置为0
    for i in range(len(sample_scaled)):
        sum1 = 0
        z = 0
        for j in range(2, len(sample_scaled[i])):
            sum1 += sample_scaled[i][j]
            if sum1 > L_max or sample_scaled[i][j] == 1:
                z = j
                for k in range(j, len(sample_scaled[i])):
                    sample_scaled[i][k] = 0
                break
            else:
                z = len(sample_scaled[i])
        f = int(343 / (4 * (sample_scaled[i][0] * 0.001 * (z - 2) + ((z - 2) - 1) * 0.001)))
        f_array.append(f)
    # print('范围进行限制：\n', sample_scaled)
    print(sample_scaled.shape)

    merged_arr = np.column_stack((sample_scaled, f_array))
    np.savetxt('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/comsol模型/2.4.1sobol/comsol_sobol_sample.txt', merged_arr, fmt='%d')
    print(merged_arr.shape)


if __name__ == '__main__':
    #sobol2()
    test()