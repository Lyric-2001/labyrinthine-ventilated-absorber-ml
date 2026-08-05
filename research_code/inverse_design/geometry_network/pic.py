import matplotlib.pyplot as plt
import numpy as np
def picture():
    a = np.loadtxt('E:/Graduation_project/1.chapter_data_save/3chapter/latin_maxerror_47.txt',
                   skiprows=5, encoding='utf-8')
    b = np.loadtxt('E:/Graduation_project/1.chapter_data_save/3chapter/Latin_test/3.1_47.txt',
                   skiprows=5, encoding='utf-8')
    plt.figure(figsize=(10, 8))
    plt.plot(a[:,0][55:255], a[:,1][55:255], label='predict')
    plt.plot(b[:,0][55:255], b[:,1][55:255], '--', label='true')
    # plt.plot(range(f_l, f_r, f_interval), c[:, 1],label='通风数值解',color='green')

    # plt.savefig('E:/毕设：空间盘绕/0.chapter_data_save/2chaper/2.3.1结构{}验证图.png'.format(name))
    plt.show()
if __name__ == '__main__':
    picture()
