import numpy as np
def reComsol(comsol_data_name,comresult_dir_name ,save_name,n):# 输入文件路径，结果文件目录，修正后数据保存路径，文件数
    comsol_data = np.loadtxt(comsol_data_name,encoding='UTF-8')
    print('comsol_data:',comsol_data.shape) # 加载输入数据并打印形状
    num = 0
    change = []   # 初始化计数器和需修改的样本列表
    for i in range(n):   # 循环处理样本
        name = (comresult_dir_name +'3.1_{}.txt').format(i) #构建结果文件名 
        a = np.loadtxt(name, encoding='UTF-8', skiprows=5)  # 加载结果，跳过前5行
        x = a[:, 0]#频率
        mat = a[:, 1]#系数
        locat = np.argmax(mat) #最大吸声系数位置
        peak = x[locat] #对应的峰值

        if abs(peak-comsol_data[i][-1])>270: # 如果峰值与输入数据相差大于270Hz，则需要修改
            print(i,comsol_data[i],end='\t') # 打印样本索引和输入数据
            num += 1 # 计数器加1
            change.append(i) # 将样本索引添加到需修改的列表中
            sum1 = 0
            z = 0   #通道数
            for j in range(1, len(comsol_data[i])):
                sum1 += comsol_data[i][j]
                if comsol_data[i][j] == 0:
                    z = j
                    break
                else:
                    z = len(comsol_data[i])
            f = int(343 / (4 * (comsol_data[i][0] * 0.001 * (z - 1) + ((z - 1) - 1) * 0.001))) # 声学公式计算理论共振频率
            print('z:', z,'f:', f,'peak',peak,'前三个系数：',mat[0:3],)
            if peak == x[0] :  # 如果峰值在第一个位置，则将峰值减去150Hz，，，，否则直接使用真实峰值
                comsol_data[i][-1] = peak-150
            elif peak == x[-1]:
                comsol_data[i][-1] = peak+150
            else:
                comsol_data[i][-1] = peak  
    print('重新仿真num:',num)  # 输出仿真数量
    #np.savetxt(save_name, comsol_data, fmt='%.1f')

    # np.savetxt('E:/Graduation_project/1.chapter_data_save/3chapter/'
    #           'change2.txt', change, fmt='%d')
def reComsol_peak(comresult_dir_name,changelmped,n): #检查峰值评率差异数据
    changelmped_data = np.loadtxt(changelmped, encoding='UTF-8') #加载数据和计算峰值频率，
    num = 0
    for i in range(n):
        name = (comresult_dir_name +'3.1_{}.txt').format(i)
        a = np.loadtxt(name, encoding='UTF-8', skiprows=5)
        x = a[:, 0]#频率
        mat = a[:, 1]#系数
        locat = np.argmax(mat)
        peak = x[locat]
        if abs(peak-changelmped_data[i][-1])>150: #检查差值是否超过150Hz
            print(i,'peak:',peak,changelmped_data[i])
            num += 1
    print(f"{num}个样本差距过大")

if __name__ == "__main__":
    #E:/毕设：空间盘绕/0.chapter_data_save/3chapter/
    #E:/Graduation_project/1.chapter_data_save/3chapter/
    # Curation note: resolved an unfinished merge conflict; 1=train, 0=test.
    num = 0
    name = 'train' if num > 0 else "test"
    n = 1200 if num > 0 else 200 # 读取comresult_dir_name下n个文件

    comsol_data_name = 'E:/Graduation_project/1.chapter_data_save/3chapter/'+name+'_comsol_inputdata_peak.txt'
    comresult_dir_name = 'E:/Graduation_project/1.chapter_data_save/3chapter/'+name+'_comsol_result/'

    save_name = 'E:/毕设：空间盘绕/0.chapter_data_save/3chapter/test_comsol_inputdata_peak_re.txt'
    changelmped = 'E:/Graduation_project/1.chapter_data_save/3chapter/'+name+"_comsol_inputdata_changelmped.txt"
    #reComsol(comsol_data_name,comresult_dir_name,save_name,n)
    reComsol_peak(comresult_dir_name, changelmped, n)

