import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# 示例散点数据
name2 = 8
name1 = 'E:/毕设：空间盘绕/0.chapter_data_save/2chaper/comsol模型/2.4.1仿真结果/2.4.1_'
name3 = '.txt'
name = name1+str(name2)+name3
a = np.loadtxt(name, encoding='UTF-8', skiprows=5)
x_data = a[:, 0]
y_data = a[:, 1]

# 执行线性回归拟合
slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

# 根据回归系数生成拟合直线的x值
x_fit = np.linspace(min(x_data), max(x_data), 100)

# 计算拟合直线的y值
y_fit = slope * x_fit + intercept

# 绘制原始散点和拟合直线
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_fit, y_fit, 'r', label='Fit')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()