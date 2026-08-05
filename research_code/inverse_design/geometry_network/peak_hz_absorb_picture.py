import random

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('./Latin_train_absorpara_model2.txt')[::571]
print('data.shape', data.shape)
peak_hz = data[:,1]
absorb =  data[:,3]
lhz = peak_hz - data[:,0]
rhz = data[:,2]-peak_hz
W = data[:,4]
w1 = data[:,5]
s= data[:,6]
plt.figure(figsize=(8, 5))
plt.scatter(s,absorb)
plt.show()
# l_r_hz = {'12':[14,16],'14':[12,14,16,18],'16':[14,16,18,20],
#           '18':[16,18,20,22],'20':[18,20,22],'22':[20,22]}
# print(l_r_hz['14'])
# a =np.array( [1,2,3])
# print(a*2)
# print(3//2)
# print(random.choice([25,37,48]))