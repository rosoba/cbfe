'''
Created on 22.02.2018

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt

folder = 'D:\\data\\2018-02-14_DPO_Leipzig\\'

# monotonic
dm1, fm1 = np.loadtxt(folder + 'DPOUC21A.txt')
dm2, fm2 = np.loadtxt(folder + 'DPOUC22A.txt')
dm3, fm3 = np.loadtxt(folder + 'DPOUC23A.txt')
dm4, fm4 = np.loadtxt(folder + 'DPOUC31A.txt')
plt.plot(dm1, fm1)
plt.plot(dm2, fm2)
plt.plot(dm3, fm3)
plt.plot(dm4, fm4)
plt.ylim(0,)

# 2_cycles
d2c1, f2c1 = np.loadtxt(folder + 'DPOUC21B.txt')
d2c2, f2c2 = np.loadtxt(folder + 'DPOUC23B.txt')
d2c3, f2c3 = np.loadtxt(folder + 'DPOUC33B.txt')
# d2c1, f2c1 = np.loadtxt(folder + '.txt')
plt.figure()
plt.plot(d2c1, f2c1, color='k')
plt.plot(d2c2, f2c2, color='k')
plt.plot(d2c3, f2c3, color='k')
plt.ylim(0,)

# 3_cycles
d3c1, f3c1 = np.loadtxt(folder + 'DPOUC22B.txt')
d3c2, f3c2 = np.loadtxt(folder + 'DPOUC31B.txt')
d3c3, f3c3 = np.loadtxt(folder + 'DPOUC32B.txt')
d3c4, f3c4 = np.loadtxt(folder + 'DPOUC33A.txt')
plt.figure()
plt.plot(d3c1, f3c1, alpha=0.5)
plt.plot(d3c2, f3c2, alpha=0.5)
plt.plot(d3c3, f3c3, alpha=0.5)
plt.plot(d3c4, f3c4, alpha=0.5)
plt.ylim(0,)
plt.show()