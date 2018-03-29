'''
Created on 26.03.2018

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt
# import os
# from tensile_test import CompositeTensileTest
# from cb import NonLinearCB
# from fe_nls_solver_cb import MATSEval, FETS1D52ULRH, TStepper, TLoop
# from stats.misc.random_field.random_field_1D import RandomField


# import data
f_name = 'DKUC32'

folder = 'D:\\data\\2018-01-07_TT_Leipzig\\raw\\'
DKBC11 = np.loadtxt(folder + f_name + '.ASC', delimiter=';')
f_DKBC11 = DKBC11[:, 1]
d1_DKBC11 = DKBC11[:, 3]
d2_DKBC11 = DKBC11[:, 4]
d3_DKBC11 = DKBC11[:, 5]

d_DKBC11 = -((d1_DKBC11 + d2_DKBC11) * 0.25 + d3_DKBC11 * 0.50)

plt.plot(-d1_DKBC11, f_DKBC11)
plt.plot(-d2_DKBC11, f_DKBC11)
plt.plot(-d3_DKBC11, f_DKBC11)
plt.plot(d_DKBC11, f_DKBC11, 'k', lw=2)

plt.figure()
plt.plot(np.arange(len(d_DKBC11)), d_DKBC11)

plt.show()

np.savetxt(
    folder + f_name + '.txt', np.vstack((d_DKBC11[0:499], f_DKBC11[0:499])))

d, f = np.loadtxt(folder + f_name + '.txt')
plt.plot(d, f)
plt.show()
