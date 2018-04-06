'''
Created on 20.03.2018

@author: Yingxiong
'''
from os.path import join

from ibvpy.api import BCDof

from inverse.fem_inverse import MATSEval, FETS1D52ULRH, TStepper, TLoop
import matplotlib.pyplot as plt
from matresdev.db.simdb.simdb import simdb
import numpy as np


folder = join(simdb.exdata_dir,
              'double_pullout_tests',
              '2018-02-14_DPO_Leipzig',
              )

# import data

# measured data
filenames = ['DPOUC21A', 'DPOUC22A', 'DPOUC23A', 'DPOUC31A']
lb_arr = [98.32, 97.88, 97.14, 98.02]
n_rov_array = [8, 8, 8, 8]
dm1, fm1 = np.loadtxt(join(folder, 'DPOUC21A.txt'))
dm2, fm2 = np.loadtxt(join(folder,  'DPOUC22A.txt'))
dm3, fm3 = np.loadtxt(join(folder,  'DPOUC23A.txt'))
#dm4, fm4 = np.loadtxt(join(folder,  'DPOUC31A.txt'))

# specify which test to be evaluated
i_test = 2

dm_arr = [dm1, dm2, dm3]
fm_arr = [fm1, fm2, fm3]
plt.plot(dm_arr[i_test], fm_arr[i_test], label=filenames[i_test])
#plt.plot(dm2, fm2)
#plt.plot(dm3, fm3)
#plt.plot(dm4, fm4)
plt.title('Original test diagram')
plt.xlabel('crack opening [mm]')
plt.ylabel('Force [kN]')
plt.legend(loc='best')
plt.show()


# skip the first part where the remaining concrete at the notch is intact.
# For this skipped part, a linear function is assumed
d1 = np.hstack((0, np.linspace(0.135, 8, 100)))
f1 = np.interp(d1, dm1, fm1)

d2 = np.hstack((0, np.linspace(0.155, 8, 100)))
f2 = np.interp(d2, dm2, fm2)

d3 = np.hstack((0, np.linspace(0.135, 8, 100)))
f3 = np.interp(d3, dm3, fm3)

d_arr = [d1, d2, d3]
f_arr = [f1, f2, f3]

plt.plot(d_arr[i_test], f_arr[i_test] * 1000., label=filenames[i_test])
plt.title('Modified test diagram')
plt.xlabel('crack opening [mm]')
plt.ylabel('Force [N]')
plt.legend(loc='best')
plt.show()

mats = MATSEval(E_m=32701)

fets = FETS1D52ULRH(A_m=100. * 15. - n_rov_array[i_test] * 1.84,
                    A_f=n_rov_array[i_test] * 1.84)

ts = TStepper(mats_eval=mats,
              fets_eval=fets,
              L_x=lb_arr[i_test],  # half of specimen length
              n_e_x=20  # number of elements
              )

n_dofs = ts.domain.n_dofs
ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),  # the fixed DOF
              BCDof(var='u', dof=n_dofs - 1, value=1.0)]  # the DOF on which the displacement is applied

w_arr = np.hstack(
    (np.linspace(0, 0.15, 13), np.linspace(0.35, 4.2, 31)))

#w_arr = np.linspace(0., 4.2, 51)

pf_arr = np.interp(w_arr, d_arr[i_test] / 2., f_arr[i_test]) * 1000.

plt.plot(w_arr, pf_arr)
plt.title('Interpolated test diagram')
plt.xlabel('slip [mm]')
plt.ylabel('Force [N]')
plt.legend(loc='best')
plt.show()

# w_arr:

tl = TLoop(ts=ts, w_arr=w_arr, pf_arr=pf_arr, n=3)


slip, bond = tl.eval()

np.set_printoptions(precision=4)
print 'slip'
print [np.array(slip)]
print 'bond'
print [np.array(bond)]

plt.plot(slip, bond, label=filenames[i_test])
plt.title('Calibrated bond-slip law')
plt.xlabel('slip [mm]')
plt.ylabel('bond [N/mm]')
plt.legend(loc='best')
plt.show()
