'''
Created on 25.12.2017

@author: Yingxiong
'''
from cbfe.fe_nls_solver_incre import MATSEval, FETS1D52ULRH, TStepper, TLoop
import numpy as np
from ibvpy.api import BCDof
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

data = np.loadtxt('D:\\F-s-test.txt').T

x = data[1]
y = data[0]

y0 = data[0] * 1000. / np.pi / 25. / 125.


interp = interp1d(x, y0)

mat = MATSEval(E_m=30000.,
               E_f=200000.)

fet = FETS1D52ULRH(A_m=250. * 250. - np.pi * 12.5 ** 2,
                   A_f=np.pi * 12.5 ** 2,
                   L_b=np.pi * 25.)

ts = TStepper(mats_eval=mat,
              fets_eval=fet,
              L_x=125.,  # half of speciment length
              n_e_x=20  # number of elements
              )

n_dofs = ts.domain.n_dofs

ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),  # the fixed DOF
              BCDof(var='u', dof=n_dofs - 1, value=1.1)]  # the DOF on which the displacement is applied

ts.mats_eval.slip = [0.,  0.055,  0.125,  0.1917,  0.2583,  0.325,  0.3917,
                     0.4583,  0.525,  0.5917,  0.6583,  0.725,  0.7917,  0.8583,
                     0.925,  0.9917,  1.0583,  1.125]
# ts.mats_eval.bond = [0.,   7.8261,  10.2581,  12.337,  13.92,  16.0419,
#                      17.6201,  18.3639,  18.9591,  19.2747,  19.642,  19.8493,
# 19.9709,  19.963,  19.8794,  19.8616,  19.5439,  19.3754]

ts.mats_eval.bond = interp(ts.mats_eval.slip).tolist()


tl = TLoop(ts=ts)

U_record, F_record, sf_record, sig_m_record, sig_f_record = tl.eval()
n_dof = 2 * ts.domain.n_active_elems + 1
plt.plot(U_record[:, n_dof], F_record[:, n_dof], label='loaded')
plt.plot(U_record[:, 1], F_record[:, n_dof], label='free')

plt.plot(x, y * 1000., '--')

plt.legend()
plt.show()
