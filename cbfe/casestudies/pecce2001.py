'''
Created on 22.11.2017

@author: Yingxiong
'''
import numpy as np
import matplotlib.pyplot as plt
from cbfe.fe_nls_solver_incre import TStepper, TLoop, MATSEval, FETS1D52ULRH
from ibvpy.api import BCDof


mat = MATSEval(E_m=28484,
               E_f=42000)

fet = FETS1D52ULRH(A_m=290. * 250.,
                   A_f=np.pi * (12.7 / 2.) ** 2,
                   L_b=np.pi * 12.7)

ts = TStepper(mats_eval=mat,
              fets_eval=fet)

ts.L_x = 12.7 * 40.
ts.n_e_x = 20

# test 2
# tau_m = 16.8
# s_m = 0.35
# alpha = 0.87
# p = 0.06

# test 1
tau_m = 11.3
s_m = 0.43
alpha = 0.85
p = 0.05

slip = np.linspace(0, 5, 1000)
bond = tau_m * (slip / s_m) ** alpha * (slip <= s_m) + \
    tau_m * (1. + p - p * slip / s_m) * (slip > s_m)

# plt.plot(slip, bond)
# plt.show()


ts.mats_eval.slip = slip.tolist()
ts.mats_eval.bond = bond.tolist()

n_dofs = ts.domain.n_dofs

ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
              BCDof(var='u', dof=n_dofs - 1, value=5.0)]


tl = TLoop(ts=ts)

U_record, F_record, sf_record, sig_m_record, sig_f_record = tl.eval()
n_dof = 2 * ts.domain.n_active_elems + 1
plt.plot(U_record[:, n_dof], F_record[
         :, n_dof] / 1000., label='loaded end')
plt.plot(U_record[:, 1], F_record[:, n_dof] / 1000., label='free end')

ex2_loaded_x = np.array(
    [0.0, 0.044, 0.117, 0.221, 0.320, 0.445, 0.581, 0.655, 1.352, 2.150, 3.028, 3.979])
ex2_loaded_y = np.array([0.0, 2.000, 4.778, 8.261, 10.898, 13.864, 15.842, 16.596, 14.486, 12.002, 9.330, 6.376
                         ])

ex2_free_x = np.array([0.0, 0.0, 0.031, 0.046, 0.082,
                       0.123, 0.228, 0.402, 0.661, 0.920, 1.147, 1.433, 2.528])
ex2_free_y = np.array([0.071, 4.165, 6.659, 10.330, 12.495,
                       14.566, 16.591, 15.981, 14.761, 13.634, 12.649, 11.334, 6.406])

ex1_free = np.array([0.0, 0.022, 0.040, 1.827, 0.074, 3.462, 0.142, 5.998, 0.176, 7.245, 0.234, 8.406, 0.307, 9.308, 0.394, 10.383, 0.476, 11.070, 0.520, 11.242,
                     0.651, 11.197, 0.859, 10.808, 1.078, 10.332, 1.325, 9.814, 1.582, 9.467, 1.844, 9.077, 2.091, 8.429, 2.261, 8.212, 2.469, 8.382,
                     2.658, 7.735, 2.842, 8.206, 2.988, 7.731, 3.158, 7.988, 3.327, 7.642, 3.536, 7.295, 3.773, 7.078, 4.011, 6.688, 4.297, 6.384, 4.549, 6.381,
                     4.821, 6.120, 5.000, 5.946])

ex1_loaded = np.array([-0.008, -0.021, 0.055, 2.300, 0.103, 3.375, 0.171, 4.794, 0.229, 6.041, 0.292, 7.502, 0.379, 8.706, 0.433, 9.006, 0.535, 9.994,
                       0.627, 10.896, 0.724, 11.326, 0.835, 11.238, 1.039, 10.935, 1.238, 10.503, 1.412, 10.114, 1.834, 9.421, 2.144, 8.859, 2.445, 8.296,
                       2.673, 8.294, 2.833, 7.690, 3.002, 8.118, 3.153, 7.816, 3.381, 7.899, 3.657, 7.337, 3.856, 7.206, 4.147, 6.558, 5.005, 5.989])


# plt.plot(ex1_free[0::2], ex1_free[1::2] * ts.L_x * fet.L_b / 1000.)
# plt.plot(ex1_loaded[0::2], ex1_loaded[1::2] * ts.L_x * fet.L_b / 1000.)


plt.xlabel('displacement [mm]')
plt.ylabel('pull-out force [KN]')

plt.legend(loc='best')

plt.show()
