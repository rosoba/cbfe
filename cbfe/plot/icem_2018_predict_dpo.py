'''
Created on 26.03.2018

@author: Yingxiong
'''
from os.path import join

from ibvpy.api import BCDof

from cbfe.fe_nls_solver_incre import MATSEval, FETS1D52ULRH, TStepper, TLoop
import matplotlib.pyplot as plt
from matresdev.db.simdb.simdb import simdb
import numpy as np

# specify font options for plots
params = {'legend.fontsize': 20,
          #         'legend.linewidth': 2,
          u'font.size': 20,
          u'font.family': 'Times New Roman',
          u'font.style': 'normal'}
plt.rcParams.update(params)

folder = join(simdb.exdata_dir,
              'double_pullout_tests',
              '2018-02-14_DPO_Leipzig',
              )

filenames = ['DPOUC21A', 'DPOUC22A', 'DPOUC23A', 'DPOUC31A']
dm1, fm1 = np.loadtxt(join(folder, 'DPOUC21A.txt'))
dm2, fm2 = np.loadtxt(join(folder,  'DPOUC22A.txt'))
dm3, fm3 = np.loadtxt(join(folder,  'DPOUC23A.txt'))
#dm4, fm4 = np.loadtxt(join(folder,  'DPOUC31A.txt'))


# skip the first part where the remaining concrete at the notch is intact
d1 = np.hstack((0, np.linspace(0.135, 8, 100)))
f1 = np.interp(d1, dm1, fm1)

d2 = np.hstack((0, np.linspace(0.155, 8, 100)))
f2 = np.interp(d2, dm2, fm2)

d3 = np.hstack((0, np.linspace(0.135, 8, 100)))
f3 = np.interp(d3, dm3, fm3)


# V1: DPOUC21A
slip1 = np.array([0.,  0.0333,  0.0833,  0.1333,  0.1833,  0.3488,  0.645,
                  0.9413,  1.2375,  1.5338,  1.83,  2.1263,  2.4225,  2.7188,
                  3.015,  3.3112,  3.6075,  3.9038,  4.1013,  4.2])
bond1 = np.array([0.,  16.8861,  24.9765,  30.0492,  30.7128,  31.3073,
                  33.9753,  36.8538,  38.2666,  39.3197,  40.4113,  41.4363,
                  42.3787,  43.2716,  44.3788,  45.3996,  46.3284,  47.1807,
                  47.1634,  47.1792])

# V2: DPOUC22A
slip2 = np.array([0.,  0.0333,  0.0833,  0.1333,  0.1833,  0.3488,  0.645,
                  0.9413,  1.2375,  1.5338,  1.83,  2.1263,  2.4225,  2.7188,
                  3.015,  3.3112,  3.6075,  3.9038,  4.1013,  4.2])
bond2 = np.array([0.,   7.3007,  21.5706,  26.1297,  27.8516,  32.4116,
                  37.4999,  41.7746,  43.3564,  43.6474,  44.732,  45.7868,
                  46.8455,  47.6992,  48.6035,  49.2856,  50.0781,  50.0285,
                  50.2618,  50.0146])

# V3: DPOUC23A
slip3 = np.array([0.,  0.025,  0.0625,  0.1,  0.1375,  0.4783,  0.8633,
                  1.2483,  1.6333,  2.0183,  2.4033,  2.7883,  3.1733,  3.5583,
                  3.9433,  4.2])
bond3 = np.array([0.,   8.1756,  20.7844,  24.4558,  27.5246,  30.8728,
                  33.1819,  34.8412,  36.1375,  37.1243,  37.6975,  38.3017,
                  38.7764,  39.1447,  39.2908,  39.2778])

plt.figure(facecolor='white', figsize=(15.6 / 2.54, 10.4 / 2.54), dpi=100)
plt.subplots_adjust(
    left=0.17, right=0.95, bottom=0.17, top=0.9, wspace=0.2, hspace=0.2)

plt.plot(slip1, bond1, color='k', label=filenames[0])
plt.plot(slip2, bond2, color='g', label=filenames[1])
plt.plot(slip3, bond3, color='b', label=filenames[2])

slip_avg = np.linspace(0, 4, 50)
bond_avg = np.interp(slip_avg, slip1, bond1) / 3. + np.interp(slip_avg,
                                                              slip2, bond2) / 3. + np.interp(slip_avg, slip3, bond3) / 3.

print [slip_avg]
print [bond_avg]


plt.plot(slip_avg, bond_avg, 'r--', lw=3, label='Average')
#plt.title('Calibrated bond slip laws')
plt.xlabel('material point slip [mm]')
plt.ylabel('bond [N/mm]')
plt.legend(loc=4)
plt.show()

mats = MATSEval(E_m=32701)

fets = FETS1D52ULRH(A_m=100. * 15. - 8. * 1.84,
                    A_f=8. * 1.84)

ts = TStepper(fets_eval=fets,
              L_x=100.,  # half of speciment length
              n_e_x=20  # number of elements
              )
n_dofs = ts.domain.n_dofs

ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
              BCDof(var='u', dof=n_dofs - 1, value=4.0)]
tl = TLoop(ts=ts)


def plt_expri(L_x, slip, bond, d, f, label, color):
    tl.ts.L_x = L_x
    tl.ts.mats_eval.slip = slip.tolist()
    tl.ts.mats_eval.bond = bond.tolist()
    U_record, F_record, sf, sig_m, sig_f = tl.eval()
    n_dof = 2 * ts.domain.n_active_elems + 1
    plt.plot(U_record[:, n_dof], F_record[:, n_dof],
             marker='.', color=color, markevery=5)
    plt.plot(d[d <= 8.0] / 2., f[d <= 8.0] * 1000.,
             '--', color=color, label=label)


plt.figure(facecolor='white', figsize=(15.6 / 2.54, 10.4 / 2.54), dpi=100)
plt.subplots_adjust(
    left=0.17, right=0.95, bottom=0.17, top=0.9, wspace=0.2, hspace=0.2)
plt_expri(98.32, slip1, bond1, d1, f1, label='DPOUC21A', color='k')
plt_expri(97.88, slip2, bond2, d2, f2, label='DPOUC22A', color='g')
plt_expri(97.14, slip3, bond3, d3, f3, label='DPOUC23A', color='b')
#plt.title('verification of pullout force')
plt.xlabel('pull-out slip [mm]')
plt.ylabel('Force [N]')
plt.legend(loc=4)
xlim = 4.5
ylim = 5000.
plt.axis([0., xlim, 0., ylim],)
plt.show()
