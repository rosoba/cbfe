'''
Created on 27.03.2018

@author: Yingxiong
'''
from os.path import join

from cb import NonLinearCB
from fe_nls_solver_cb import MATSEval, FETS1D52ULRH, TStepper, TLoop
import matplotlib.pyplot as plt
from matresdev.db.simdb.simdb import simdb
import numpy as np
from stats.misc.random_field.random_field_1D import RandomField
from tensile_test import CompositeTensileTest

params = {'legend.fontsize': 12,
          #         'legend.linewidth': 2,
          u'font.size': 12,
          u'font.family': 'Times New Roman',
          u'font.style': 'normal'}
plt.rcParams.update(params)

folder = join(simdb.exdata_dir,
              'tensile_tests',
              'buttstrap_clamping',
              '2018-01-07_TTb_Leipzig',
              )


DKBC11 = np.loadtxt(join(folder, 'DKBC11.txt'))
DKBC12 = np.loadtxt(join(folder,  'DKBC12.txt'))
DKBC13 = np.loadtxt(join(folder,  'DKBC13.txt'))
DKBC14 = np.loadtxt(join(folder,  'DKBC14.txt'))
DKBC15 = np.loadtxt(join(folder,  'DKBC15.txt'))
DKBC16 = np.loadtxt(join(folder,  'DKBC16.txt'))

DKUC21 = np.loadtxt(join(folder, 'DKUC21.txt'))
DKUC22 = np.loadtxt(join(folder, 'DKUC22.txt'))
DKUC23 = np.loadtxt(join(folder, 'DKUC23.txt'))
DKUC31 = np.loadtxt(join(folder, 'DKUC31.txt'))
DKUC32 = np.loadtxt(join(folder, 'DKUC32.txt'))

plt.plot(DKBC11[0] / 355., DKBC11[1] * 1000. / 1500., 'k', label='short')
plt.plot(DKBC12[0] / 358., DKBC12[1] * 1000. / 1500., 'k')
plt.plot(DKBC13[0] / 357., DKBC13[1] * 1000. / 1500., 'k')
plt.plot(DKBC14[0] / 357., DKBC14[1] * 1000. / 1500., 'k')
plt.plot(DKBC15[0] / 360., DKBC15[1] * 1000. / 1500., 'k')
plt.plot(DKBC16[0] / 355., DKBC16[1] * 1000. / 1500., 'k')

plt.plot(DKUC21[0] / 357., DKUC21[1] * 1000. / 1500., 'b', label='long')
plt.plot(DKUC22[0] / 359., DKUC22[1] * 1000. / 1500., 'b')
plt.plot(DKUC23[0] / 343., DKUC23[1] * 1000. / 1500., 'b')
plt.plot(DKUC31[0] / 352., DKUC31[1] * 1000. / 1500., 'b')
plt.plot(DKUC32[0] / 349., DKUC32[1] * 1000. / 1500., 'b')

# E11 = (DKBC11[1][400] - DKBC11[1][200]) * 1000. / \
#     1500. * 350 / (DKBC11[0][400] - DKBC11[0][200])
# plt.plot([0, 0.0075], [0, DKBC11[0][-1] / 350. * E11])
#
# E12 = (DKBC12[1][400] - DKBC12[1][200]) * 1000. / \
#     1500. * 350 / (DKBC12[0][400] - DKBC12[0][200])
# plt.plot([0, 0.0075], [0, DKBC12[0][-1] / 350. * E12])
#
# E13 = (DKBC13[1][400] - DKBC13[1][200]) * 1000. / \
#     1500. * 350 / (DKBC13[0][400] - DKBC13[0][200])
# plt.plot([0, 0.0075], [0, DKBC13[0][-1] / 350. * E13])
#
# E14 = (DKBC14[1][400] - DKBC14[1][200]) * 1000. / \
#     1500. * 350 / (DKBC14[0][400] - DKBC14[0][200])
# plt.plot([0, 0.0075], [0, DKBC14[0][-1] / 350. * E14])
#
# E15 = (DKBC15[1][400] - DKBC15[1][200]) * 1000. / \
#     1500. * 350 / (DKBC15[0][400] - DKBC15[0][200])
# plt.plot([0, 0.0075], [0, DKBC15[0][-1] / 350. * E15])
#
# E16 = (DKBC16[1][400] - DKBC16[1][200]) * 1000. / \
#     1500. * 350 / (DKBC16[0][400] - DKBC16[0][200])
# plt.plot([0, 0.0075], [0, DKBC16[0][-1] / 350. * E16])
#
# E21 = (DKUC21[1][400] - DKUC21[1][200]) * 1000. / \
#     1500. * 350 / (DKUC21[0][400] - DKUC21[0][200])
# plt.plot([0, 0.0075], [0, DKUC21[0][-1] / 350. * E21])
#
# E22 = (DKUC22[1][400] - DKUC22[1][200]) * 1000. / \
#     1500. * 350 / (DKUC22[0][400] - DKUC22[0][200])
# plt.plot([0, 0.0075], [0, DKUC22[0][-1] / 350. * E22])
#
# E23 = (DKUC23[1][400] - DKUC23[1][200]) * 1000. / \
#     1500. * 350 / (DKUC23[0][400] - DKUC23[0][200])
# plt.plot([0, 0.0075], [0, DKUC23[0][-1] / 350. * E23])

# vf = 8. * 1.85 / 1500

# E11 = (DKBC11[1][400] - DKBC11[1][200]) / (DKBC11[0][400] - DKBC11[0][200])
# plt.plot([0, 0.0075], [0, 0.0075 * E11])

# print E11 / vf, E12 / vf, E13 / vf, E14 / vf, E15 / vf, E16 / vf, E21 /
# vf, E22 / vf, E23 / vf

plt.legend()
# plt.show()


if False:
    plt.figure()

    n_2 = 'k'
    n_3 = 'g'

    plt.plot(DKBC11[0] / 355., DKBC11[1] *
             1000. / 1500., n_2, label='2 cracks')
    plt.plot(DKBC12[0] / 358., DKBC12[1] * 1000. / 1500., n_2)
    plt.plot(DKBC13[0] / 357., DKBC13[1] * 1000. / 1500., n_2)
    plt.plot(DKBC14[0] / 357., DKBC14[1] * 1000. / 1500., n_2)
    plt.plot(DKBC15[0] / 360., DKBC15[1] *
             1000. / 1500., n_3, label='3 cracks')
    plt.plot(DKBC16[0] / 355., DKBC16[1] * 1000. / 1500., n_3)
    plt.legend()

    plt.figure()

    plt.plot(DKUC21[0] / 357., DKUC21[1] *
             1000. / 1500., n_3, label='3 cracks')
    plt.plot(DKUC22[0] / 359., DKUC22[1] *
             1000. / 1500., n_2, label='2 cracks')
    plt.plot(DKUC23[0] / 343., DKUC23[1] * 1000. / 1500., n_3)
    plt.plot(DKUC31[0] / 352., DKUC31[1] * 1000. / 1500., n_2)
    plt.plot(DKUC32[0] / 349., DKUC32[1] * 1000. / 1500., n_2)
    plt.legend()

    # plt.show()

    plt.show()

# sig_mu = 3.
# Em = 32701
# Ef = 300000
# sig_cu = 22
# vf = 8. * 1.85 / 1500
# vm = 1. - vf
# Ec = Ef * vf + Em * vm
# alpha = Em * vm / (Ef * vf)
#
#
# def ACK(sig_mu):
# ACK model
#     return [0, sig_mu / Em, (1 + 0.666 * alpha) * sig_mu / Em, (1 + 0.666 * alpha) * sig_mu / Em + (sig_cu - sig_mu * Ec / Em) / (Ef * vf)], [0, sig_mu / Em * Ec, sig_mu / Em * Ec, sig_cu]
# ack_eps, ack_sig = ACK(sig_mu)
# plt.plot(ack_eps, ack_sig, 'k', lw=2)
# plt.title('Ef = 300GPa')
# plt.show()


mats = MATSEval(E_f=283350.,
                E_m=32701.)

fets = FETS1D52ULRH(A_m=100. * 15. - 8. * 1.84,
                    A_f=8. * 1.84)

ts = TStepper(mats_eval=mats,
              fets_eval=fets)

slip = np.array([0.,  0.08163265,  0.16326531,  0.24489796,  0.32653061,
                 0.40816327,  0.48979592,  0.57142857,  0.65306122,  0.73469388,
                 0.81632653,  0.89795918,  0.97959184,  1.06122449,  1.14285714,
                 1.2244898,  1.30612245,  1.3877551,  1.46938776,  1.55102041,
                 1.63265306,  1.71428571,  1.79591837,  1.87755102,  1.95918367,
                 2.04081633,  2.12244898,  2.20408163,  2.28571429,  2.36734694,
                 2.44897959,  2.53061224,  2.6122449,  2.69387755,  2.7755102,
                 2.85714286,  2.93877551,  3.02040816,  3.10204082,  3.18367347,
                 3.26530612,  3.34693878,  3.42857143,  3.51020408,  3.59183673,
                 3.67346939,  3.75510204,  3.83673469,  3.91836735,  4.]).tolist()

bond = np.array([0.,  22.8196719,  28.46209226,  29.68753463,
                 30.8023522,  31.81904245,  32.784257,  33.66000354,
                 34.53025696,  35.35037697,  36.17049699,  36.97111783,
                 37.56621184,  37.95859091,  38.35096998,  38.74334905,
                 38.99005988,  39.20511549,  39.4201711,  39.65136099,
                 39.94290047,  40.21273858,  40.48240333,  40.74687194,
                 41.00761627,  41.26029761,  41.49180967,  41.71649435,
                 41.940841,  42.16518765,  42.38315886,  42.58626232,
                 42.78936577,  42.99246923,  43.21251614,  43.43229047,
                 43.65063078,  43.86709291,  44.05708319,  44.24611786,
                 44.42858805,  44.6117749,  44.79588207,  44.97998923,
                 45.15764457,  45.2578736,  45.34191603,  45.42595845,
                 45.50215681,  45.53411323]).tolist()

cb = NonLinearCB(A_c=100. * 15.,
                 tstepper=ts,
                 max_w_p=1.84 * 8 * 2167,  # [N]
                 slip=slip,
                 bond=bond,
                 n_BC=100)

'''
uncomment this loop in order to run multiple random simulations at once
'''
# n_crack_end = 0
# while n_crack_end < 3:


random_field = RandomField(seed=False,
                               lacor=4,
                               length=350,
                               nx=1000,
                               nsim=1,
                               loc=.0,
                               # stdev=0.2,
                               # mean=3.22,
                               shape=5,
                               scale=8.60718591,
                               distr_type='Weibull')
    
ctt = CompositeTensileTest(n_x=1000,
                               L=350,
                               cb=cb,
                               sig_mu_x=random_field.random_field,
                               strength=21.26)
    
    
sig_c_i, z_x_i, BC_x_i, sig_c_u, n_crack = ctt.get_cracking_history()
    
#     print n_crack
#     n_crack_end = n_crack

load_arr1 = np.unique(
    np.hstack((np.linspace(0, sig_c_u, 100), sig_c_i)))
eps_c_arr1 = ctt.get_eps_c_arr(sig_c_i, z_x_i, BC_x_i, load_arr1)

plt.figure(facecolor='white', figsize=(8.8 / 2.54, 8.8 / 2.54), dpi=100)
plt.subplots_adjust(
    left=0.19, right=0.96, bottom=0.15, top=0.93, wspace=0.25, hspace=0.2)
# plot diagram with composite stress vs. strain
# plt.plot(eps_c_arr1 * 1000, load_arr1, 'k', lw=2)
# plt.ylabel('Composite Stress [MPa]')
# plot diagram with textile stress vs. strain
plt.plot(eps_c_arr1 * 1000, load_arr1 *
         (fets.A_f + fets.A_m) / fets.A_f, 'k', lw=2)
plt.title('Matrix - RF shape parameter %1.1f ' %
          random_field.shape)
plt.ylabel('Textile Stress [MPa]')
plt.xlabel('Strain [1E+3]')
plt.show()
