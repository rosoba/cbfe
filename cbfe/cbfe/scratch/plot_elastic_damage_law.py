'''
Created on 04.12.2015

@author: Yingxiong
'''
import numpy as np
from matplotlib import pyplot as plt


E = 170000


def g_f(eps_f, eps_f_max):
    e = eps_f * (eps_f > eps_f_max) + eps_f_max * (eps_f <= eps_f_max)
    w_f = 500. * (e - 0.011) * (e > 0.011) * \
        (e <= 0.013) + 1. * (e > 0.013)
    return np.amin(w_f, 0.99999999), e

# derivative of g_f


def d_g_f(eps_f, eps_f_max):
    dx = 1e-6
    return (g_f(eps_f + dx, eps_f_max)[0] - g_f(eps_f, eps_f_max)[0]) / dx


def plastic_stiffness(w, eps_f, eps_f_max, sig_e):
    return -d_g_f(eps_f, eps_f_max) * sig_e + (1 - w) * E


eps_arr = np.hstack([np.linspace(0, 0.012, 240), np.linspace(
    0.012, 0.006, 120), np.linspace(0.006, 0.014, 160)])
w_arr = []
sig_e_arr = []
sig_n_arr = []
K_arr = []

eps_f_max = 0.
for eps in eps_arr:
    sig_e = E * eps
    sig_e_arr.append(sig_e)
    w, eps_f_max = g_f(eps, eps_f_max)
    w_arr.append(w)
    sig_n_arr.append((1 - w) * sig_e)
    K_arr.append(plastic_stiffness(w, eps, eps_f_max, sig_e))

fig, ax1 = plt.subplots()
ax1.plot(eps_arr, sig_e_arr, label='effective stress')
ax1.plot(eps_arr, sig_n_arr, label='nominal stress')
ax1.set_ylabel('stress')
ax1.set_xlabel('strain')
# ax1.set_ylim(0, 0.095)
plt.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(eps_arr, w_arr, '--', label='damage factor')
ax2.set_ylabel('damage factor')
ax2.set_ylim([0, 1.2])
plt.legend(loc=1)
# plt.show()

# print np.diff(eps_arr)
plt.figure()
K1 = np.diff(sig_n_arr) / np.diff(eps_arr)
plt.plot(eps_arr[1::], K1, marker='x', label='numerical')
plt.plot(eps_arr, K_arr, label='analytical')
plt.legend()
plt.show()
