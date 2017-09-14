'''
Created on 04.12.2015

@author: Yingxiong
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import derivative


sigma_y = 1.0
K_bar = 0.1
E = 1.0
# g = lambda k: 0.5 - 0.5 * np.exp(-k)
# g = lambda k: 1. / (1 + np.exp(-2 * k + 6.))
g = lambda k: 1. / (1 + np.exp(-2 * k + 6.)) * 0.6 + \
    0.2 * (k > 0.1) + 2 * k * (k <= 0.1)


def plastic_stiffness(w, kappa, sig_e):
    # return -E / (E + K_bar) * 0.5 * np.exp(-kappa) * sig_e + (1 - w) * E *
    # K_bar / (E + K_bar)
    # return -E / (E + K_bar) * 2. * np.exp(6. - 2. * kappa) / (np.exp(6. - 2.
    # * kappa) + 1.) ** 2. * sig_e + (1. - w) * E * K_bar / (E + K_bar)
    return -E / (E + K_bar) * derivative(g, kappa, dx=1e-6) * sig_e + (1. - w) * E * K_bar / (E + K_bar)


sig_e_arr = [0.]
sig_n_arr = [0.]
eps_arr = [0.]
w_arr = [0.]
K_arr = []

sig_e = 0.
sig_n = 0.
eps = 0.
alpha = 0.
kappa = 0.
n = 0


while n <= 960:
    if n <= 800:
        d_eps = 0.005
    else:
        d_eps = -0.005
    eps += d_eps
    sig_e_trial = sig_e + E * d_eps
    f_trial = abs(sig_e_trial) - (sigma_y + K_bar * alpha)
    if f_trial <= 1e-8:
        sig_e = sig_e_trial
    else:
        d_gamma = f_trial / (E + K_bar)
        alpha += d_gamma
        kappa += d_gamma
        sig_e = sig_e_trial - d_gamma * E * np.sign(sig_e_trial)

    w = g(kappa)
    sig_n = (1 - w) * sig_e

    sig_e_arr.append(sig_e)
    sig_n_arr.append(sig_n)
    eps_arr.append(eps)
    w_arr.append(w)
    if f_trial <= 1e-8:
        K_arr.append((1 - w) * E)
    else:
        K_arr.append(plastic_stiffness(w, kappa, sig_e))
    n += 1

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
ax2.set_ylim([0, 1])
plt.legend(loc=1)
# plt.show()

# print np.diff(eps_arr)
plt.figure()
K1 = np.diff(sig_n_arr) / np.diff(eps_arr)

plt.plot(eps_arr[1::], K1, marker='x', label='numerical')
plt.plot(eps_arr[1::], K_arr, label='analytical')
plt.legend()
plt.show()
