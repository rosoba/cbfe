'''
Created on Apr 6, 2018

@author: rch

Study of the scaling of Weibull distribution
based on the prescibed initial crack level
'''

from scipy.special import gamma
from scipy.stats import norm, weibull_min

import matplotlib.pyplot as plt
import numpy as np
import pylab as p
from stats.misc.random_field.random_field_1D import RandomField


l_rho = 4  # autocorrelation length
sig_min = 3.224  # minimum cracking strength
# m = 4.0  # shape parameter
length = 350.  # specimen length


def f_L(L_c, m):  # size scaling
    return (l_rho / (L_c + l_rho)) ** (1 / m)


m_arr = np.array([3, 5, 10], dtype=np.float_)
s_arr = sig_min / (f_L(length, m_arr) * gamma(1 + 1 / m_arr))
sig_arr = np.linspace(0, 20, 100)

params = {'legend.fontsize': 12,
          #         'legend.linewidth': 2,
          u'font.size': 12,
          u'font.family': 'Times New Roman',
          u'font.style': 'normal'}
plt.rcParams.update(params)

fig = p.figure(facecolor='white', figsize=(15.6 / 2.54, 7.8 / 2.54), dpi=100)
fig.subplots_adjust(
    left=0.1, right=0.96, bottom=0.18, top=0.94, wspace=0.3, hspace=0.3)

for s, m in zip(s_arr, m_arr):
    random_field = RandomField(seed=False,
                               lacor=l_rho,
                               length=length,
                               nx=1000,
                               nsim=1,
                               loc=.0,
                               scale=s,
                               shape=m,
                               distr_type='Weibull')

    p.subplot(121)
    p.plot(random_field.xgrid, random_field.random_field)
    p.ylim(ymin=0.0)

    p.subplot(122)
    p.subplot(122).set_ylabel('Probability')
    p.subplot(122).set_xlabel('Composite Stress [MPa]')
    sig_pdf_arr = weibull_min.pdf(sig_arr, m, 0, s)
    p.plot(sig_arr, sig_pdf_arr, label='m = %1.0f ' % random_field.shape)
    p.legend()

print 's', s_arr
print 'm', m_arr
p.subplot(121).set_ylabel('Composite Stress [MPa]')
p.subplot(121).set_xlabel('Position along specimen [mm]')
p.plot([0, length], [sig_min, sig_min], color='black')
p.show()
