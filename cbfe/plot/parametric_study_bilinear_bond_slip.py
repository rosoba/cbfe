'''
Created on 15.02.2017

@author: Yingxiong
'''

import matplotlib.pyplot as plt
import numpy as np
from cbfe.scratch.fe_nls_solver_incre1 import MATSEval, FETS1D52ULRH, TStepper, TLoop


from ibvpy.api import BCDof

ts = TStepper(n_e_x=100.)
n_dofs = ts.domain.n_dofs
tl = TLoop(ts=ts)

# print n_dofs
#
# tl.ts.mats_eval.slip = [0., 100.]
# tl.ts.mats_eval.bond = [0., 100.]


# bs = ts.mats_eval.b_s_law
#
# bs1 = ts.mats_eval.G
#
# print bs(-4.)
# print bs1([-4., -5.])
#
#
# print dfsdfsdfsdf


def predict(L_x, u, slip, bond):

    tl.ts.bc_list = [BCDof(var='u', dof=n_dofs - 2, value=0.0),
                     BCDof(var='u', dof=n_dofs - 1, value=u)]
    tl.ts.L_x = L_x
    tl.ts.mats_eval.slip = slip.tolist()
    tl.ts.mats_eval.bond = bond.tolist()
    U_record, F_record, sig_record = tl.eval()
    return U_record, F_record, sig_record

# slip = np.array([0., 0.5, 1., 1.5, 2.0, 2.5, 3.0])
# bond = np.array([0., 25., 40., 53, 65, 50, 115])

# slip = np.array([0., 1e-8, 100.])
# bond = np.array([0., 40., 40.])

# slip = np.array([0., 0.4, 4.5])
# bond = np.array([0., 50., 50])


slip = np.array([0.0, 0.15000000000000002, 0.40000000000000002, 2.0, 4.0])
bond = np.array([0.0, 18.000000000170711, 47.937059019961012,
                 50.021030274276697, 50.652307876262164])


slip2 = np.array([0., 0.5, 1.5, 2.5,  3.5,  4.5])
bond2 = np.array([0., 60., 40, 60,  40,  60])


def nl_bond(slip):
    x = slip
    y = np.zeros_like(x)
    y[x < 1.05] = 0.1 * x[x < 1.05] - 0.05 * x[x < 1.05] ** 2
    y[x > 1.05] = 0.1 * 1.05 - 0.05 * \
        1.05 ** 2 - 0.005 * (x[x > 1.05] - 1.05)
    return 1200. * y

# slip2 = np.linspace(0, 3, 100)
# bond2 = nl_bond(slip2)
# plt.plot(x, nl_bond(x))
# plt.ylim(0, 80)
# plt.show()


plt.subplot(221)
# plt.plot(slip, bond)
plt.plot(slip2, bond2)
plt.xlabel('slip')
plt.ylabel('bond')

U1, F1, sig1 = predict(700, 4, slip, bond)
U2, F2, sig2 = predict(700, 4, slip2, bond2)
# U2, F2, sig2 = predict(700, 4.5, slip2, bond2)

# U3, F3, sig3 = predict(1000, 3., slip2, bond2)

plt.subplot(222)
# plt.plot(U1[:, n_dofs - 1], F1[:, n_dofs - 1], lw=2)
plt.plot(U2[:, n_dofs - 1], F2[:, n_dofs - 1])
np.savetxt('D:\\1.txt', np.vstack((U2[:, n_dofs - 1], F2[:, n_dofs - 1])))
# plt.plot(U3[:, n_dofs - 1], F3[:, n_dofs - 1])
plt.xlabel('displacement')
plt.ylabel('pull-out force')

plt.subplot(223)
X = np.linspace(0, ts.L_x, ts.n_e_x + 1)
u1_node = np.reshape(U1[-1, :], (-1, 2)).T
u2_node = np.reshape(U2[-1, :], (-1, 2)).T
# u3_node = np.reshape(U3[-1, :], (-1, 2)).T

# plt.plot(X,  u1_node[1] - u1_node[0])
# plt.plot(X, u2_node[1])
# plt.plot(X, u2_node[0])
plt.plot(X,  u2_node[1] - u2_node[0])
# plt.plot(X,  u3_node[1] - u3_node[0])
plt.ylim(0, 6)

plt.xlabel('x')
plt.ylabel('slip')

plt.subplot(224)
X_ip = np.repeat(X, 2)[1:-1]
# plt.plot(X_ip, sig1[-1, :])
plt.plot(X_ip, sig2[-1, :])
# plt.plot(X_ip, sig3[-1, :])
plt.xlabel('x')
plt.ylabel('shear flow')


plt.show()


# slip2 = u2_node[1] - u2_node[0]
#
# print slip2
#
# print np.argmin(np.abs(slip2 - 3.5))
# x2_slip2 = X[np.argmin(np.abs(slip2 - 3.5))]
# print x2_slip2
# #
# print X_ip
# #
# print len(X_ip)
# print len(slip2)
# #
# plt.plot(X, slip2)
# plt.show()
#
#
# x = np.linspace(0, 700, 2000)
# y = np.interp(x, X_ip, sig1[-1, :])
#
#
# print 'F_integral', np.trapz(y, x)
# print 'F', F1[-1][-1]
#
# print 'slip2', slip2[np.argmin(np.abs(slip2 - 3.5))]
#
# x2 = np.linspace(0, x2_slip2, 2000)
# y2 = np.interp(x2, X_ip, sig2[-1, :])
#
# print 'F2_integral', np.trapz(y2, x2)
