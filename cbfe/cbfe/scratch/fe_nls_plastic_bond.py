from envisage.ui.workbench.api import WorkbenchApplication
from mayavi.sources.api import VTKDataSource, VTKFileReader

from traits.api import implements, Int, Array, HasTraits, Instance, \
    Property, cached_property, Constant, Float, List

from ibvpy.api import BCDof
from ibvpy.fets.fets_eval import FETSEval, IFETSEval
from ibvpy.mats.mats1D import MATS1DElastic
from ibvpy.mats.mats1D5.mats1D5_bond import MATS1D5Bond
from ibvpy.mesh.fe_grid import FEGrid
from mathkit.matrix_la.sys_mtx_assembly import SysMtxAssembly
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.widgets import Slider


class MATSEval(HasTraits):

    E_m = Float(10, tooltip='Stiffness of the matrix',
                auto_set=False, enter_set=False)

    E_f = Float(10, tooltip='Stiffness of the fiber',
                auto_set=False, enter_set=False)

    E_b = Float(0.1, tooltip='Bond stiffness')

    sigma_y = Float(0.3,
                    label="sigma_y",
                    desc="Yield stress",
                    enter_set=True,
                    auto_set=False)

    K_bar = Float(0.03,  # 191e-6,
                  label="K",
                  desc="Plasticity modulus",
                  enter_set=True,
                  auto_set=False)

    H_bar = Float(0.0,  # 191e-6,
                  label="H",
                  desc="Hardening modulus",
                  enter_set=True,
                  auto_set=False)

    def get_corr_pred(self, eps, d_eps, sig, t_n, t_n1, alpha, q):
        n_e, n_ip, n_s = eps.shape
        D = np.zeros((n_e, n_ip, 3, 3))
        D[:, :, 0, 0] = self.E_m
        D[:, :, 2, 2] = self.E_f
        sig_trial = sig[:, :, 1] + self.E_b * d_eps[:,:, 1]
        xi_trial = sig_trial - q
        f_trial = abs(xi_trial) - (self.sigma_y + self.K_bar * alpha)
        elas = f_trial <= 1e-8
        plas = f_trial > 1e-8
        E_p = ( self.E_b * ( self.K_bar + self.H_bar ) ) / \
            (self.E_b + self.K_bar + self.H_bar)
        D[:, :, 1, 1] = self.E_b*elas + E_p*plas
        d_sig = np.einsum('...st,...t->...s', D, d_eps)
        sig += d_sig

        d_gamma = f_trial / (self.E_b + self.K_bar + self.H_bar) * plas
        alpha += d_gamma
        q += d_gamma * self.H_bar * np.sign(xi_trial)

        sig[:, :, 1] = sig_trial - d_gamma * self.E_b * np.sign(xi_trial)

        return sig, D, alpha, q

    n_s = Constant(3)

    def get_bond_slip(self):
        '''for plotting the bond slip relationship
        '''
        s_arr = np.hstack((np.linspace(0, 4. * self.sigma_y / self.E_b, 100),
                           np.linspace(4. * self.sigma_y / self.E_b, 3. * self.sigma_y / self.E_b, 25)))
        b_arr = np.zeros_like(s_arr)
        sig_e = 0.
        alpha = 0.

        for i in range(1, len(s_arr)):
            d_eps = s_arr[i] - s_arr[i - 1]
            sig_e_trial = sig_e + self.E_b * d_eps
            f_trial = abs(sig_e_trial) - (self.sigma_y + self.K_bar * alpha)
            if f_trial <= 1e-8:
                sig_e = sig_e_trial
            else:
                d_gamma = f_trial / (self.E_b + self.K_bar)
                alpha += d_gamma
                sig_e = sig_e_trial - d_gamma * self.E_b * np.sign(sig_e_trial)
            b_arr[i] = sig_e

        return s_arr, b_arr


class FETS1D52ULRH(FETSEval):

    '''
    Fe Bar 2 nodes, deformation
    '''

    implements(IFETSEval)

    debug_on = True

    # Dimensional mapping
    dim_slice = slice(0, 1)

    n_nodal_dofs = Int(2)

    dof_r = Array(value=[[-1], [1]])
    geo_r = Array(value=[[-1], [1]])
    vtk_r = Array(value=[[-1.], [1.]])
    vtk_cells = [[0, 1]]
    vtk_cell_types = 'Line'

    n_dof_r = Property
    '''Number of node positions associated with degrees of freedom. 
    '''
    @cached_property
    def _get_n_dof_r(self):
        return len(self.dof_r)

    n_e_dofs = Property
    '''Number of element degrees
    '''
    @cached_property
    def _get_n_dofs(self):
        return self.n_nodal_dofs * self.n_dof_r

    def _get_ip_coords(self):
        offset = 1e-6
        return np.array([[-1 + offset, 0., 0.], [1 - offset, 0., 0.]])

    def _get_ip_weights(self):
        return np.array([1., 1.], dtype=float)

    # Integration parameters
    #
    ngp_r = 2

    def get_N_geo_mtx(self, r_pnt):
        '''
        Return geometric shape functions
        @param r_pnt:
        '''
        r = r_pnt[0]
        N_mtx = np.array([[0.5 - r / 2., 0.5 + r / 2.]])
        return N_mtx

    def get_dNr_geo_mtx(self, r_pnt):
        '''
        Return the matrix of shape function derivatives.
        Used for the conrcution of the Jacobi matrix.
        '''
        return np.array([[-1. / 2, 1. / 2]])

    def get_N_mtx(self, r_pnt):
        '''
        Return shape functions
        @param r_pnt:local coordinates
        '''
        return self.get_N_geo_mtx(r_pnt)

    def get_dNr_mtx(self, r_pnt):
        '''
        Return the derivatives of the shape functions
        '''
        return self.get_dNr_geo_mtx(r_pnt)


class TStepper(HasTraits):

    '''Time stepper object for non-linear Newton-Raphson solver.
    '''

    mats_eval = Property(Instance(MATSEval))
    '''Finite element formulation object.
    '''
    @cached_property
    def _get_mats_eval(self):
        return MATSEval()

    fets_eval = Property(Instance(FETS1D52ULRH))
    '''Finite element formulation object.
    '''
    @cached_property
    def _get_fets_eval(self):
        return FETS1D52ULRH()

    # Number of elements
    n_e_x = 10
    # length
    L_x = Float(15.0)

    domain = Property(Instance(FEGrid), depends_on='L_x')
    '''Diescretization object.
    '''
    @cached_property
    def _get_domain(self):
        # Element definition
        domain = FEGrid(coord_max=(self.L_x,),
                        shape=(self.n_e_x,),
                        fets_eval=self.fets_eval)
        return domain

    bc_list = List(Instance(BCDof))

    J_mtx = Property(depends_on='L_x')
    '''Array of Jacobian matrices.
    '''
    @cached_property
    def _get_J_mtx(self):
        fets_eval = self.fets_eval
        domain = self.domain
        # [ d, n ]
        geo_r = fets_eval.geo_r.T
        # [ d, n, i ]
        dNr_geo = geo_r[:, :, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)
        # [ n_e, n_geo_r, n_dim_geo ]
        elem_x_map = domain.elem_X_map
        # [ n_e, n_ip, n_dim_geo, n_dim_geo ]
        J_mtx = np.einsum('ind,enf->eidf', dNr_geo, elem_x_map)
        return J_mtx

    J_det = Property(depends_on='L_x')
    '''Array of Jacobi determinants.
    '''
    @cached_property
    def _get_J_det(self):
        return np.linalg.det(self.J_mtx)

    B = Property(depends_on='L_x')
    '''The B matrix
    '''
    @cached_property
    def _get_B(self):
        '''Calculate and assemble the system stiffness matrix.
        '''
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval
        domain = self.domain

        n_s = mats_eval.n_s

        n_dof_r = fets_eval.n_dof_r
        n_nodal_dofs = fets_eval.n_nodal_dofs

        n_ip = fets_eval.n_gp
        n_e = domain.n_active_elems
        #[ d, i]
        r_ip = fets_eval.ip_coords[:, :-2].T
        # [ d, n ]
        geo_r = fets_eval.geo_r.T
        # [ d, n, i ]
        dNr_geo = geo_r[:, :, None] * np.array([1, 1]) * 0.5
        # [ i, n, d ]
        dNr_geo = np.einsum('dni->ind', dNr_geo)

        J_inv = np.linalg.inv(self.J_mtx)

        # shape function for the unknowns
        # [ d, n, i]
        Nr = 0.5 * (1. + geo_r[:, :, None] * r_ip[None,:])
        dNr = 0.5 * geo_r[:, :, None] * np.array([1, 1])

        # [ i, n, d ]
        Nr = np.einsum('dni->ind', Nr)
        dNr = np.einsum('dni->ind', dNr)
        Nx = Nr
        # [ n_e, n_ip, n_dof_r, n_dim_dof ]
        dNx = np.einsum('eidf,inf->eind', J_inv, dNr)

        B = np.zeros((n_e, n_ip, n_dof_r, n_s, n_nodal_dofs), dtype='f')
        B_N_n_rows, B_N_n_cols, N_idx = [1, 1], [0, 1], [0, 0]
        B_dN_n_rows, B_dN_n_cols, dN_idx = [0, 2], [0, 1], [0, 0]
        B_factors = np.array([-1, 1], dtype='float_')
        B[:, :,:, B_N_n_rows, B_N_n_cols] = (B_factors[None, None,:] *
                                              Nx[:, :, N_idx])
        B[:, :,:, B_dN_n_rows, B_dN_n_cols] = dNx[:,:,:, dN_idx]

        return B

    def apply_essential_bc(self):
        '''Insert initial boundary conditions at the start up of the calculation.. 
        '''
        self.K = SysMtxAssembly()
        for bc in self.bc_list:
            bc.apply_essential(self.K)

    def apply_bc(self, step_flag, K_mtx, F_ext, t_n, t_n1):
        '''Apply boundary conditions for the current load increement
        '''
        for bc in self.bc_list:
            bc.apply(step_flag, None, K_mtx, F_ext, t_n, t_n1)

    def get_corr_pred(self, step_flag, U, d_U, eps, sig, t_n, t_n1, alpha, q):
        '''Function calculationg the residuum and tangent operator.
        '''
        mats_eval = self.mats_eval
        fets_eval = self.fets_eval
        domain = self.domain
        elem_dof_map = domain.elem_dof_map
        n_e = domain.n_active_elems
        n_dof_r, n_dim_dof = self.fets_eval.dof_r.shape
        n_nodal_dofs = self.fets_eval.n_nodal_dofs
        n_el_dofs = n_dof_r * n_nodal_dofs
        # [ i ]
        w_ip = fets_eval.ip_weights

        d_u_e = d_U[elem_dof_map]
        #[n_e, n_dof_r, n_dim_dof]
        d_u_n = d_u_e.reshape(n_e, n_dof_r, n_nodal_dofs)
        #[n_e, n_ip, n_s]
        d_eps = np.einsum('einsd,end->eis', self.B, d_u_n)

        # update strain
        eps += d_eps

        # material response state variables at integration point
        sig, D, alpha, q = mats_eval.get_corr_pred(
            eps, d_eps, sig, t_n, t_n1, alpha, q)

        # system matrix
        self.K.reset_mtx()
        Ke = np.einsum('i,einsd,eist,eimtf,ei->endmf',
                       w_ip, self.B, D, self.B, self.J_det)

        self.K.add_mtx_array(
            Ke.reshape(-1, n_el_dofs, n_el_dofs), elem_dof_map)

        # internal forces
        # [n_e, n_n, n_dim_dof]
        Fe_int = np.einsum('i,eis,einsd,ei->end',
                           w_ip, sig, self.B, self.J_det)
        F_int = -np.bincount(elem_dof_map.flatten(), weights=Fe_int.flatten())
        self.apply_bc(step_flag, self.K, F_int, t_n, t_n1)
        return F_int, self.K, eps, sig, alpha, q


class TLoop(HasTraits):

    ts = Instance(TStepper)
    d_t = Float(0.03)
    t_max = Float(1.0)
    k_max = Int(200)
    tolerance = Float(1e-5)

    def eval(self):

        self.ts.apply_essential_bc()

        t_n = 0.
        t_n1 = t_n
        n_dofs = self.ts.domain.n_dofs
        n_e = self.ts.domain.n_active_elems
        n_ip = self.ts.fets_eval.n_gp
        n_s = self.ts.mats_eval.n_s
        U_k = np.zeros(n_dofs)
        eps = np.zeros((n_e, n_ip, n_s))
        sig = np.zeros((n_e, n_ip, n_s))
        alpha = np.zeros((n_e, n_ip))
        q = np.zeros((n_e, n_ip))

        U_record = np.zeros(n_dofs)
        F_record = np.zeros(n_dofs)
        sf_record = np.zeros(2 * n_e)
        t_record = [t_n]
        eps_record = [np.zeros_like(eps)]
        sig_record = [np.zeros_like(sig)]

        while t_n1 <= self.t_max:
            t_n1 = t_n + self.d_t
            k = 0
            step_flag = 'predictor'
            d_U = np.zeros(n_dofs)
            d_U_k = np.zeros(n_dofs)
#             print '=============================='
#             print 't=====', t_n
            while k < self.k_max:
                R, K, eps, sig, alpha, q = self.ts.get_corr_pred(
                    step_flag, U_k, d_U_k, eps, sig, t_n, t_n1, alpha, q)
                F_ext = -R
#                 print 'R1', R
                K.apply_constraints(R)
#                 print 'R2', R
                d_U_k = K.solve()
#                 print k, d_U
                d_U += d_U_k
                if np.linalg.norm(R) < self.tolerance:
                    F_record = np.vstack((F_record, F_ext))
                    U_k += d_U
                    U_record = np.vstack((U_record, U_k))
                    sf_record = np.vstack((sf_record, sig[:, :, 1].flatten()))
                    eps_record.append(np.copy(eps))
                    sig_record.append(np.copy(sig))
                    t_record.append(t_n1)
                    break
                k += 1
                step_flag = 'corrector'

            t_n = t_n1
        return U_record, F_record, sf_record, np.array(t_record), eps_record, sig_record

if __name__ == '__main__':

    ts = TStepper()
    n_dofs = ts.domain.n_dofs
    ts.bc_list = [BCDof(var='u', dof=0, value=0.0),
                  BCDof(var='u', dof=n_dofs - 1, value=5.0)]

    tl = TLoop(ts=ts)
    U_record, F_record, sf_record, t_record, eps_record, sig_record = tl.eval()

    n_dof = 2 * ts.domain.n_active_elems + 1
#     print F_record[:, n_dof]
#     print U_record[:, n_dof]

    ax1 = plt.subplot(231)
    slip, bond = ts.mats_eval.get_bond_slip()
    l_bs, = ax1.plot(slip, bond)
    ax1.set_title('bond-slip law')

    ax2 = plt.subplot(232)
    l_po, = ax2.plot(U_record[:, n_dof], F_record[:, n_dof])
    marker_po, = ax2.plot(U_record[-1, n_dof], F_record[-1, n_dof], 'ro')
    ax2.set_title('pull-out force-displacement curve')

    ax3 = plt.subplot(234)
    X = np.linspace(0, ts.L_x, ts.n_e_x + 1)
    X_ip = np.repeat(X, 2)[1:-1]
    l_sf, = ax3.plot(X_ip, sf_record[-1, :])
    ax3.set_ylim([0, 0.5])
    ax3.set_title('shear flow in the bond interface')

    ax4 = plt.subplot(235)
    U = np.reshape(U_record[-1, :], (-1, 2)).T
    l_u0, = ax4.plot(X, U[0])
    l_u1, = ax4.plot(X, U[1])
    l_us, = ax4.plot(X, U[1] - U[0])
    ax4.set_title('displacement and slip')

    ax5 = plt.subplot(233)
    l_eps0, = ax5.plot(X_ip, eps_record[-1][:, :, 0].flatten())
    l_eps1, = ax5.plot(X_ip, eps_record[-1][:, :, 2].flatten())
    ax5.set_title('strain')

    ax6 = plt.subplot(236)
    l_sig0, = ax6.plot(X_ip, sig_record[-1][:, :, 0].flatten())
    l_sig1, = ax6.plot(X_ip, sig_record[-1][:, :, 2].flatten())
    ax6.set_title('stress')

    ax_k_bar = plt.axes([0.1, 0.51, 0.35, 0.02])
    s_k_bar = Slider(
        ax_k_bar, 'K_bar', -0.01, 0.05, valfmt='%1.3f', valinit=0.03)

    ax_e_b = plt.axes([0.1, 0.48, 0.35, 0.02])
    s_e_b = Slider(
        ax_e_b, 'E_b', 0.05, 0.35, valfmt='%1.2f', valinit=0.1)

    ax_sig_y = plt.axes([0.1, 0.45, 0.35, 0.02])
    s_sig_y = Slider(
        ax_sig_y, 'sigma_y', 0.03, 0.30, valfmt='%1.2f', valinit=0.30)

    def update_k_bar(val):
        global U_record, F_record, sf_record, t_record, eps_record, sig_record
        ts.mats_eval.K_bar = s_k_bar.val
        slip, bond = ts.mats_eval.get_bond_slip()
        l_bs.set_data(slip, bond)
        U_record, F_record, sf_record, t_record, eps_record, sig_record = tl.eval()
        l_po.set_data(U_record[:, n_dof], F_record[:, n_dof])
        marker_po.set_data(U_record[-1, n_dof], F_record[-1, n_dof])
        l_sf.set_ydata(sf_record[-1, :])
        U = np.reshape(U_record[-1, :], (-1, 2)).T
        l_u0.set_ydata(U[0])
        l_u1.set_ydata(U[1])
        l_us.set_ydata(U[1] - U[0])
        l_eps0.set_ydata(eps_record[-1][:, :, 0].flatten())
        l_eps1.set_ydata(eps_record[-1][:, :, 2].flatten())
        l_sig0.set_ydata(sig_record[-1][:, :, 0].flatten())
        l_sig1.set_ydata(sig_record[-1][:, :, 2].flatten())
    s_k_bar.on_changed(update_k_bar)

    def update_e_b(val):
        global U_record, F_record, sf_record, t_record, eps_record, sig_record
        ts.mats_eval.E_b = s_e_b.val
        slip, bond = ts.mats_eval.get_bond_slip()
        l_bs.set_data(slip, bond)
        U_record, F_record, sf_record, t_record, eps_record, sig_record = tl.eval()
        l_po.set_data(U_record[:, n_dof], F_record[:, n_dof])
        marker_po.set_data(U_record[-1, n_dof], F_record[-1, n_dof])
        l_sf.set_ydata(sf_record[-1, :])
        U = np.reshape(U_record[-1, :], (-1, 2)).T
        l_u0.set_ydata(U[0])
        l_u1.set_ydata(U[1])
        l_us.set_ydata(U[1] - U[0])
        l_eps0.set_ydata(eps_record[-1][:, :, 0].flatten())
        l_eps1.set_ydata(eps_record[-1][:, :, 2].flatten())
        l_sig0.set_ydata(sig_record[-1][:, :, 0].flatten())
        l_sig1.set_ydata(sig_record[-1][:, :, 2].flatten())
    s_e_b.on_changed(update_e_b)

    def update_sig_y(val):
        global U_record, F_record, sf_record, t_record, eps_record, sig_record
        ts.mats_eval.sigma_y = s_sig_y.val
        slip, bond = ts.mats_eval.get_bond_slip()
        l_bs.set_data(slip, bond)
        U_record, F_record, sf_record, t_record, eps_record, sig_record = tl.eval()
        l_po.set_data(U_record[:, n_dof], F_record[:, n_dof])
        marker_po.set_data(U_record[-1, n_dof], F_record[-1, n_dof])
        l_sf.set_ydata(sf_record[-1, :])
        U = np.reshape(U_record[-1, :], (-1, 2)).T
        l_u0.set_ydata(U[0])
        l_u1.set_ydata(U[1])
        l_us.set_ydata(U[1] - U[0])
        l_eps0.set_ydata(eps_record[-1][:, :, 0].flatten())
        l_eps1.set_ydata(eps_record[-1][:, :, 2].flatten())
        l_sig0.set_ydata(sig_record[-1][:, :, 0].flatten())
        l_sig1.set_ydata(sig_record[-1][:, :, 2].flatten())
    s_sig_y.on_changed(update_sig_y)

    ax_t = plt.axes([0.53, 0.51, 0.35, 0.02])
    s_t = Slider(
        ax_t, 'time', 0.00, 1.02, valfmt='%1.2f', valinit=1.02)

    def update_t(val):
        t = s_t.val
        idx = (np.abs(t - t_record)).argmin()
        marker_po.set_data(U_record[idx, n_dof], F_record[idx, n_dof])
        l_sf.set_ydata(sf_record[idx, :])
        U = np.reshape(U_record[idx, :], (-1, 2)).T
        l_u0.set_ydata(U[0])
        l_u1.set_ydata(U[1])
        l_us.set_ydata(U[1] - U[0])
        l_eps0.set_ydata(eps_record[idx][:, :, 0].flatten())
        l_eps1.set_ydata(eps_record[idx][:, :, 2].flatten())
        l_sig0.set_ydata(sig_record[idx][:, :, 0].flatten())
        l_sig1.set_ydata(sig_record[idx][:, :, 2].flatten())
    s_t.on_changed(update_t)

    ax_l_x = plt.axes([0.53, 0.48, 0.35, 0.02])
    s_l_x = Slider(ax_l_x, 'length', 5, 15, valfmt='%1.0f', valinit=15)

    def update_l_x(val):
        global U_record, F_record, sf_record, t_record, eps_record, sig_record
        ts.L_x = s_l_x.val
        U_record, F_record, sf_record, t_record, eps_record, sig_record = tl.eval()
        l_po.set_data(U_record[:, n_dof], F_record[:, n_dof])
        marker_po.set_data(U_record[-1, n_dof], F_record[-1, n_dof])
        X = np.linspace(0, ts.L_x, ts.n_e_x + 1)
        X_ip = np.repeat(X, 2)[1:-1]
        l_sf.set_data(X_ip, sf_record[-1, :])
        U = np.reshape(U_record[-1, :], (-1, 2)).T
        l_u0.set_data(X, U[0])
        l_u1.set_data(X, U[1])
        l_us.set_data(X, U[1] - U[0])
        l_eps0.set_data(X_ip, eps_record[-1][:, :, 0].flatten())
        l_eps1.set_data(X_ip, eps_record[-1][:, :, 2].flatten())
        l_sig0.set_data(X_ip, sig_record[-1][:, :, 0].flatten())
        l_sig1.set_data(X_ip, sig_record[-1][:, :, 2].flatten())
        ax3.set_xlim(0, ts.L_x)
        ax4.set_xlim(0, ts.L_x)
        ax5.set_xlim(0, ts.L_x)
        ax6.set_xlim(0, ts.L_x)
    s_l_x.on_changed(update_l_x)

    plt.subplots_adjust(
        left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.6, wspace=0.2)
    mng = plt.get_current_fig_manager()
    mng.frame.Maximize(True)
    plt.show()
