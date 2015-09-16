'''
Created on Sep 12, 2015

@author: rch
'''

from mayavi.sources.api import VTKDataSource, VTKFileReader

import numpy as np

from ibvpy.mesh.fe_grid import FEGrid
from ibvpy.fets.fets2D import FETS2D4Q

from ibvpy.mats.mats2D.mats2D_elastic.mats2D_elastic import MATS2DElastic
fets_eval = FETS2D4Q(mats_eval=MATS2DElastic(E=10.))

if __name__ == '__main__':

    # Number of elements
    n_E = 2
    # length
    L_x = 10.0
    # Discretization
    domain = FEGrid(coord_max=(L_x, L_x),
                    shape = (n_E, n_E),
                    fets_eval = fets_eval)
    # element array with nodal coordinates
    # [ e, n, d ]
    E_X = domain.elem_X_map
    # [ r, i ]
    r_ip = fets_eval.ip_coords[:, :-1].T
    # [ i ]
    w_ip = fets_eval.ip_weights
    # [ d, n ]
    geo_r = fets_eval.geo_r.T
    # [ d, n, i ]
    dNr_geo = geo_r[:, :, None] * (1 + np.flipud( r_ip )[:, None,:] * np.flipud(geo_r)[:,:, None] ) / 4.0
    # [ i, n, d ]
    dNr_geo = np.einsum('dni->ind', dNr_geo)
    # [ e, i, d, f ]
    J_mtx = np.einsum('ind,enf->eidf', dNr_geo, E_X)
    J_inv = np.linalg.inv(J_mtx)
    J_det = np.linalg.det(J_mtx)

    # shape function for the unknowns are identical with the geomeetrical
    # approximation
    dNr = dNr_geo

    print 'dNr', dNr[0, 0]
    print 'J_inv', J_inv[0, 0]
    print 'dNx', np.dot(J_inv[0, 0].T, dNr[0, 0])
    print 'J_inv', J_inv.shape
    print 'J_dNr', dNr.shape
    # [ e, i, n, d ]
    dNx = np.einsum('eidf,inf->eind', J_inv, dNr)

    #######
    # B [ e, i, s, n, d ]
    dNx0 = dNx[0, 0, 0]
    print 'dN_X', dNx0

    imap = np.array([[0, 1], [0, 0]], dtype='int')

    print 'imap', dNx0[imap]

    # Map to B matrix

    # assemble D matrix

    # Make the K matrix
