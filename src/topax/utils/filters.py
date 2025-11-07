'''

Filters for topology optimization
    
Last modified: 29/06/2024

'''

import jax
import jax.numpy as np
import scipy
import numpy as onp

from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper


# Helmholtz type PDE-based filter

class HelmholtzFilter:
    
    '''
    
    Lazarov, Boyan Stefanov, and Ole Sigmund. 
    "Filters in topology optimization based on Helmholtz‐type differential equations." 
    IJNME 86.6 (2011): 765-781.
    
    '''
    
    def __init__(self, problem, rmin):
        
        self.citation = 'Helmholtz2010'
        
        problem = HelmholtzPDE(problem.mesh[0], vec=1, 
                                     dim=problem.dim,
                                     ele_type=problem.fes[0].ele_type)
        
        fwd_pred = ad_wrapper(problem, use_petsc=True, use_petsc_adjoint=True)
        
        # Mapping from node to cell, may need further check (weipeng) -- '1*v*dx'
        # (1, num_quads, num_nodes) * (num_cells, num_quads,1) --> (num_cells,num_nodes)
        weight_n2c_map = onp.sum(problem.fes[0].shape_vals[None,:,:] * 
                                 problem.fes[0].JxW[:,:,None],axis=1)
        
        
        def update_filter_fns(rmin):
            
            Rmin = rmin/(2.*np.sqrt(3))
            
            def filter_fns(x):
                # (num_nodes, 1)
                xf = fwd_pred([x, Rmin])[0]
                xf = np.sum(np.take(xf, problem.fes[0].cells)
                                         *weight_n2c_map,axis=1)
                return xf.reshape(-1,1)
            
            self.compute_density = filter_fns
            
            
        
        self.update_filter_fns = update_filter_fns
        
        self.rmin = rmin
        self.update_filter_fns(self.rmin)



class HelmholtzPDE(Problem):
    
    def get_universal_kernel(self):
           
        def universal_kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW, cell_v_grads_JxW, cell_psi, R_min):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_JxW: (num_vars, num_quads)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            
            ## Split
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat) 
            cell_sol = cell_sol_list[0]
            cell_shape_grads_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads = cell_shape_grads_list[0]
            cell_v_grads_JxW_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW = cell_v_grads_JxW_list[0]
            cell_JxW = cell_JxW[0]
            
            ## Handles the term 'R_min^2 * inner(grad(psi_f),grad(v)) * dx'
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim) 
            # -> (num_quads, vec, dim)
            psi_f_grad = np.sum(cell_sol[None,:,:,None] * cell_shape_grads[:,:,None,:],axis=1)
            # (num_quads, 1, vec, dim) * (num_quads, num_nodes, 1, dim) ->  (num_nodes, vec) 
            val1 = np.sum(R_min**2 * psi_f_grad[:,None,:,:] * cell_v_grads_JxW,axis=(0,-1))
            
            ## Handles the term 'inner(psi_f,v)*dx'
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, vec) 
            psi_f = np.sum(cell_sol[None,:,:] * self.fes[0].shape_vals[:,:,None],axis=1)
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec) 
            val2 = np.sum(psi_f[:,None,:] * self.fes[0].shape_vals[:,:,None] * cell_JxW[:,None,None],axis=0)
            
            ## Handles the term 'inner(psi,v) * dx'
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec) 
            val3 = np.sum(cell_psi[:,None,:] * self.fes[0].shape_vals[:,:,None] * cell_JxW[:,None,None],axis=0)
            
            weak_form = [val1 + val2 - val3]
            
            return jax.flatten_util.ravel_pytree(weak_form)[0]
        
        return universal_kernel
    
    def set_params(self, params):
        theta, R_min = params
        # (num_cells, num_quads, 1)
        self.internal_vars = [np.repeat(theta[:,None,:],repeats=self.fes[0].num_quads,axis=1),
                              R_min * np.ones((self.fes[0].num_cells,1))]
        
        

# Convolution type filter

class ConvFilter:
    
    def __init__(self, problem, rmin):
        self.citation = 'Sigmund1997'
        # H, Hs = compute_filter_kd_tree(problem.fes[0], rmin)
        H, Hs = global_search(problem.fes[0], rmin)
        self.compute_sensitivity = lambda dc, x: H@(dc * x)/Hs[:,None]/onp.maximum(x,1e-3)
        self.compute_density = lambda x: H@x/Hs


def compute_filter_kd_tree(fe, rmin):
    """
    Copied and modified from tianju's original codes
    
    This function is created by Tianju. Not from the original code.
    We use k-d tree algorithm to compute the filter.
    """
    cell_centroids = onp.mean(onp.take(fe.points, fe.cells, axis=0), axis=1)
    flex_num_cells = len(fe.flex_inds)
    flex_cell_centroids = onp.take(cell_centroids, fe.flex_inds, axis=0)

    V = onp.sum(fe.JxW)
    avg_elem_V = V/fe.num_cells

    avg_elem_size = avg_elem_V**(1./fe.dim)

    kd_tree = scipy.spatial.KDTree(flex_cell_centroids)
    I = []
    J = []
    V = []
    for i in range(flex_num_cells):
        num_nbs = int((2*(rmin/avg_elem_size+1) + 1)**2)
        dd, ii = kd_tree.query(flex_cell_centroids[i], num_nbs)
        neighbors = onp.take(flex_cell_centroids, ii, axis=0)
        vals = onp.where(rmin - dd > 0., rmin - dd, 0.)
        I += [i]*num_nbs
        J += ii.tolist()
        V += vals.tolist()
    H_sp = scipy.sparse.csc_array((V, (I, J)), shape=(flex_num_cells, flex_num_cells))

    # TODO(Tianju): No need to create the full matrix. 
    # Will cause memory issue for large size problem.
    # High priority!

    # H = H_sp.todense()
    # Hs = onp.sum(H, 1)
    
    H = H_sp
    Hs = H.sum(1)
    
    return H, Hs


def global_search(fe, rmin):
    '''
    Only for structured QUAD4 mesh, as in the original paper (2011.88-line)
    
    '''
    
    def unpack_mesh(num_cells, num_nodes):
        
        # nelx*nely = num_cells
        # (nelx+1)*(nely+1) = num_nodes
        
        a = 1
        b = (num_cells+1-num_nodes)
        c = num_cells
        
        nelx = int(onp.maximum((-b+onp.sqrt(b**2-4*a*c))/(2*a), (-b-onp.sqrt(b**2-4*a*c))/(2*a)))
        nely = int(num_cells/nelx)
        
        assert nelx*nely == num_cells, 'Wrong estimation of nelx and nely!'
        
        return nelx, nely
    
    nelx, nely = unpack_mesh(fe.num_cells, fe.num_total_nodes)
    
    
    # from topopt.py 
    # see https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python
    nfilter = int(nelx * nely * ((2 * (onp.ceil(rmin) - 1) + 1) ** 2))
    iH = onp.zeros(nfilter)
    jH = onp.zeros(nfilter)
    sH = onp.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(onp.maximum(i - (onp.ceil(rmin) - 1), 0))
            kk2 = int(onp.minimum(i + onp.ceil(rmin), nelx))
            ll1 = int(onp.maximum(j - (onp.ceil(rmin) - 1), 0))
            ll2 = int(onp.minimum(j + onp.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - onp.sqrt(((i - k) * (i - k) + (j - l) * (j - l)))
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = onp.maximum(0.0, fac)
                    cc = cc + 1
    
    # Finalize assembly and convert to csc format
    H = scipy.sparse.csc_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).todense()
    Hs = H.sum(1)
    
    return np.array(H), np.array(Hs)
