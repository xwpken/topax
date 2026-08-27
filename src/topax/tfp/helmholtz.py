import jax
import jax.numpy as np
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
        
        self.citation = (
            "Lazarov, B.S. and Sigmund, O. (2011) Filters in topology "
            "optimization based on Helmholtz-type differential equations. "
            "IJNME, 86(6), 765-781"
        )
        
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
            sg_list = [cell_shape_grads[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :]     
                                     for i in range(self.num_vars)]
            cell_shape_grads = sg_list[0]
            vg_list = [cell_v_grads_JxW[:, self.num_nodes_cumsum[i]: self.num_nodes_cumsum[i+1], :, :]     
                                     for i in range(self.num_vars)]
            cell_v_grads_JxW = vg_list[0]
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
        
        


