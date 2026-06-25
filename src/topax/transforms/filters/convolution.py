from functools import singledispatchmethod

import jax.numpy as np
import numpy as onp
import scipy

from ..base import Transform

class ConvolutionFilter(Transform):
    """Convolution filter"""
    
    def __init__(self, problem, options: dict = None):
        options = options or {}
        rmin = options.get('rmin', 1.5)
        method = options.get('method', 'global_search')
        
        super().__init__(name=f"ConvolutionFilter_rmin_{rmin}")
        self.problem = problem
        self._rmin = rmin
        self._method = method
        # Precompute filter matrix
        self.H, self.Hs = self._compute_filter_matrix(problem.fes[0], rmin, method)
        
        self._citation = 'Sigmund1997' 
        
    @singledispatchmethod
    def __call__(self, arg):
        """Apply convolution filter based on input type"""
        raise TypeError(f"Unsupported argument type: {type(arg)}")
    
    @__call__.register
    def _(self, x: np.ndarray):
        """Density filtering: x_filtered = H @ (x/ Hs)"""
        x_col = x.reshape(-1, 1)
        x_filtered = (self.H @ x_col)/ self.Hs
        return x_filtered.reshape(x.shape)
    
    @__call__.register
    def _(self, args: tuple):
        """Sensitivity filtering: dc_filtered = H @ (dc * x) / (Hs * max(x, 1e-3))"""
        assert len(args) == 2, "Input should be (dc, x)"
        
        dc, x = args
        dc_col = dc.reshape(-1, 1)
        x_col = x.reshape(-1, 1)
        
        numerator = self.H @ (dc_col * x_col)
        denominator = self.Hs * np.maximum(x_col, 1e-3)
        
        return (numerator / denominator).reshape(dc.shape)
    
    def _compute_filter_matrix(self, fe, rmin, method):
        """Compute filter matrix using specified method"""
        if method == 'kd_tree':
            return self._compute_filter_kd_tree(fe, rmin)
        elif method == 'global_search':
            return self._compute_filter_global_search(fe, rmin)
        else:
            raise ValueError(f"Unknown filter method: {method}")
    
    def _compute_filter_kd_tree(self, fe, rmin):
        """Compute filter using KD-tree algorithm"""
        cell_centroids = onp.mean(onp.take(fe.points, fe.cells, axis=0), axis=1)
        flex_num_cells = len(fe.flex_inds)
        flex_cell_centroids = onp.take(cell_centroids, fe.flex_inds, axis=0)

        V = onp.sum(fe.JxW)
        avg_elem_V = V / fe.num_cells
        avg_elem_size = avg_elem_V ** (1.0 / fe.dim)

        kd_tree = scipy.spatial.KDTree(flex_cell_centroids)
        I, J, V_vals = [], [], []
        
        safety_margin = 1
        num_nbs = int((2 * (onp.ceil(rmin / avg_elem_size) + safety_margin) + 1) ** fe.dim)
        
        for i in range(flex_num_cells):
            dd, ii = kd_tree.query(flex_cell_centroids[i], num_nbs)
            vals = onp.where(rmin - dd > 0.0, rmin - dd, 0.0)
            I.extend([i] * num_nbs)
            J.extend(ii.tolist())
            V_vals.extend(vals.tolist())
        
        H_sp = scipy.sparse.csc_array((V_vals, (I, J)), 
                                    shape=(flex_num_cells, flex_num_cells))
        H = H_sp.todense()
        Hs = (onp.sum(H, 1)).reshape(-1,1)
        
        return H, Hs
    
    def _compute_filter_global_search(self, fe, rmin):
        
        """Compute filter using global search (for structured QUAD4 mesh)"""
        
        def estimate_mesh_size(num_cells, num_nodes):
            a = 1
            b = (num_cells + 1 - num_nodes)
            c = num_cells
            nelx = int(onp.maximum((-b + onp.sqrt(b**2 - 4*a*c)) / (2*a), 
                                   (-b - onp.sqrt(b**2 - 4*a*c)) / (2*a)))
            nely = int(num_cells / nelx)
            assert nelx * nely == num_cells
            return nelx, nely
        
        nelx, nely = estimate_mesh_size(fe.num_cells, fe.num_total_nodes)
        
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
                        cc += 1
        
        H = scipy.sparse.csc_matrix((sH, (iH, jH)), 
                                  shape=(nelx * nely, nelx * nely)).todense()
        Hs = (H.sum(1)).reshape(-1,1)
        
        return H, Hs
    
    @property
    def radius(self):
        return self._rmin
    
    @property
    def citation(self):
        """Get citation information"""
        return self._citation
    
    @property
    def method(self):
        return self._method
    
    def get_config(self):
        """Get filter configuration"""
        return {
            'radius': self.radius,
            'method': self.method,
            'citation': self.citation
        }
    
    def __repr__(self):
        return f"ConvolutionFilter(radius={self.radius}, matrix='{self.method}')"