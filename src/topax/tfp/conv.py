import jax.numpy as np
import numpy as onp
import scipy

from .base import Transform


class Conv:

    """Convolution filter matrix builder

    Builds the filter matrix H and row-sum Hs for a given mesh and
    filter radius.  Use ``density()`` and ``sensitivity()`` to obtain
    ready-to-use filter objects.
    """

    def __init__(self, problem, rmin=1.5, method='global_search'):
        self.rmin = rmin
        self.method = method
        self.H, self.Hs = self._build_H(problem.fes[0], rmin, method)
        self._citation = (
            "Sigmund, O. (2001) A 99 line topology optimization code "
            "written in Matlab. SMO, 21(2), 120-127"
        )

    def density(self):
        """Return a DensityFilter (Transform) for use in a Pipeline."""
        return DensityFilter(self.H, self.Hs, self._citation)

    def sensitivity(self):
        """Return a SensitivityFilter for filtering sensitivities."""
        return SensitivityFilter(self.H, self.Hs, self._citation)

    def _build_H(self, fe, rmin, method):
        if method == 'kd_tree':
            return self._build_kd(fe, rmin)
        elif method == 'global_search':
            return self._build_global(fe, rmin)
        else:
            raise ValueError(f"Unknown filter method: {method}")

    def _build_global(self, fe, rmin):
        """Compute filter matrix for a regular QUAD4 mesh"""

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
        Hs = (H.sum(1)).reshape(-1, 1)
        return H, Hs

    def _build_kd(self, fe, rmin):
        """Compute filter matrix using KD-tree for unstructured meshes"""
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
        Hs = (onp.sum(H, 1)).reshape(-1, 1)
        return H, Hs

    def __repr__(self):
        return (
            f"Conv(rmin={self.rmin}, method='{self.method}')\n"
            f"  citation: {self._citation}"
        )


class DensityFilter(Transform):

    """Density filter (Transform), callable as density_filter(x)."""

    def __init__(self, H, Hs, citation):
        self.H = H
        self.Hs = Hs
        self._citation = citation

    def __call__(self, x):
        x_col = x.reshape(-1, 1)
        x_filtered = (self.H @ x_col) / self.Hs
        return x_filtered.reshape(x.shape)

    def __repr__(self):
        return (
            f"DensityFilter(shape={self.H.shape})\n"
            f"  citation: {self._citation}"
        )


class SensitivityFilter:

    """Sensitivity filter, callable as sens_filter(dc, x)."""

    def __init__(self, H, Hs, citation):
        self.H = H
        self.Hs = Hs
        self._citation = citation

    def __call__(self, dc, x):
        dc_col = dc.reshape(-1, 1)
        x_col = x.reshape(-1, 1)
        numerator = self.H @ (dc_col * x_col)
        denominator = self.Hs * np.maximum(x_col, 1e-3)
        return (numerator / denominator).reshape(dc.shape)

    def __repr__(self):
        return (
            f"SensitivityFilter(shape={self.H.shape})\n"
            f"  citation: {self._citation}"
        )
