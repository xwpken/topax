import jax.numpy as np
import numpy as onp
import scipy
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper


def build_conv_filter(problem, rmin=1.5, method="global_search"):
    """Build the dense matrix of a linear convolution filter.

    Parameters
    ----------
    problem : Problem
        JAX-FEM problem whose first finite element defines the design mesh.
    rmin : float
        Filter radius.
    method : {"global_search", "kd_tree"}
        Neighborhood construction method.

    Returns
    -------
    H : ndarray
        Filter-weight matrix.
    Hs : ndarray
        Row sums of ``H`` with shape ``(num_design, 1)``.
    """
    fe = problem.fes[0]
    if method == "global_search":
        H, Hs = _build_regular_filter(fe, rmin)
    elif method == "kd_tree":
        H, Hs = _build_kd_filter(fe, rmin)
    else:
        raise ValueError(f"Unknown convolution-filter method: {method}")
    return np.asarray(H), np.asarray(Hs)


def _build_regular_filter(fe, rmin):
    num_cells = fe.num_cells
    num_nodes = fe.num_total_nodes
    b = num_cells + 1 - num_nodes
    discriminant = b**2 - 4 * num_cells
    roots = (
        (-b + onp.sqrt(discriminant)) / 2,
        (-b - onp.sqrt(discriminant)) / 2,
    )
    nelx = int(max(roots))
    nely = int(num_cells / nelx)
    if nelx * nely != num_cells:
        raise ValueError("global_search requires a structured quadrilateral mesh")

    radius = int(onp.ceil(rmin))
    rows = []
    cols = []
    values = []

    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            for k in range(max(i - radius + 1, 0), min(i + radius, nelx)):
                for ell in range(max(j - radius + 1, 0), min(j + radius, nely)):
                    weight = max(0.0, rmin - onp.hypot(i - k, j - ell))
                    rows.append(row)
                    cols.append(k * nely + ell)
                    values.append(weight)

    H = scipy.sparse.csc_matrix(
        (values, (rows, cols)), shape=(num_cells, num_cells)
    ).toarray()
    Hs = H.sum(axis=1, keepdims=True)
    return H, Hs


def _build_kd_filter(fe, rmin):
    centroids = onp.mean(onp.take(fe.points, fe.cells, axis=0), axis=1)
    flex_inds = onp.asarray(fe.flex_inds)
    flex_centroids = onp.take(centroids, flex_inds, axis=0)

    domain_volume = onp.sum(fe.JxW)
    mean_cell_volume = domain_volume / fe.num_cells
    mean_cell_size = mean_cell_volume ** (1.0 / fe.dim)
    neighbor_count = int((2 * (onp.ceil(rmin / mean_cell_size) + 1) + 1) ** fe.dim)
    neighbor_count = min(neighbor_count, len(flex_inds))

    tree = scipy.spatial.KDTree(flex_centroids)
    rows = []
    cols = []
    values = []
    for row, centroid in enumerate(flex_centroids):
        distances, indices = tree.query(centroid, neighbor_count)
        distances = onp.atleast_1d(distances)
        indices = onp.atleast_1d(indices)
        weights = onp.maximum(rmin - distances, 0.0)
        rows.extend([row] * neighbor_count)
        cols.extend(indices.tolist())
        values.extend(weights.tolist())

    num_flex = len(flex_inds)
    H = scipy.sparse.csc_matrix(
        (values, (rows, cols)), shape=(num_flex, num_flex)
    ).toarray()
    Hs = H.sum(axis=1, keepdims=True)
    return H, Hs


class HelmholtzFilter:
    """Helmholtz PDE filter by Lazarov and Sigmund.

    Parameters
    ----------
    problem : Problem
        JAX-FEM problem defining the mesh.
    rmin : float
        Filter radius.
    """

    def __init__(self, problem, rmin):
        self.problem = _HelmholtzPDE(
            problem.mesh[0],
            vec=1,
            dim=problem.dim,
            ele_type=problem.fes[0].ele_type,
        )
        solver_options = {"petsc_solver": {}}
        self.fwd_pred = ad_wrapper(
            self.problem,
            solver_options=solver_options,
            adjoint_solver_options=solver_options,
        )
        fe = self.problem.fes[0]
        self.node_to_cell_weights = onp.sum(
            fe.shape_vals[None, :, :] * fe.JxW[:, :, None], axis=1
        )
        self.cell_volumes = onp.sum(fe.JxW, axis=1, keepdims=True)
        self.rmin = rmin

    def __call__(self, x):
        """Filter a cell-density field and return cell densities."""
        radius = self.rmin / (2.0 * np.sqrt(3.0))
        filtered = self.fwd_pred([x, radius])[0]
        nodal_values = np.take(filtered.reshape(-1), self.problem.fes[0].cells)
        cell_values = np.sum(
            nodal_values * self.node_to_cell_weights,
            axis=1,
            keepdims=True,
        )
        return cell_values / self.cell_volumes


class _HelmholtzPDE(Problem):
    def get_universal_kernel(self):
        def universal_kernel(
            cell_sol_flat,
            x,
            cell_shape_grads,
            cell_JxW,
            cell_v_grads_JxW,
            cell_psi,
            radius,
        ):
            cell_sol = self.unflatten_fn_dof(cell_sol_flat)[0]
            start = self.num_nodes_cumsum[0]
            end = self.num_nodes_cumsum[1]
            shape_grads = cell_shape_grads[:, start:end, :]
            test_grads = cell_v_grads_JxW[:, start:end, :, :]
            quadrature_weights = cell_JxW[0]

            psi_grad = np.sum(
                cell_sol[None, :, :, None] * shape_grads[:, :, None, :],
                axis=1,
            )
            diffusion = np.sum(
                radius**2 * psi_grad[:, None, :, :] * test_grads,
                axis=(0, -1),
            )
            psi = np.sum(
                cell_sol[None, :, :] * self.fes[0].shape_vals[:, :, None],
                axis=1,
            )
            reaction = np.sum(
                psi[:, None, :]
                * self.fes[0].shape_vals[:, :, None]
                * quadrature_weights[:, None, None],
                axis=0,
            )
            source = np.sum(
                cell_psi[:, None, :]
                * self.fes[0].shape_vals[:, :, None]
                * quadrature_weights[:, None, None],
                axis=0,
            )
            return (diffusion + reaction - source).reshape(-1)

        return universal_kernel

    def set_params(self, params):
        theta, radius = params
        self.internal_vars = [
            np.repeat(theta[:, None, :], repeats=self.fes[0].num_quads, axis=1),
            radius * np.ones((self.fes[0].num_cells, 1)),
        ]
