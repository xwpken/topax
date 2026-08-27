"""Compliant mechanism (force inverter) — half model using symmetry."""

import jax.numpy as np

from jax_fem.problem import Problem
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem.solver import ad_wrapper


class CompliantMechanism(Problem):

    def custom_init(self, xPhys2Mat):
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))
        self.xPhys2Mat = xPhys2Mat
        Lx = self.fe.points[:, 0].max()
        node_mask = np.all(
            np.isclose(self.fe.points, np.array([Lx, 0.0]), atol=1e-5), axis=1
        )
        self.output_node = int(np.argwhere(node_mask)[0, 0])

    def get_tensor_map(self):
        def stress(u_grad, xPhys):
            E = self.xPhys2Mat(xPhys)
            nu = 0.3
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2*nu))
            lmbda = 2*mu*lmbda/(lmbda+2*mu)
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress

    def get_surface_maps(self):
        def input_traction(u, x):
            return np.array([2., 0.])
        return [input_traction]

    def set_params(self, params):
        full_params = np.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]


def prep_fem(Nx, Ny, Lx, Ly, xPhys2Mat, k_out=0.1):
    # k_out reserved for future spring implementation
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx, Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type], ele_type)

    fixed_y0 = Ly - 2. * Ly / Ny
    def left_fixed_top(point):
        on_left = np.isclose(point[0], 0., atol=1e-5)
        on_top = np.greater(point[1], fixed_y0)
        return np.logical_and(on_left, on_top)

    def symmetry(point):
        return np.isclose(point[1], 0., atol=1e-5)

    dirichlet_bc_info = [
        [left_fixed_top, left_fixed_top, symmetry],
        [0, 1, 1],
        [lambda p: 0.] * 3,
    ]

    def input_loc(point):
        return np.logical_and(
            np.isclose(point[0], 0., atol=1e-5),
            np.isclose(point[1], 0., atol=Ly/Ny + 1e-5),
        )

    location_fns = [input_loc]

    problem = CompliantMechanism(
        mesh, vec=2, dim=2, ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
        additional_info=(xPhys2Mat,),
    )

    fwd_pred = ad_wrapper(
        problem,
        solver_options={'petsc_solver': {}},
        adjoint_solver_options={'petsc_solver': {}},
    )

    return fwd_pred, problem
