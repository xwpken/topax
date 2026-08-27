"""

2D heat conduction problem

Minimize thermal compliance (dissipated power) with material volume constraints

"""

import jax.numpy as np

from jax_fem.problem import Problem
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem.solver import ad_wrapper


class HeatConduction(Problem):

    def custom_init(self, xPhys2Mat):
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))
        self.xPhys2Mat = xPhys2Mat

    def get_tensor_map(self):
        def heat_flux(u_grad, xPhys):
            k = self.xPhys2Mat(xPhys)
            return k * u_grad
        return heat_flux

    def get_mass_map(self):
        def source(T, x, *args):
            return -np.ones_like(T)
        return source

    def set_params(self, params):
        full_params = np.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]


def prep_fem(Nx, Ny, Lx, Ly, xPhys2Mat):
    
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx, Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type], ele_type)

    def left_middle(point):
        return np.logical_and(np.isclose(point[0], 0., atol=Lx/Nx+1e-5),
                              np.isclose(point[1], Ly/2., atol=Ly/Ny+1e-5))

    problem = HeatConduction(mesh, vec=1, dim=2, ele_type=ele_type,
                             dirichlet_bc_info=[[left_middle], [0], [lambda p: 0.]],
                             additional_info=(xPhys2Mat,))

    fwd_pred = ad_wrapper(problem,
                          solver_options={'petsc_solver': {}},
                          adjoint_solver_options={'petsc_solver': {}})
    
    return fwd_pred, problem