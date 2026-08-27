'''
2D hyperelasticity with geometric nonlinearity

See Wang et al., "Interpolation scheme for fictitious domain techniques and
topology optimization of finite strain elastic problems", CMAME, 2014.

'''
import jax
import jax.numpy as np

from jax_fem.problem import Problem
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem.solver import ad_wrapper

from topax.mat import EnergyInterp


class Hyperelastic(Problem):

    def custom_init(self, xPhys2Mat):
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))
        self.xPhys2Mat = xPhys2Mat
        self.energy = EnergyInterp()

    def get_tensor_map(self):
        def stress(u_grad, theta):
            rho = theta[0]
            penal = self.xPhys2Mat.penal
            E = self.xPhys2Mat(rho)
            nu = 0.4
            mu = E / (2. * (1. + nu))
            lam = E * nu / ((1. + nu) * (1. - 2. * nu))

            def PK1_stress(F):
                I = np.eye(F.shape[-1])
                E_gl = 0.5 * (F.T @ F - I)
                S = lam * np.trace(E_gl) * I + 2. * mu * E_gl
                return F @ S

            return self.energy(u_grad, rho, penal, mu, lam, PK1_stress)
        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 0.012])
        return [surface_map]

    def set_params(self, params):
        full_params = np.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]

    def compute_compliance(self, sol):
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        u_face = sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :] \
                 * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        u_face = np.sum(u_face, axis=2)
        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        traction = -jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points)
        val = np.sum(traction * u_face * nanson_scale[:, :, None])
        return val


def prep_fem(Nx, Ny, Lx, Ly, xPhys2Mat):
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def fixed_location(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def load_location(point):
        return np.logical_and(
            np.isclose(point[0], Lx, atol=1e-5),
            np.isclose(point[1], Ly/2, atol=2*(Ly/Ny) + 1e-5),
        )

    def dirichlet_val(point):
        return 0.

    dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]
    location_fns = [load_location]

    problem = Hyperelastic(
        mesh, vec=2, dim=2, ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
        additional_info=(xPhys2Mat,),
    )

    fwd_pred = ad_wrapper(
        problem,
        solver_options={'newton': {'linear': {'petsc_solver': {}}}},
        adjoint_solver_options={'petsc_solver': {}},
    )

    return fwd_pred, problem
