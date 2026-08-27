'''

2D MBB beam 

See https://link.springer.com/article/10.1007/s00158-010-0594-7

'''

import jax
import jax.numpy as np

from jax_fem.problem import Problem
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem.solver import ad_wrapper

class Elasticity(Problem):
    
    def custom_init(self, xPhys2Mat):
        self.fe = self.fes[0]
        self.fe.flex_inds = np.arange(len(self.fe.cells))
        self.xPhys2Mat = xPhys2Mat

    def get_tensor_map(self):
        def stress(u_grad, xPhys):
            # SIMP model
            E = self.xPhys2Mat(xPhys)
            nu = 0.3
            # Plane strain
            mu = E/(2.*(1.+nu))
            lmbda = E*nu/((1+nu)*(1-2*nu)) 
            # Plane strain -> Plane Stress
            lmbda = 2*mu*lmbda/(lmbda+2*mu) 
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 1.])
        return [surface_map]

    def set_params(self, params):
        # Override base class method.
        full_params = np.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = np.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]

    def compute_compliance(self, sol):
        # Surface integral
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec) * # (num_selected_faces, num_face_quads, num_nodes, 1)
        u_face = sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :] * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        u_face = np.sum(u_face, axis=2) # (num_selected_faces, num_face_quads, vec)
        # (num_selected_faces, num_face_quads, dim)
        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        traction = -jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points) # (num_selected_faces, num_face_quads, vec)
        val = np.sum(traction * u_face * nanson_scale[:, :, None])
        return val


def prep_fem(Nx, Ny, Lx, Ly, xPhys2Mat):
    
    # Mesh
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx, Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type], ele_type)
    
    # BCs
    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right_bottom_corner(point):
        return np.logical_and(np.isclose(point[0], Lx, atol=1e-5), 
                              np.isclose(point[1], 0., atol=1e-5))

    def left_middle(point):
        return np.logical_and(np.isclose(point[0], 0, atol = Lx/Nx + 1e-5), 
                              np.isclose(point[1], Ly, atol=1e-5))
    
    def dirichlet_val(point):
        return 0.
    
    dirichlet_bc_info = [[left, right_bottom_corner], 
                         [0, 1], [dirichlet_val]*2]
    location_fns = [left_middle]
    
    # Problem
    problem = Elasticity(mesh, vec=2, dim=2, ele_type=ele_type, 
                         dirichlet_bc_info=dirichlet_bc_info, 
                         location_fns=location_fns,
                         additional_info=(xPhys2Mat,))
    
    # Differentiable wrapper
    solver_options = {'petsc_solver':{'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, 
                          solver_options=solver_options,
                          adjoint_solver_options=solver_options)
    
    return fwd_pred, problem
    
    