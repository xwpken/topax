"""
C-shape forward analysis (Wang et al. 2014, Section 2, Fig. 1)

Geometry: 10 m x 10 m domain, plane stress
    C-shaped solid with 1 m thick arms (top, bottom, left)
Mesh: 10 x 10 QUAD4 (100 elements)
Material: E = 1, nu = 0.3, E_min = 1e-9
BC: fixed left edge
Loads: f1 = 0.002 (downward) at top-right free end
       f2 = 0.003 (rightward) at bottom-right free end
"""

import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)

import numpy as onp
import matplotlib.pyplot as plt

from jax_fem.problem import Problem
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh
from jax_fem.solver import ad_wrapper

from topax.mat import SIMP, EnergyInterp


class CShape(Problem):
    """St. Venant-Kirchhoff C-shape with optional energy interpolation."""

    def custom_init(self, xPhys2Mat, use_interp):
        self.fe = self.fes[0]
        self.xPhys2Mat = xPhys2Mat
        self.use_interp = use_interp
        if use_interp:
            self.energy = EnergyInterp()

    def get_surface_maps(self):
        # Traction direction: top pulled down, bottom pulled right.
        # t comes via internal_vars_surfaces from set_params.
        return [
            lambda u, x, t: np.array([0., t]),
            lambda u, x, t: np.array([-t, 0.]),
        ]

    def set_params(self, params):
        # params = (density, t_top, t_bot)
        rho, t_top, t_bot = params
        self.internal_vars = [
            np.repeat(rho[None], self.fe.num_quads, axis=0).transpose(1, 0, 2)
        ]
        nq = self.fe.num_face_quads
        self.internal_vars_surfaces = [
            (t_top * np.ones((len(self.boundary_inds_list[0]), nq)),),
            (t_bot * np.ones((len(self.boundary_inds_list[1]), nq)),),
        ]

    def get_tensor_map(self):
        def stress(u_grad, theta):
            rho = theta[0]
            penal = self.xPhys2Mat.penal
            E = self.xPhys2Mat(rho)
            nu = 0.3

            # St. Venant-Kirchhoff with plane stress assumption
            mu = E / (2. * (1. + nu))
            lam = E * nu / ((1. + nu) * (1. - 2. * nu))
            lam = 2. * mu * lam / (lam + 2. * mu)

            def PK1_stress(F):
                E_gl = 0.5 * (F.T @ F - np.eye(F.shape[-1]))
                S = lam * np.trace(E_gl) * np.eye(F.shape[-1]) + 2. * mu * E_gl
                return F @ S

            # Energy interpolation
            if self.use_interp:
                return self.energy(u_grad, rho, penal, mu, lam, PK1_stress)
            return PK1_stress(np.eye(u_grad.shape[-1]) + u_grad)

        return stress


def prep_fem(Nx, Ny, Lx, Ly, xPhys2Mat, interp=True):
    ele_type = 'QUAD4'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    y_tol = Ly / Ny + 1e-5

    # Loaded edges: top-right and bottom-right of the C-shape
    location_fns = [
        lambda p: np.logical_and(
            np.isclose(p[0], Lx, atol=1e-5), np.isclose(p[1], Ly, atol=y_tol)),
        lambda p: np.logical_and(
            np.isclose(p[0], Lx, atol=1e-5), np.isclose(p[1], 0., atol=y_tol)),
    ]

    def fixed_location(point):
        return np.isclose(point[0], 0., atol=1e-5)

    problem = CShape(
        mesh, vec=2, dim=2, ele_type=ele_type,
        dirichlet_bc_info=[[fixed_location]*2, [0, 1], [lambda p: 0.]*2],
        location_fns=location_fns,
        additional_info=(xPhys2Mat, interp),
    )

    fwd_pred = ad_wrapper(
        problem,
        solver_options={'newton': {'linear': {'spsolve_solver': {}}}},
        adjoint_solver_options={'spsolve_solver': {}},
    )

    return fwd_pred, meshio_mesh.points, problem

if __name__ == '__main__':
#%% SETUP
    Nx, Ny = 10, 10
    Lx, Ly = 10.0, 10.0
    xPhys2Mat = SIMP(E_max=1.0, E_min=1e-9, penal=3)

    # C-shape
    rho = onp.ones((Nx, Ny))
    rho[1:, 1:-1] = 1e-9
    rho = onp.ascontiguousarray(rho.flatten()[:, None])

    cases = [('f1=0.002, f2=0.003', 0.002, 0.003),
             ('f1=0.018, f2=0.027 (9x)', 0.018, 0.027)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    kw = dict(cmap='gray_r', vmin=0, vmax=1, edgecolors='k', linewidth=0.3)

#%% SOLVE AND PLOT
    for j, interp in enumerate([False, True]):
        name = 'Direct' if not interp else 'Interp'
        fwd_pred, points, _ = prep_fem(Nx, Ny, Lx, Ly, xPhys2Mat, interp=interp)
        for idx, (label, t_top, t_bot) in enumerate(cases):
            sol = fwd_pred((rho, t_top, t_bot))[0]
            ax = axes[idx, j]
            X = points[:, 0].reshape(Nx + 1, Ny + 1).T
            Y = points[:, 1].reshape(Nx + 1, Ny + 1).T
            C = rho[:, 0].reshape(Nx, Ny).T
            X += sol[:, 0].reshape(Nx + 1, Ny + 1).T
            Y += sol[:, 1].reshape(Nx + 1, Ny + 1).T
            ax.pcolormesh(X, Y, C, **kw)
            ax.axis('equal')
            ax.set_title(f'{name} ({label})')

    plt.tight_layout()
    plt.show()
