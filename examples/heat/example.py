"""

2D heat conduction topology optimization

Minimize thermal compliance (dissipated power) with material volume constraints

"""

import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)

import numpy as onp
import matplotlib.pyplot as plt

from jax_fem import logger
logger.setLevel("WARNING")

from topax.top import Density
from topax.mat import SIMP
from topax.tfp import Conv
from topax.opt import OC

from problem import prep_fem

#%% SETUP
vf = 0.4
rmin = 2.0

xPhys2Mat = SIMP(E_max=1, E_min=1e-9, penal=3)

Nx, Ny = 60, 60
Lx, Ly = Nx, Ny
fwd_pred, problem = prep_fem(Nx, Ny, Lx, Ly, xPhys2Mat)


def J_total(xPhys):
    sol = fwd_pred(xPhys)[0]
    fe = problem.fe
    T_quad = np.sum(sol[fe.cells] * fe.shape_vals[None, :, :], axis=1, keepdims=True)
    return np.sum(T_quad * fe.JxW[:, :, None])


def volume_constraint(xPhys):
    return np.sum(xPhys) - vf * xPhys.size


conv2d = Conv(problem, rmin=rmin)
sens_filter = conv2d.sensitivity()

optimizer = OC(move=0.2, damping=0.5)

x0 = vf * np.ones((Nx * Ny, 1))
topo = Density(problem,
              x=x0, transform=conv2d.density(),
              obj=J_total, cons=[volume_constraint])


#%% OPTIMIZATION LOOP
loop = 0
change = 1
xnew = x0
while change > 0.01 and loop < 500:
    loop += 1
    J, dJ, c, dc = topo.eval()
    dc = dc[0]
    xold = topo.x.copy()
    dJ = sens_filter(dJ, xold)
    xnew = optimizer.update(topo, dJ, dc, vf)
    topo.update(xnew)
    vol = topo.compute_vf()
    change = np.max(np.abs(xnew - xold))
    print(f' It.:{loop:5d}, Obj.:{J:11.4f}, Vol.:{vol:7.3f}, ch.:{change:7.3f}')
    field = onp.flip(topo.x_phys.reshape(Ny, Nx, order='F'), axis=0)
    plt.imshow(field, cmap='hot_r', vmin=0, vmax=1)
    plt.axis('equal')
    plt.axis('off')
    plt.pause(0.01)
